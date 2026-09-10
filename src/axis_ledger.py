r"""축마다 **어떤 근거로 그 이름이 됐는가**를 남긴다.

왜 필요한가
-----------
`resolve_shape.stats` 는 규칙별 **집계**만 준다. 그래서 "이 축이 왜 `d_head` 인가" 를 물을 수
없고, 값이 겹치는 축이 조용히 해결된 자리를 못 찾는다. Llama-4-Maverick 은 값 충돌 질문을
0 으로 보고하는데, 실제로는 `d_head`/`E`(둘 다 128)가 **scope 정규식 하나로** 갈린 축이
10,176개다. 그게 맞는 판정이어도 아무도 검증하지 않았고 기록도 없다 -- 모듈 경로가 하나만
어긋나면 경고 없이 뒤집힌다(2026-09-11 실측).

등급
----
외부 검토가 정한 네 등급이다. **`scope_inferred` 는 빈칸이 아니다** -- scope 라는 약한 근거가
있으므로 후보를 보존하되 "확정" 으로 보이지 않게 한다.

    confirmed        소스·앵커·구조가 확인했다
    scope_inferred   scope 정규식만이 후보를 갈랐다
    heuristic        산술로 지어낸 이름이거나 마지막 수단인 재사용이다
    open_tie         갈랐어야 하는데 근거가 없다
    unresolved       이름을 붙일 근거 자체가 없다(정수로 남는다)

`heuristic` 은 외부 검토의 네 등급에 없지만 뺄 수 없다. 값이 겹치지 않아도 `heur_*` 규칙은
**이 seq_len 에서만 참인 이름을 지어낸다**(`heuristic-fabricated-labels`). 확정에 섞으면
지어낸 이름이 "확인됨" 뒤에 숨는다.

세는 단위가 셋이다. 서로 비교하면 안 된다.

    occurrence       CSV/JSONL 에서 영향받는 칸 수
    class            독립적인 의미 축 수
    question         같은 증거 하나로 답할 수 있는 질문 수

메모리
------
V4-Pro 는 축이 66만이다. 축마다 dict 를 만들면 안 된다. 문자열은 intern 해서 정수로 두고,
축당으로는 작은 정수 코드만 갖는다. 상세 chain 은 **값이 겹친 축과 나중에 덮어쓴 축만**
남긴다.
"""
import collections
import json
import os

# 판정 단계. 뒤로 갈수록 나중에 쓴 것이다 -- 최종 라벨은 마지막 writer 의 것이다.
SCOPED_SYMBOL = "scoped_symbol"
PINNED_FROM_OPERAND = "pinned_from_operand"
ANCHOR_DECLARED_WIDTH = "anchor_declared_width"
ANCHOR_PROPAGATED = "anchor_propagated"
STRUCTURAL = "structural"
SOURCE_OVERRIDE = "source_override"

# 등급
CONFIRMED = "confirmed"
SCOPE_INFERRED = "scope_inferred"
OPEN_TIE = "open_tie"
UNRESOLVED = "unresolved"

HEURISTIC = "heuristic"

# 어떤 판정 단계가 "확인됨" 인가. 나머지는 근거의 세기로 따로 정한다.
_CONFIRMING = {PINNED_FROM_OPERAND, ANCHOR_DECLARED_WIDTH, ANCHOR_PROPAGATED,
               STRUCTURAL, SOURCE_OVERRIDE}

# 리졸버의 규칙 이름(`symbolic_shape._r`)을 등급으로 옮긴다. **충돌이 없어도 확정이 아닌
# 것들이 있다** -- 지어낸 이름(`heur_*`)과 마지막 수단인 재사용은 근거가 약하고, `bare` 는
# 이름이 아예 없는 것이다. 이걸 안 가르면 "확정" 안에 지어낸 이름이 숨는다.
_RULE_GRADE = {
    "runtime": CONFIRMED,              # B/T/1 -- 구조가 정한다
    "scoped_symbol": CONFIRMED,
    "scoped_formula": CONFIRMED,
    "derived_formula": CONFIRMED,
    "plain_symbol": CONFIRMED,         # 스코프가 없는 심볼. 겹치지 않으면 확정으로 둔다
    "passthrough": CONFIRMED,
    "out_of_scope_symbol": SCOPE_INFERRED,
    "reused_symbol": HEURISTIC,        # 같은 shape 에서 이미 쓴 이름 재사용 -- 마지막 수단
    "heur_half": HEURISTIC,
    "heur_multiple": HEURISTIC,
    "heur_product": HEURISTIC,
    "bare": UNRESOLVED,                # 이름을 못 붙였다. 정수로 남는다
}


class Ledger:
    """축 자리 -> 판정. 자리는 `(op_id, field, shape_index, axis)` 다."""

    def __init__(self):
        self._interned = {}
        self._names = []
        self.reason = {}          # site -> code id (마지막 writer)
        self.label = {}           # site -> label id
        self.raw_cands = {}       # site -> 후보 튜플 id  (값이 겹친 축만)
        self.scoped_cands = {}    # site -> scope 통과 후보 튜플 id (겹친 축만)
        self.chain = {}           # site -> [(reason id, label id)]  (덮어쓴 축만)

    # --- intern -------------------------------------------------------
    def i(self, s):
        if s is None:
            return -1
        got = self._interned.get(s)
        if got is None:
            got = len(self._names)
            self._interned[s] = got
            self._names.append(s)
        return got

    def s(self, idx):
        return self._names[idx] if 0 <= idx < len(self._names) else None

    # --- 기록 ---------------------------------------------------------
    def record(self, site, label, reason, raw=None, scoped=None):
        """리졸버의 **최초** 판정."""
        self.reason[site] = self.i(reason)
        self.label[site] = self.i(str(label))
        if raw and len(raw) > 1:
            self.raw_cands[site] = self.i("|".join(sorted(raw)))
            self.scoped_cands[site] = self.i("|".join(sorted(scoped or ())))

    def overwrite(self, site, label, reason):
        """앵커·구조·교정이 뒤에 다시 쓴 것. **최초 판정을 지우지 않는다.**"""
        prev = self.chain.get(site)
        if prev is None:
            prev = self.chain[site] = []
            if site in self.reason:
                prev.append((self.reason[site], self.label[site]))
        prev.append((self.i(reason), self.i(str(label))))
        self.reason[site] = self.i(reason)
        self.label[site] = self.i(str(label))

    # --- 등급 ---------------------------------------------------------
    def grade(self, site):
        r = self.s(self.reason.get(site, -1))
        if r in _CONFIRMING:
            return CONFIRMED
        raw = self.raw_cands.get(site)
        if raw is None:
            # 값이 겹치지 않았다. 그래도 **규칙의 세기**로 등급이 갈린다.
            return _RULE_GRADE.get(r, UNRESOLVED)
        scoped = (self.s(self.scoped_cands.get(site, -1)) or "").split("|")
        scoped = [x for x in scoped if x]
        if _RULE_GRADE.get(r) in (HEURISTIC, UNRESOLVED):
            return _RULE_GRADE[r]          # 후보가 있어도 답한 방식이 약하면 그쪽이 맞다
        if len(scoped) == 1:
            return SCOPE_INFERRED
        return OPEN_TIE if len(scoped) > 1 else UNRESOLVED

    def question_key(self, site):
        """같은 증거 하나로 답할 수 있는 질문. 등급이 확정이면 `None`."""
        g = self.grade(site)
        if g not in (SCOPE_INFERRED, OPEN_TIE, HEURISTIC):
            return None
        return (self.s(self.raw_cands.get(site, -1)), self.s(self.label.get(site, -1)), g)

    # --- 보고 ---------------------------------------------------------
    def counts(self):
        occ = collections.Counter()
        qs = collections.Counter()
        for site in self.label:
            g = self.grade(site)
            occ[g] += 1
            q = self.question_key(site)
            if q:
                qs[q] += 1
        return occ, qs

    def coverage_ok(self):
        """등식이 성립하는가. `occurrence = confirmed + scope + open + unresolved`."""
        occ, _ = self.counts()
        return sum(occ.values()) == len(self.label)

    def write(self, model_dir, phase, full_subdir="full"):
        """`full/<phase>.axis_resolution.jsonl`. 확정이 아닌 자리만 적는다."""
        occ, qs = self.counts()
        path = os.path.join(model_dir, full_subdir, f"{phase}.axis_resolution.jsonl")
        n = 0
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"kind": "summary", "occurrences": dict(occ),
                                "questions": len(qs), "sites": len(self.label),
                                "coverage_ok": self.coverage_ok()},
                               ensure_ascii=False) + "\n")
            for (raw, label, g), cnt in qs.most_common():
                f.write(json.dumps({"kind": "question", "grade": g, "candidates": raw,
                                    "label": label, "occurrences": cnt},
                                   ensure_ascii=False) + "\n")
                n += 1
            for site in sorted(self.label):
                g = self.grade(site)
                if g == CONFIRMED:
                    continue
                rec = {"kind": "site", "op_id": site[0], "field": site[1],
                       "shape_index": site[2], "axis": site[3],
                       "grade": g, "label": self.s(self.label[site]),
                       "reason": self.s(self.reason.get(site, -1))}
                if site in self.raw_cands:
                    rec["candidates"] = self.s(self.raw_cands[site])
                    rec["scoped"] = self.s(self.scoped_cands[site])
                if site in self.chain:
                    rec["chain"] = [[self.s(a), self.s(b)] for a, b in self.chain[site]]
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return path, occ, n
