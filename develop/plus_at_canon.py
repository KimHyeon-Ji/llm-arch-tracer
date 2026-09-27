r"""+@ 층의 canonical cell 계층. csv/jsonl 을 **같은 표현**으로 읽고 되쓴다.

왜 있는가
---------
외부 검토(2026-09-27)가 두 가지를 못 박았다.

  1. V1 은 "파일 바이트 동일" 이 아니라 **canonical cell 비교**여야 한다. csv/jsonl 을 다시
     serialize 하면 줄바꿈·공백·인용이 바뀌어 바이트 비교로는 통과할 수 없다.
  2. csv 와 jsonl 을 **같은 파서로 shape 배열화해** 비교해야 한다(문자열 비교 금지).

그래서 이 파일이 유일한 파서·렌더러다. 적용기와 독립 matcher 가 둘 다 이것만 쓴다.

canonical cell key
------------------
    (phase, op_id, field, shape_index, axis)

  field         "input_shape" | "weight_shape" | "output_shape"
  shape_index   input/output 은 피연산자 번호, weight 는 항상 0
  axis          그 shape 안의 축 위치

**왕복 보장**: `render_cell` 이 원본 csv 문자열을 **정확히** 복원하지 못하면 이 모듈을
쓰면 안 된다. `selftest_roundtrip()` 이 발행본 전체에서 그것을 검사한다.

실행:  .venv\Scripts\python.exe develop\plus_at_canon.py <모델디렉터리>
"""
import csv
import io
import json
import os
import sys

FIELDS = ("input_shape", "weight_shape", "output_shape")
FLAT = ("weight_shape",)               # 중첩이 아니라 평평한 shape
csv.field_size_limit(1 << 26)


def parse_csv_shape(text: str, field: str):
    """csv 의 shape 셀 -> shape 목록. 빈 셀은 `[]`.

    input/output:  "[[a, b], [c]]"  -> [["a","b"], ["c"]]
    weight:        "[a, b]"         -> [["a","b"]]   (shape_index 0 하나로 정규화)
    """
    s = (text or "").strip()
    if not s or s == "[]":
        return []
    if field in FLAT:
        inner = s[1:-1] if s.startswith("[") and s.endswith("]") else s
        toks = [t.strip() for t in inner.split(",") if t.strip()]
        return [toks] if toks else []
    if not (s.startswith("[") and s.endswith("]")):
        raise ValueError(f"shape 셀 형식이 아니다: {text!r}")
    body, out, depth, cur = s[1:-1], [], 0, ""
    for ch in body:
        if ch == "[":
            depth += 1
            if depth == 1:
                cur = ""
                continue
        elif ch == "]":
            depth -= 1
            if depth == 0:
                out.append([t.strip() for t in cur.split(",") if t.strip()])
                continue
        if depth >= 1:
            cur += ch
    return out


def render_csv_shape(shapes, field: str) -> str:
    """shape 목록 -> csv 셀 문자열. `parse_csv_shape` 의 정확한 역이다."""
    if not shapes:
        return ""
    if field in FLAT:
        return "[" + ", ".join(str(x) for x in shapes[0]) + "]"
    return "[" + ", ".join(
        "[" + ", ".join(str(x) for x in sh) + "]" for sh in shapes) + "]"


def parse_jsonl_shape(value, field: str):
    """jsonl 의 shape 값 -> 같은 표현. weight 는 평평하므로 한 겹 씌운다."""
    if value is None:
        return []
    if field in FLAT:
        return [[str(x) for x in value]] if value else []
    return [[str(x) for x in sh] for sh in value if isinstance(sh, list)]


def unparse_jsonl_shape(shapes, field: str, original):
    """shape 목록 -> jsonl 값. 원본이 None 이었으면 None 을 유지한다."""
    if original is None:
        return None
    if field in FLAT:
        return [str(x) for x in shapes[0]] if shapes else []
    return [[str(x) for x in sh] for sh in shapes]


def cells_of_row(phase: str, op_id, shapes_by_field: dict):
    """한 행의 canonical cell 들을 (key, value) 로 낸다."""
    for field in FIELDS:
        for si, sh in enumerate(shapes_by_field.get(field) or []):
            for ax, v in enumerate(sh):
                yield (phase, int(op_id), field, si, ax), str(v)


def expand_layers(text):
    """`layers` 열을 실제 층 목록으로 펼친다.

    **범위 표기를 쓴다**: "1-2,4-6,8-10" -> [1,2,4,5,6,8,9,10] (8 개).
    콤마만 쪼개면 3 개로 읽혀 `repeat` 과 안 맞는다 -- 실제로 이 착오로 V9 가 6,342 행을
    거짓 실패시켰다(2026-09-27).
    """
    out = []
    for part in (text or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, _, b = part.partition("-")
            if a.strip().isdigit() and b.strip().isdigit():
                out.extend(range(int(a), int(b) + 1))
                continue
        if part.isdigit():
            out.append(int(part))
    return out


def read_csv_rows(path: str):
    """(fieldnames, rows) -- rows 는 dict 목록. 순서를 보존한다."""
    with io.open(path, encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        return rd.fieldnames, list(rd)


def read_jsonl_rows(path: str):
    out = []
    with io.open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def csv_cells(path: str, phase: str):
    _, rows = read_csv_rows(path)
    out = {}
    for r in rows:
        shapes = {f: parse_csv_shape(r.get(f), f) for f in FIELDS}
        for k, v in cells_of_row(phase, r["op_id"], shapes):
            out[k] = v
    return out


def jsonl_cells(path: str, phase: str):
    out = {}
    for r in read_jsonl_rows(path):
        shapes = {f: parse_jsonl_shape(r.get(f), f) for f in FIELDS}
        for k, v in cells_of_row(phase, r["op_id"], shapes):
            out[k] = v
    return out


def selftest_roundtrip(model_dir: str, phases=("prefill", "decode")):
    """발행본 전체에서 (a) csv 왕복이 정확하고 (b) csv 와 jsonl 의 cell 이 같은지.

    (a) 가 깨지면 이 모듈로 파일을 되쓸 수 없다 -- 비대상 행까지 문자열이 바뀐다.
    (b) 가 깨지면 두 파일이 이미 서로 다르다는 뜻이므로 +@ 를 얹기 전에 멈춰야 한다.
    """
    bad = []
    for ph in phases:
        cpath = os.path.join(model_dir, f"{ph}.csv")
        jpath = os.path.join(model_dir, f"{ph}.jsonl")
        if not (os.path.exists(cpath) and os.path.exists(jpath)):
            bad.append(f"{ph}: 파일 없음")
            continue
        _, rows = read_csv_rows(cpath)
        rt = 0
        for r in rows:
            for f in FIELDS:
                orig = r.get(f) or ""
                back = render_csv_shape(parse_csv_shape(orig, f), f)
                if back != orig.strip():
                    rt += 1
                    if rt <= 3:
                        bad.append(f"{ph} op{r['op_id']} {f}: {orig!r} -> {back!r}")
        if rt:
            bad.append(f"{ph}: csv 왕복 불일치 {rt} 셀")
        cc, jj = csv_cells(cpath, ph), jsonl_cells(jpath, ph)
        if cc != jj:
            only_c = set(cc) - set(jj)
            only_j = set(jj) - set(cc)
            diff = [k for k in set(cc) & set(jj) if cc[k] != jj[k]]
            bad.append(f"{ph}: csv/jsonl cell 불일치 -- csv만 {len(only_c)} "
                       f"jsonl만 {len(only_j)} 값다름 {len(diff)}")
            for k in (list(only_c)[:2] + list(only_j)[:2] + diff[:2]):
                bad.append(f"    {k}  csv={cc.get(k)!r} jsonl={jj.get(k)!r}")
        else:
            print(f"  {ph:<8} cell {len(cc):>7}  왕복 OK, csv==jsonl OK")
    return bad


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    md = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models", "moonshotai__Kimi-K3")
    print(f"canonical 자기검사: {md}")
    bad = selftest_roundtrip(md)
    if bad:
        print("\n**실패**")
        for b in bad:
            print("  " + b)
        return 1
    print("\n통과")
    return 0


if __name__ == "__main__":
    sys.exit(main())
