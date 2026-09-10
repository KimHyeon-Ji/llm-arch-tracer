r"""semantic 이벤트/barrier 의 **작은 고정 사례 5개**. 모델을 안 띄운다.

왜 필요한가
-----------
지금 barrier 검증은 47개 모델 재트레이스(40분)로만 된다. 그래서 설계를 바꿀 때마다
"돌려보고 결과에 맞춰 기준을 세우는" 순서가 된다. 외부 검토가 그걸 막으려고 요구한
다섯 사례를 여기에 못 박는다.

  1. no-op `repeat_kv` (n_rep=1)      -- 같은 텐서가 돌아온다. 이벤트는 남아야 한다.
  2. materialized `repeat_kv` (n_rep>1) -- 실제 ATen 이 돈다. noop=False.
  3. barrier 사이의 **무관한 permute** -- 막으면 안 된다(대조군 먼저 통과).
  4. no-op 뒤 **별칭 재사용**          -- 이벤트 전/후 논리 출처가 갈려야 한다.
  5. **등록 안 된 구현**              -- 축을 추측하지 말고 resolved=False 로 둔다.

3, 4, 5 는 **지금 일부러 빨간불이다**(`XFAIL`). #5(축별 semantic barrier + SemanticSpec 레지스트리)가
끝나면 초록으로 바뀐다. 빨간불이 아니라 초록이 되면 러너가 알린다 -- 기준을 갱신하라는 뜻이다.

실행:  .venv\Scripts\python.exe develop\test_semantic_fixtures.py
"""
import io
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJ, "src"))
if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import axis_classes as ac        # noqa: E402
import semantic_events as SE     # noqa: E402

XFAIL = {3, 4, 5}                   # #5 가 끝나면 비운다


class _Tracer:
    """`SemanticWrappers` 가 읽는 최소 트레이서. 텐서 대신 해시 가능한 표식을 쓴다.

    `logical_source` / `bump_version` 은 #5 가 트레이서에 넣을 것의 **자리**다. 지금은
    항상 ActualPort 를 돌려주므로 case4 가 빨간불이다.
    """

    def __init__(self, last_op_id=100):
        self.rows = [{"op_id": last_op_id}]
        self.tensor_uid = {}
        self.frozen_source_of_pre_row = None

    def uid(self, obj, n):
        self.tensor_uid[obj] = n
        return obj

    def logical_source(self, t):
        src = getattr(self, "_logical", {}).get(self.tensor_uid.get(t))
        out = src or ("op", self.tensor_uid.get(t), 0)
        if self.frozen_source_of_pre_row is None:
            self.frozen_source_of_pre_row = out
        return out


def _fake_transformers(fn):
    """`repeat_kv` 를 가진 가짜 transformers 서브모듈을 sys.modules 에 꽂는다."""
    m = types.ModuleType("transformers.fixture_mod")
    fn.__module__ = "transformers.fixture_mod"
    m.repeat_kv = fn
    sys.modules["transformers.fixture_mod"] = m
    return m


def _run_repeat_kv(fn, x, n_rep, tracer):
    m = _fake_transformers(fn)
    try:
        with SE.SemanticWrappers(tracer):
            out = m.repeat_kv(x, n_rep)
        return out, SE.events()
    finally:
        sys.modules.pop("transformers.fixture_mod", None)


# --- 1. no-op repeat_kv ------------------------------------------------
def case1():
    tr = _Tracer(); x = tr.uid(object(), 7)
    out, ev = _run_repeat_kv(lambda h, n, *a, **k: h, x, 1, tr)
    e = [z for z in ev if z["kind"] == "repeat_kv"]
    assert len(e) == 1, f"이벤트 {len(e)}개"
    assert e[0]["noop"] is True, "n_rep=1 인데 noop 이 아니다"
    assert out is x, "no-op 인데 다른 텐서가 돌아왔다"
    assert e[0]["in_tensor_id"] == e[0]["out_tensor_id"] == 7, e[0]
    assert e[0]["at_op_id"] == 100, "경계 시점을 못 남겼다"


# --- 2. materialized n_rep>1 ------------------------------------------
def case2():
    tr = _Tracer()
    x = tr.uid(object(), 7)
    y = tr.uid(object(), 8)
    out, ev = _run_repeat_kv(lambda h, n, *a, **k: y, x, 4, tr)
    e = [z for z in ev if z["kind"] == "repeat_kv"][0]
    assert e["noop"] is False, "새 텐서인데 noop 이라고 적혔다"
    assert e["n_rep"] == 4 and e["in_tensor_id"] == 7 and e["out_tensor_id"] == 8, e


# --- 3. barrier 사이의 무관한 permute ---------------------------------
def _mamba_transpose_rows():
    """attention 과 무관한 mamba 전치 하나. `scalar_args` 는 **트레이서가 쓰는 형식**이다."""
    rows = [
        {"op_id": 10, "op_type": "linear", "module_path": "m.layers.0.self_attn",
         "input_sources": [None], "input_shape": [["B", "T", "D"]],
         "output_shape": [["B", "T", "D"]]},
        {"op_id": 50, "op_type": "transpose", "module_path": "m.layers.0.mamba",
         "input_sources": [None], "scalar_args": {"pos": [1, 2]},
         "input_shape": [["B", "d_conv", "L"]], "output_shape": [["B", "L", "d_conv"]]},
    ]
    conc = {10: {"input_shape": [[1, 4, 8]], "output_shape": [[1, 4, 8]]},
            50: {"input_shape": [[1, 3, 5]], "output_shape": [[1, 5, 3]]}}
    return rows, conc


def _transpose_edges(bars):
    rows, conc = _mamba_transpose_rows()
    return [e for e in ac.lineage_edges(rows, conc, noop_barriers=bars)
            if e[0][0] == 50 and e[1][0] == 50]


def case3():
    """barrier 는 **그 텐서/축**의 경계다. 시간 구간이 아니다.

    지금 규칙은 `min(bars) <= oid <= max(bars)` 인 전치를 전부 막는다. barrier 가 층마다
    흩어져 있으면 사이의 무관한 op 까지 통째로 막힌다 -- Zamba2 에서 전치 449개 중 355개가
    막히고 그중 330개가 attention 밖(mamba 275)이었다. 함대로는 5,134개 중 1,790개다
    (2026-09-10 실측).

    **대조군을 먼저 통과시킨다.** barrier 없이도 간선이 안 생기면 이 사례는 barrier 결함을
    재현하는 것이 아니라 그냥 규칙이 안 도는 것이다 -- 실제로 처음에 `scalar_args` 를
    `{"dim0":1,"dim1":2}` 로 썼다가 그 함정에 빠졌다(외부 검토 2026-09-10 이 짚었다).
    `_perm_from_args` 는 `args["pos"]` 를 읽는다(`src/axis_classes.py`).
    """
    assert _transpose_edges(()), "대조군 실패: barrier 가 없는데도 전치 간선이 없다"
    assert _transpose_edges((10, 90)), "무관한 mamba 전치가 barrier 때문에 막혔다"


# --- 4. no-op 뒤 별칭 재사용 ------------------------------------------
def case4():
    """no-op 이면 입출력이 **같은 텐서**다. 경계 이전 소비자와 이후 소비자가 갈려야 한다.

    검사하는 것: 이벤트 **전에** 기록된 행의 `input_sources` 는 그대로 남고, 이벤트 **뒤에**
    같은 텐서를 소비하는 행은 SemanticPort 를 가리켜야 한다. 시점만 확인하면 논리 version
    구현이 하나도 없어도 통과한다(외부 검토 2026-09-10).

    `old = x` 를 이벤트 뒤에 다시 쓰는 **진짜 별칭 재사용**도 같이 본다. 같은 파이썬 객체라
    트레이서가 `old` 와 `y` 를 구별할 수 없으므로, 그 패턴은 "이벤트 뒤 소비자는 전부 새
    version" 이라는 **제한을 명시**하거나 unsupported 로 검출해야 한다.
    """
    tr = _Tracer(last_op_id=100)
    x = tr.uid(object(), 7)

    pre = tr.logical_source(x)                 # 이벤트 **전** 소비자가 본 것
    out, ev = _run_repeat_kv(lambda h, n, *a, **k: h, x, 1, tr)
    e = [z for z in ev if z["kind"] == "repeat_kv"][0]
    post = tr.logical_source(out)              # 이벤트 **뒤** 소비자가 볼 것

    assert e["in_tensor_id"] == e["out_tensor_id"], "no-op 인데 텐서 id 가 갈렸다"
    assert pre != post, (
        f"이벤트 전후의 논리 출처가 같다 ({pre!r}) -- 경계가 표현되지 않았다")
    assert post[0] == "sem", f"이벤트 뒤 출처가 SemanticPort 가 아니다: {post!r}"
    assert tr.frozen_source_of_pre_row == pre, "이미 기록된 행의 출처가 바뀌었다"

    old = x                                    # 진짜 별칭 재사용
    assert tr.logical_source(old) == post, (
        "같은 객체인데 별칭이 옛 version 을 가리킨다 -- 그 구별은 불가능하므로 "
        "'이벤트 뒤 소비자는 전부 새 version' 이라는 제한이 명시돼야 한다")


# --- 5. 등록 안 된 구현 -----------------------------------------------
def case5():
    """레이아웃이 `[B, n_kv, T, D]` 가 아닌 구현이면 축 1 이 아니다. 추측하면 안 된다."""
    tr = _Tracer()
    x = tr.uid(object(), 7)

    def odd_layout(h, n, *a, **k):        # [B, T, n_kv, D] 를 쓰는 가상의 구현
        return h
    out, ev = _run_repeat_kv(odd_layout, x, 1, tr)
    e = [z for z in ev if z["kind"] == "repeat_kv"][0]
    assert e.get("resolved") is False, (
        "등록 안 된 구현인데 축을 확정했다고 적혔다 -- 지금은 axis=1 하드코딩이다")


CASES = [case1, case2, case3, case4, case5]


def main():
    red = []
    unexpected_green = []
    for i, fn in enumerate(CASES, 1):
        try:
            fn()
            ok, msg = True, ""
        except AssertionError as ex:
            ok, msg = False, str(ex).splitlines()[0][:110]
        tag = "XFAIL" if i in XFAIL else ""
        if ok and i in XFAIL:
            unexpected_green.append(i)
            print(f"  {i}. {fn.__doc__ or fn.__name__:.60}  초록 (XFAIL 인데 통과)")
        elif ok:
            print(f"  {i}. PASS   {(fn.__doc__ or fn.__name__).splitlines()[0][:64]}")
        elif i in XFAIL:
            print(f"  {i}. {tag}  {(fn.__doc__ or fn.__name__).splitlines()[0][:56]}\n"
                  f"        -> {msg}")
        else:
            red.append(i)
            print(f"  {i}. **FAIL** {(fn.__doc__ or fn.__name__).splitlines()[0][:56]}\n"
                  f"        -> {msg}")
    print()
    if unexpected_green:
        print(f"XFAIL 이 통과했다: {unexpected_green} -- XFAIL 집합을 갱신하라.")
        return 1
    if red:
        print(f"FAIL: {red}")
        return 1
    print(f"5개 중 {5 - len(XFAIL)}개 통과, {sorted(XFAIL)} 는 #5 대기(XFAIL).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
