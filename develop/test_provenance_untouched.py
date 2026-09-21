r"""설명을 만드느라 **발행되는 등급 집계를 건드리지 않는가.**

`label_provenance` 가 읽는 `resolver.stats` 는 호출마다 누적되는 카운터다. 설명 문구를
만들려고 shape 을 한 번 더 풀면 그 축들이 두 번 세어진다 -- 2026-09-21 에 실제로 그랬다:
`_known_limits` / `label_only_symbols` 를 resolver 로 배선했더니 DeepSeek-V4-Pro 의
heuristic 이 **549 -> 915**. 라벨은 하나도 안 바뀌었고 **숫자만 틀렸다.**

발행본의 등급 집계가 그 숫자이므로, 조용히 틀리면 읽는 쪽은 알 수 없다. 게이트의
`heur 퇴행` 검사가 잡아 줬지만, 기준선이 없는 새 모델에서는 안 잡힌다.

실행:
    .venv\Scripts\python.exe develop\test_provenance_untouched.py
"""
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import summarize                                               # noqa: E402


class _FakeResolver:
    """호출될 때마다 stats 를 올리는, 진짜와 같은 성질의 최소 대역."""

    def __init__(self):
        self.stats = {"scoped_symbol": 10, "heur_multiple": 5, "bare": 2}

    def __call__(self, shape, module_path, is_weight=False):
        for _ in shape or ():
            self.stats["heur_multiple"] = self.stats.get("heur_multiple", 0) + 1
        return [str(x) for x in (shape or ())]


ROWS = [{"module_path": "model.layers.0.self_attn",
         "input_shape": [[3, 96, 5, 128]], "output_shape": [[3, 96, 5, 128]],
         "weight_shape": None}] * 4
SYM = {"n_h_kda": 96, "d_head_kda": 128, "d_chunk": 64, "E_shared": 2, "d_moe": 3072}


def run(name, fn):
    r = _FakeResolver()
    before = dict(r.stats)
    fn(r)
    ok = r.stats == before
    print(f"{'OK ' if ok else '**실패**'} {name:34} "
          + ("지표 그대로" if ok else f"{before} -> {r.stats}"))
    return 0 if ok else 1


def main():
    f = 0
    f += run("label_only_symbols",
             lambda r: summarize.label_only_symbols(ROWS, SYM, r, 320))
    f += run("_known_limits",
             lambda r: summarize._known_limits(SYM, ROWS, r))
    # 대역이 진짜로 stats 를 올리는지 -- 시험이 아무것도 안 보는 것을 막는다
    r = _FakeResolver()
    n0 = sum(r.stats.values())
    r([1, 2, 3], "m")
    f += 0 if sum(r.stats.values()) > n0 else 1
    print(f"{'OK ' if sum(r.stats.values()) > n0 else '**실패**'} "
          f"{'대역이 실제로 stats 를 올린다':34} {n0} -> {sum(r.stats.values())}")
    print()
    if f:
        print(f"**{f}개 실패 -- 설명을 만들면서 발행 지표를 바꾼다**")
    else:
        print("3/3 — 설명을 만들어도 등급 집계는 그대로다")
    return 1 if f else 0


if __name__ == "__main__":
    sys.exit(main())
