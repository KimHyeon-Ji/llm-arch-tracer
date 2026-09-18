r"""`classify_unmatched` 가 **토큰 중복을 분할로 승인하지 않는가**.

외부 검토(Codex 2026-09-18)가 코드로 짚은 반례다. shape 집합만 비교하면 아래 둘을 못
가른다 -- 들어오는 shape 도 나가는 shape 도 같기 때문이다:

    옛: alias(x)                  # [4] -> [4]
    새: a = x[:2]; cat([a, a])    # [4] -> [2] -> [4]
    x = [0,1,2,3] 인데 새 판은 [0,1,0,1] 이다.

검사기가 못 보는 결함은 없는 것과 같으므로, 이 반례를 테스트로 박아 둔다.

실행:  .venv\Scripts\python.exe develop\test_partition_soundness.py
"""
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

from transition_diff import classify_unmatched                 # noqa: E402

FAILS = []


def check(name, got, want):
    ok = got == want
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: {got!r}")
    if not ok:
        FAILS.append(f"{name}: {got!r} != {want!r}")


def main():
    print("분할 건전성")

    # 1) 중복 -- 승인하면 안 된다
    old = [{"op_id": 1, "raw_op": "aten.alias.default",
            "input_shape": [[4]], "output_shape": [[4]]}]
    new = [{"op_id": 10, "raw_op": "aten.slice.Tensor",
            "input_shape": [[4]], "output_shape": [[2]]},
           {"op_id": 11, "raw_op": "aten.cat.default",
            "input_shape": [[2], [2]], "output_shape": [[4]]}]
    ports = {10: {"op_id": 10, "scalar_args": {"pos": [0, 0, 2]}, "input_tensor_ids": [7]}}
    k, _ = classify_unmatched(old, new, {}, {}, ports_new=ports)
    check("cat([a, a]) 는 분할이 아니다", k != "sequence_partition_expected", True)

    # 2) 진짜 분할 -- 승인해야 한다
    new2 = [{"op_id": 20, "raw_op": "aten.slice.Tensor",
             "input_shape": [[4]], "output_shape": [[2]]},
            {"op_id": 21, "raw_op": "aten.slice.Tensor",
             "input_shape": [[4]], "output_shape": [[2]]},
            {"op_id": 22, "raw_op": "aten.cat.default",
             "input_shape": [[2], [2]], "output_shape": [[4]]}]
    ports2 = {20: {"op_id": 20, "scalar_args": {"pos": [0, 0, 2]}, "input_tensor_ids": [7]},
              21: {"op_id": 21, "scalar_args": {"pos": [0, 2, 4]}, "input_tensor_ids": [7]}}
    k2, _ = classify_unmatched(old, new2, {}, {}, ports_new=ports2)
    check("[0,2)+[2,4) 는 분할이다", k2, "sequence_partition_expected")

    # 3) 범위를 모르면 승인하지 않는다
    k3, _ = classify_unmatched(old, new2, {}, {}, ports_new={})
    check("ports 없으면 승인 안 함", k3 != "sequence_partition_expected", True)

    print()
    if FAILS:
        print(f"분할 건전성 FAIL {len(FAILS)}건")
        for f in FAILS:
            print("   ", f)
        return 1
    print("분할 건전성 PASS")
    return 0


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    raise SystemExit(main())
