r"""구간 동치 대조가 **말할 자격이 있는 것만 말하는가.**

내 거짓 통과 셋은 모두 같은 형태였다 -- 검사가 **변화시키지 않은 차원**에 구멍이 있었다.

    1. 배치 조각을 b=0 만 비교했다            (외부 검토 2026-09-19)
    2. 짝을 못 지은 레코드를 미증명으로 안 셌다  (외부 검토 2026-09-19)
    3. 대조할 것이 0 건일 때 `0 == 0` 으로 통과했다 (2026-09-20, 라벨만 바뀐 전환)

3번이 이 파일의 대상이다. 라벨만 바뀐 전환은 재실행할 구간이 없으므로 구간 동치를
**주장해서는 안 된다.**

실행:
    .venv\Scripts\python.exe develop\test_segment_verdict.py
"""
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import transition_release as TR                                # noqa: E402

SEG = "layout_lowering_verified"          # SEGMENT_CATS 의 한 항목
NON = "fabrication_withdrawn"             # 구간 범주가 아닌 라벨 전환


def run(name, cats, records, want):
    seg, got = TR._segment_verdict(cats, records)
    ok = got == want
    print(f"{'OK ' if ok else '**실패**'} {name:44} {got:18} (기대 {want})")
    return 0 if ok else 1


def main():
    f = 0
    # 3번 반례: 라벨만 바뀐 전환 -- 아무것도 주장하면 안 된다
    f += run("라벨만 바뀜 (같음 + 지어낸 이름 거둠)",
             {"같음": 5_600_531, NON: 552}, 0, "nothing_to_prove")
    f += run("전환 자체가 없음", {}, 0, "nothing_to_prove")
    # 진짜 구간 전환이 있고 전부 대조됐다
    f += run("구간 552 건 전부 대조", {"같음": 10, SEG: 552}, 552, "covered")
    # 구간은 있는데 대조가 0 건 -- 절대 통과하면 안 된다
    f += run("구간 552 건인데 대조 0 건", {SEG: 552}, 0, "mismatch")
    # 대조는 했는데 수가 모자라다
    f += run("구간 552 건에 대조 300 건", {SEG: 552}, 300, "mismatch")
    # 구간이 없는데 대조 레코드만 있다 -- 짝이 안 맞는다
    f += run("구간 0 건인데 대조 300 건", {"같음": 9, NON: 1}, 300, "mismatch")
    print()
    if f:
        print(f"**{f}개 실패 -- 증명이 말할 자격 없는 것을 말한다**")
    else:
        print("6/6 — 대조할 것이 없으면 아무것도 주장하지 않는다")
    return 1 if f else 0


if __name__ == "__main__":
    sys.exit(main())
