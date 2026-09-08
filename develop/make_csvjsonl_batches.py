"""Codex CSV/JSONL 검토 요청 배치(A~J)를 다시 조립한다.

왜 필요한가
-----------
2026-09-02 라운드의 배치 파일은 손으로 만들어졌다. 그 뒤 배치 A~J 판정을 반영해
25개 모델의 `<phase>.csv` / `<phase>.jsonl` 이 바뀌었으므로, **재검토 요청은 반드시
바뀐 산출물로 다시 만든 패킷**이어야 한다. 옛 패킷을 다시 보내면 이미 고친 것을
또 지적받는다.

구성(어느 배치에 어느 모델이 들어가는가)은 지난 라운드와 **동일하게 유지**한다.
Codex 답변이 배치 단위로 오고, 판정 파일(develop/verdicts_csvjsonl_batch*.md)도
그 단위로 쓰여 있어 대조가 가능해야 하기 때문이다.

각 배치 파일 = PREAMBLE + 배치 헤더 + 모델별 리뷰 패킷(make_review_packet.build).

실행:
    .venv\\Scripts\\python.exe develop\\make_csvjsonl_batches.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJ, "src"))
sys.path.insert(0, HERE)

import make_review_packet  # noqa: E402

PREAMBLE = os.path.join(HERE, "codex_review_request_csvjsonl_PREAMBLE.md")

# 지난 라운드와 동일한 구성. 바꾸면 판정 파일과의 대조가 깨진다.
BATCHES = {
    "A": ["MiniMaxAI__MiniMax-M2", "NX-AI__xLSTM-7b",
          "Qwen__Qwen3-Next-80B-A3B-Instruct", "allenai__OLMo-2-1124-7B-Instruct"],
    "B": ["Qwen__Qwen3.5-397B-A17B", "Qwen__Qwen3.5-4B", "allenai__OLMoE-1B-7B-0924"],
    "C": ["Qwen__Qwen3.6-27B", "Qwen__Qwen3.6-35B-A3B", "bzantium__tiny-deepseek-v3"],
    "D": ["Zyphra__Zamba2-1.2B", "deepseek-ai__DeepSeek-V2-Lite",
          "openai-community__gpt2-xl", "hf-internal-testing__tiny-random-LlamaForCausalLM"],
    "E": ["deepseek-ai__DeepSeek-V3", "ibm-granite__granite-4.0-h-small",
          "meta-llama__Llama-3.1-405B", "tencent__Hunyuan-A13B-Instruct"],
    "F": ["deepseek-ai__DeepSeek-V4-Flash"],
    "G": ["deepseek-ai__DeepSeek-V4-Flash-0731"],
    "H": ["moonshotai__Kimi-K2-Instruct", "moonshotai__Kimi-K2.6",
          "moonshotai__Kimi-K2.7-Code", "nvidia__NVIDIA-Nemotron-3-Nano-4B-BF16"],
    "I": ["nvidia__NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
          "nvidia__NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16", "tiiuae__Falcon-H1-7B-Instruct"],
    "J": ["zai-org__GLM-5.2"],
    # 배치 K 는 3차 라운드에 신설했다. 4차에서 **둘로 쪼갰다** -- 한 파일이 911,650자로
    # 다른 배치의 3.5배였고, 검토자가 끝까지 못 읽으면 하필 이번에 가장 크게 바꾼 모델이
    # 검토를 못 받는다.
    "K1": ["moonshotai__Kimi-K3"],
    "K2": ["deepseek-ai__DeepSeek-V4-Pro"],
    # 배치 L~P 는 4차에 신설했다. **이 17개 모델은 A~K 어디에도 들어간 적이 없어 CSV/JSONL
    # 외부 검토를 한 번도 받지 못했다.** 1차 패킷을 28개로 손으로 짤 때 빠졌고 그 구성이
    # 라운드마다 그대로 이어졌다. results 브랜치에 올리기 전에 전 모델이 한 번은 검토를
    # 받아야 하므로 여기서 채운다. 묶음은 아키텍처가 비슷한 것끼리 -- 검토자가 같은 계열을
    # 나란히 놓고 대조할 수 있게.
    "L": ["HuggingFaceTB__SmolLM3-3B-Base", "Qwen__Qwen2.5-0.5B",
          "google__gemma-2-2b", "google__gemma-3-270m"],
    "M": ["meta-llama__Llama-3.1-8B", "meta-llama__Llama-3.1-70B",
          "microsoft__Phi-4", "tiiuae__falcon-7b"],
    "N": ["mistralai__Mistral-Small-3.2-24B-Instruct-2506", "Qwen__Qwen3-30B-A3B",
          "baidu__ERNIE-4.5-21B-A3B-PT", "LiquidAI__LFM2-8B-A1B"],
    "O": ["openai__gpt-oss-20b", "openai__gpt-oss-120b",
          "meta-llama__Llama-4-Maverick-17B-128E"],
    "P": ["zai-org__GLM-4.5-Air", "moonshotai__Kimi-Linear-48B-A3B-Instruct"],
}

SEP = "\n\n# " + "=" * 60 + "\n# 모델: {}\n# " + "=" * 60 + "\n\n"


def main():
    preamble = open(PREAMBLE, encoding="utf-8").read()
    errors = []
    for letter, models in BATCHES.items():
        parts = [preamble, f"\n\n# 배치 {letter} — 대상 모델: " + ", ".join(models) + "\n"]
        for m in models:
            d = os.path.join(PROJ, "models", m)
            if not os.path.isdir(d):
                errors.append(f"{letter}:{m} (모델 폴더 없음)")
                continue
            try:
                parts.append(SEP.format(m))
                parts.append(make_review_packet.build(m))
            except Exception as e:
                errors.append(f"{letter}:{m} ({type(e).__name__}: {str(e)[:80]})")
        out = os.path.join(HERE, f"codex_review_request_csvjsonl_batch{letter}.md")
        text = "".join(parts)
        with open(out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"wrote {os.path.relpath(out, PROJ)}  ({len(text):,} chars, {len(models)} models)")

    # 실패한 모델이 있으면 그 배치는 불완전하다. 조용히 넘어가면 검토자가 빠진 모델을
    # 모른 채 "이상 없음"을 돌려준다.
    if errors:
        print("\n실패 %d건 — 해당 배치는 불완전하다: %s" % (len(errors), ", ".join(errors)))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
