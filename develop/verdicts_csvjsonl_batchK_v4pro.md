# 배치 K 판정 — DeepSeek-V4-Pro 의 `d_head` / `T/m_csa` 값 충돌

근거 소스: 설치된 transformers 5.14.1 (`modeling_deepseek_v4.py`) + T=1920 재트레이스.
판정일 2026-09-04.

## 주장

Codex: compressor / indexer / scorer 의 **window 축**에 `d_head` 가 붙어 있다. 그 축은
`n_windows` 여야 한다.

## 판정 — **CONFIRMED**

`modeling_deepseek_v4.py:646-657`:

    n_windows = chunk_kv.shape[1] // self.compress_rate
    chunk_kv  = chunk_kv.view(batch, n_windows, ratio, -1)
    new_kv    = chunk_kv.new_zeros((batch, n_windows, 2 * ratio, self.head_dim))

축 1 = `n_windows`, 마지막 축 = `head_dim`. 지적이 맞다.

## 왜 두 라운드 동안 못 고쳤나

`head_dim = 512`, `T = 2048`, `compress_rate = 4` → `n_windows = 512`. **두 축의 정수가 같다.**

저희 파이프라인은 "축 등가류"를 만들어 같은 축인 자리를 묶고 등가류마다 이름을 한 번만
정합니다. 등가류 간선은 **구체 shape 이 일치하는 생산자→소비자** 관계로 놓이므로, 구조가
다른 두 텐서가 같은 shape 으로 보이면 하나로 묶입니다. 그래서 등가류를 따라 고치면 반대쪽
(원래 맞던 `d_head`)이 오염되고, 무시하고 자리별로 고치면 "한 축에 이름이 둘"이라는 게이트
검사가 FAIL 했습니다. 2차·3차 라운드에서 각각 시도했다가 되돌린 것이 이 때문입니다.

## 어떻게 풀었나 — T=1920 probe

Codex 가 제안한 multi-length probe 를 썼습니다. **같은 모델을 다른 `seq_len` 으로 한 번 더
트레이스하면 T 로부터 유도되는 축만 움직여 충돌이 풀립니다.**

`T = 1920` 을 고른 이유: `m_csa = 4` 와 `m_hca = 128` 의 공배수라 나머지 없이 구조가
유지됩니다. 실측 확인 — `d_head = 512`, `T/m_csa = 480`, `T/m_hca = 15`. 세 축이 전부 갈립니다.
두 실행의 op 수는 `(module, op_type)` 별로 완전히 동일했습니다(46,268개).

프로파일 `develop/models/probe-deepseek-v4-pro-T1920.yaml` 은 `--out develop/probe_out` 으로만
돌리고 **promote 하지 않습니다**. 산출물은 T=2048 그대로입니다.

## 옮기면서 네 번 틀렸다 — 전부 도구의 규칙으로 박았다

`develop/probe_transfer.py` 의 docstring 에 있습니다.

| # | 틀린 것 | 그때 난 결과 |
|---|---|---|
| 1 | base 를 **이미 교정이 적용된** `develop/out` 으로 잡음 | 자기 자신과 비교해 386개를 헛짚음 |
| 2 | `module` 정규식을 마지막 마디(`rotary_emb$`)로만 씀 | `compressor` 와 `indexer` 양쪽에 매치 → 미발화 86 |
| 3 | 키에 `layer_type` 을 안 넣음 | hca/csa 교대 61층에서 csa 값이 hca 에 적용 → 등가류 충돌 91 |
| 4 | **자리 단위**로 옮김 | 한 등가류가 두 이름으로 쪼개짐 → 충돌 90 + `reshape_incons` 60 |

최종 형태:

- 안정 키 = `(module_key, layer_type, op_type, module-local nth, field, shape_index, axis)`.
  **shape 으로 대응시키지 않는다** — 두 실행은 shape 이 달라지는 것이 목적이다.
- 옮기는 **단위는 등가류**. 클래스마다 대표 자리 하나에만 적고 `spread: class` 로 나머지를
  끌고 간다. 등가류는 "이 자리들은 같은 축"이라는 주장이고 probe 는 "그 축의 이름"에
  답하므로, 둘을 합치는 올바른 단위가 등가류다.
- **반증 요구**: probe 가 base 이름을 실제로 깨뜨린 자리가 하나는 있어야 채택한다.
  없으면 probe 에서도 두 이름의 값이 같다는 뜻이고, 그때 이름을 바꾸는 건 판별이 아니라
  probe 실행 라벨러의 잡음을 베끼는 것이다. **이 조건이 486 클래스를 걸렀다** —
  `g_o` 와 `T/m_hca+1` 은 T=1920 에서도 둘 다 16, `d_rope` 와 `n_h_I` 는 둘 다 64.

## 결과

클래스 1,770건 교정 → `rules/label_overrides.yaml` 항목 52개(전부 `layer_types` + `spread: class`).

    reshape_incons 0  전치 0  이름생성 0  미발화 0  등가류 0  C-FAIL 0

csa 층 (소스 그대로):

    new_zeros  in [[B, T/m_csa, m_csa, 2*d_head]]  out [[B, T/m_csa, 2*m_csa, d_head]]
    copy_      in [[B, T/m_csa, m_csa, d_head], [B, T/m_csa, m_csa, d_head]]
    concat     in [[B, 1, T/m_csa, d_head-d_rope], [B, 1, T/m_csa, d_rope]] -> [[B, 1, T/m_csa, d_head]]
    topk       in [[B, T, T/m_csa]] -> [[B, T, T/m_csa], [B, T, T/m_csa]]

hca 층은 `T/m_hca` / `d_head` 로 그대로 남았습니다 — 2·3차의 "번짐"이 재발하지 않았습니다.

덤: 방향이 **반대**인 오라벨 3건(`d_head` 여야 할 자리에 `T/m_csa` 가 붙어 있던 것)도 같은
probe 가 잡아냈습니다.

## 부수 효과 — 기존 교정 하나가 죽었다

`indexer$` / `elementwise_add` / `nth=2` 의 `n_h_I -> d_rope` 항목이 발화 0 이 됐습니다.
앵커 `shape: [B, 1, d_head, n_h_I]` 의 **축 2** 가 `T/m_csa` 로 바뀌었기 때문입니다.
`shape` 앵커를 뺐습니다 — `module + op_type + nth + field + shape_index + axis + expect` 만으로
이미 한 자리를 정확히 지목하므로 잉여였습니다.

**교훈: shape 앵커는 다른 교정이 같은 shape 의 다른 축을 건드리면 낡는다.** 위치 선택자로
자리가 유일하게 지목되면 `shape` 을 쓰지 않습니다.

## 같은 방법이 안 통하는 값 충돌 3건

probe 는 **T 로부터 유도되는 축**만 가릅니다. 아래는 두 축이 모두 config 상수라 `seq_len` 을
바꿔도 같이 움직입니다. **원리적 미해결로 문서화합니다.**

| 모델 | 묶이는 두 축 | 값 |
|---|---|---|
| zai-org__GLM-5.2 | 캐시 key ↔ value | 256 |
| Zyphra__Zamba2-1.2B | mamba head ↔ head feature (decode `select` 6행) | 64 |
| MiniMax-M2 / OLMoE / V4-Flash 2종 | experts gate-up 원본 ↔ 전치본 | `d_model = 2*d_moe` |

풀려면 `d_model != 2*d_moe` 인 같은 계열 config 로 probe 해야 하는데, 그건 다른 아키텍처를
트레이스하는 것이라 판정 승격 가드(계열을 넘지 않는다)에 걸립니다.
