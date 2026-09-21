검토 대상은 **results `9f8fe9e13a0300f3f8804c1f0958b264379d2b2c`**, 생성 소스는 `db6d2be6`이다. 2026-09-21 검토.

**층 접기와 주요 projection/attention/MoE 라벨은 아래에서 소스와 대조한 자리에서 맞는다. Q0의 현재 라벨을 되돌릴 이유도 발견하지 못했다. 다만 “낡은 앵커는 대부분 B가 붙어서 그렇다”는 설명은 V4-Pro에 맞지 않는다.** K3에는 발행 심볼 정의 누락이 있고, 일부 공개 설명은 현재 표와 모순된다. 따라서 “전부 문제없음” 판정은 아니다.

이미 공개한 `scope_inferred`/`heuristic`/`open_tie`/`unresolved`, K3의 배치 전환 증거 부재·shim 토큰 수·인용 품질은 새 결함으로 세지 않았다. 여기서 “맞다”는 아래 특정 구조와 자리를 확인했다는 뜻이며, 수백만 축 전부를 사람이 재확정했다는 뜻은 아니다.

산출물·소스·규칙은 수정하지 않았다. 검토 기록과 읽기 전용 수집 스크립트만 `develop/`에 추가했다.

**1. 먼저 고칠 발행 내용**

| 구분 | 모델 / 위치 | 현재 → 필요한 표현 | 근거와 영향 |
|---|---|---|---|
| 정의 누락 | K3 `prefill.csv`와 `prefill.jsonl`, 대표 `op_id=19` | `B*n_h_kda*n_chunk`를 쓰면서 `structure.yaml`에는 `n_chunk` 정의가 없음 → 이 실행의 `n_chunk=T/d_chunk=320/64=5`를 명시하거나 식을 직접 풀어 쓸 것 | prefill **1,056행, 3,168축 자리**에서 사용한다. 식의 의미는 맞지만 발행본의 심볼표만으로는 크기를 계산할 수 없다. FLA `naive.py:106–118,130`의 `NT=T//BT`와 청크별 수축이 근거. 여기서는 T가 64의 배수다. |
| 공개 설명 오류 | V4-Pro `UNKNOWNS.md` §5 | 27건이 “대부분 `n_h→B*n_h`” → **6건**이 그 유형이고 나머지는 별도 분류 | 나머지에는 압축 토큰 축의 옛 이름, singleton, slice/concat 포트와 서수 변경이 있다. 현재 라벨은 아래 Q0 표처럼 대조해야 한다. 일괄 문자열 치환은 안 된다. |
| 공개 설명 오류 | K3 `structure.yaml:12` | “`E_shared=2`는 축 이름으로 쓰이지 않는다, shared expert 모듈 수” → **하나의 shared MLP 폭을 `E_shared*d_moe`로 넓힌다** | 실제 표의 prefill `1740`, decode `132`부터 그 식이 나온다. 양 phase 각각 253행에서 `E_shared`가 쓰인다. K3 소스 `modeling_kimi_linear.py:797–801`은 MLP 하나를 만든다. 같은 YAML의 6144 범례는 이미 올바르게 설명하므로 파일 내부도 모순이다. |
| 공개 설명 오류 | Llama-4 `UNKNOWNS.md` §3의 “MoE 결합”, `structure.yaml`의 첫 known_limit | shared+routed와 residual add가 한 행으로 합쳐짐 → **현재는 두 행이다** | 양 phase `40(add_)→41(elementwise_add)`, `65→66`. `40`의 의존성은 `[38,39]`, `41`은 `[25,40]`이다. 소스 `modeling_llama4.py:173,458`과 맞는다. 표를 고칠 것이 아니라 설명을 갱신할 자리다. |

첫 항목은 “청크 이름을 잘못 골랐다”는 지적이 아니라 **발행 심볼표의 정의 누락**이다. 그 외 네 모델의 표에서는 `B`, `T`, `structure.yaml.symbols` 밖의 미정의 식별자를 찾지 못했다. K3 decode에는 `n_chunk`가 없다.

추가로 V4-Pro에서는 routed와 shared FFN의 **clamp가 둘 다 표에 없다**. 이는 gpt-oss에서 옮겨 추정한 지적이 아니다. V4 자체의 원시 trace에서 양 phase 각각 244개를 확인했다. 대표 prefill raw `739/740`은 routed, `759/764`는 shared이고, decode raw는 `602/603`, `622/627`이다. 현재 V4 config의 limit는 **10**이다. 소스 `modeling_deepseek_v4.py:984–989,1023–1030`은 gate 상한 10, up 구간 [-10,10]을 적용한다. 요약에서는 prefill `36→37/38`, `42/43→44/45` 사이에 보이지 않는다. 이미 공개한 major-op 선별 정책에 따른 생략으로 이해할 수 있지만, V4의 개별 공개 생략 목록에는 없는 구조 정보다. 새로운 축 오라벨로 세지 않는다.

**2. Q0 — 낡은 확인/교정의 현재 자리를 직접 추적한 결과**

이 절의 **raw 번호**는 `models/<모델>/full/<phase>.trace.raw.jsonl`의 `op_id`다. 발행 csv/jsonl의 번호와 다르다. 확인 번호 #1…은 해당 모델 `full/label_confirmed.json`에서 `matched=0`인 항목의 순서다. `fleet5_artifacts_evidence.json`에 원래 selector와 현행 구조적 후보를 보존했다. 후보를 찾는 것과 의미를 확정하는 것은 구분하고, 아래 판정에는 모델 소스를 추가 대조했다.

**DeepSeek-V4-Pro: 27건을 모두 같은 원인으로 처리하면 안 된다. 현재 후속 자리에서 새 오라벨은 확인되지 않았다.**

| 확인 번호 | 과거 selector의 핵심 | 현재 자리와 판정 |
|---|---|---|
| #1, #2 | decode QK 선행 `n_h`, query `B` | raw decode `342` HCA / `1606` CSA: `[B*n_h,1,w_local+T/m_*]`. 선행축은 B와 head의 곱, query는 1. 현재가 맞다. |
| #3 | prefill PV 선행 `n_h` | raw prefill `493`, `2014`: `[B*n_h,T,d_head]`. 현재가 맞다. |
| #7, #8 | prefill QK 선행 `n_h` | raw prefill `477`, `1998`: `[B*n_h,T,T+T/m_*]`. 현재가 맞다. |
| #26 | decode PV 선행 `n_h`, query `B` | raw decode `358`, `1622`: `[B*n_h,1,d_head]`. 현재가 맞다. |
| #4, #19 | indexer prefill `slice nth=20`, `[B,n_h_I,T,d_rope]` | raw `1927`, nth 20은 **비회전 부분** `[B,n_h_I,T,c_I-d_rope]`. 회전 부분은 raw `1928`, **nth 21**, 마지막 축 `d_rope`. head 축은 양쪽 모두 `n_h_I`. 값이 둘 다 64라 서수만 무시해서는 안 된다. |
| #6, #23 | indexer decode `slice nth=2`, 마지막 `d_rope` | raw `1537`, nth 2는 `c_I-d_rope`; 회전 부분은 raw `1538`, **nth 3**, `d_rope`. 현재 두 이름이 맞다. |
| #5 | self_attn concat 2의 `[B,1,n_h,d_head]` | raw decode `312`는 과거 KV와 새 KV를 연결한 `[B,1,w_local,d_head]`. **축 2는 head 수가 아니라 캐시 길이**다. prefill의 같은 concat raw `374`는 `[B,1,T,d_head]`. 옛 `n_h` 판정을 승계하면 안 된다. |
| #9 | CSA new_zeros `[B,d_head,2*m_csa,T/m_csa]` | raw prefill `1705`: **`[B,T/m_csa,2*m_csa,d_head]`**. 8짜리 겹침 축은 여전히 `2*m_csa`가 맞지만 두 주변 축의 옛 이름은 현재 배치가 아니라 시퀀스/폭 구별 문제다. |
| #10 | indexer new_zeros `[B,d_head,2*m_csa,c_I]` | raw prefill `1807`: `[B,T/m_csa,2*m_csa,c_I]`. 마지막 `c_I`가 맞고, 두 번째는 압축 토큰 수다. |
| #11 | indexer sum `[B,d_head,c_I]` | raw prefill `1835`: `[B,T/m_csa,c_I]`. `c_I`가 맞다. |
| #12, #27 | scorer transpose의 `[B,d_head,c_I]→[B,c_I,d_head]` | raw prefill `1943` / decode `1553`: **`[B,T/m_csa,c_I]→[B,c_I,T/m_csa]`**. 압축 토큰 수는 head 폭이 아니다. |
| #13, #14 | indexer slice `[B,d_head,m_csa,c_I]` | raw prefill `1809/1812`: `[B,T/m_csa,m_csa,c_I]`. 마지막 축 `c_I`가 맞다. |
| #16 | indexer concat `[B,1,d_head,c_I]` | raw prefill `1891`: `[B,1,T/m_csa,c_I]`. 마지막 `c_I`가 맞다. |
| #15, #18, #22 | sink의 view nth 12, `[n_h]→[B,n_h,1,1]` | 실제 sink는 **nth 11**, prefill raw `481`, decode raw `346`: `[n_h]→[1,n_h,1,1]`. 다음 expand에서 B와 query 길이를 늘린다. nth 12는 현재 PV용 view여서 다른 계산이다. |
| #17, #20 | prefill concat 1 입력 0의 `[B,n_h_I,T,d_rope]` | raw `1940` 입력 0은 `[B,n_h_I,T,c_I-d_rope]`, 입력 **1**이 `[B,n_h_I,T,d_rope]`. head는 양쪽 `n_h_I`. `d_rope` 확인을 입력 0에 되살리면 잘못된다. |
| #21, #24 | decode concat 0 입력 0의 마지막 `d_rope` | raw `1550`도 입력 0은 `c_I-d_rope`, 입력 **1**은 `d_rope`. 현재가 맞다. |
| #25 | mask `[B,1,1,w_local]` | raw decode `24`: `[1,1,1,w_local]`. 배치에 broadcast하는 mask이고 마지막 축 `w_local`이 맞다. |

QK/PV 근거는 V4 소스 `717–744,810–853`; sink는 `733`; KV concat은 `821–825`와 Transformers `cache_utils.py:228–235`의 sliding cache update; partial RoPE는 `342–359`; indexer의 chunk/overlap/합산/회전은 `529–565`; scorer 전치는 `455–456`이다. 특히 `c_I-d_rope=64=d_rope`는 요청서의 단순 config 값 충돌 목록에 없는 **유도식 충돌**이다.

미발화 override 3건도 따로 확인했다.

| 교정 | 현재 상태 | 판정 |
|---|---|---|
| `o_a_proj: g_o→g_o` | prefill raw `526–532`, decode `389–395`에서 이미 `g_o` | 이름을 바꾸지 않는 교정이다. 현재 그룹 축 16은 맞다. |
| `o_a_proj transpose input axis 1: T/m_hca→g_o` | prefill raw `529`의 입력은 `[B*T,g_o,n_h*d_head/g_o]`, decode `392`는 `[B,g_o,n_h*d_head/g_o]` | 이미 `g_o`. `T/m_hca`를 다시 도입할 이유가 없다. |
| `attn_hc: T/m_hca→n_hc*n_hc` | prefill raw `140/141`의 comb split 폭은 `n_hc*n_hc`; `151/153`에서 `[n_hc,n_hc]`로 view. decode는 `79/80/90/92` | 현재가 맞다. 소스 `934–943`의 hc×hc mixer다. |

grouped projection의 근거는 V4 `322–332,792–795,870–872`. 발행본 대표는 prefill `22/80/130/188`, decode `20/74/122/176`이다.

**Llama-4-Maverick: 4건 모두 현재 표를 `B→1`로 고칠 근거가 아니다. 마지막 규칙은 특히 주의해야 한다.**

| 미발화 교정 | 현재 자리 | 판정 |
|---|---|---|
| self_attn bmm | decode raw `130/139`, 발행 `5/7`: `[B*n_h,1,…]` | query 축 1이 맞다. B는 head와 접힌 선행축에 있다. |
| feed_forward `_unsafe_view nth=0`, output axis 1 | raw prefill `311`: `[B*E*T,1]`; decode `269`: `[B*E,1]` | router score를 채널 방향으로 broadcast하는 literal 1. 현재가 맞다. |
| feed_forward `elementwise_mul nth=0`, input 1 axis 1 | raw prefill `312`, decode `270`; 발행 양 phase `29` | 두 번째 입력 `[B*E*T,1]` 또는 `[B*E,1]`이 맞다. |
| feed_forward `view nth=1`, output axis 1 | raw prefill `328`: **`[E,B*T,d_model]`**; decode `286`: **`[E,B,d_model]`** | 이 자리는 shared 합산 전 routed expert 결과를 되돌리는 reshape다. **decode axis 1의 B는 정답**이다. 과거 인용한 `reshape(-1,1)`과 지금 selector가 가리키는 연산이 다르다. `expect=1`을 없애서 규칙을 살리면 배치 축을 망가뜨린다. |

소스 `modeling_llama4.py:169–173,361–391`. 발행 `39`의 합산 입력도 `[E,B*T,d_model]` / `[E,B,d_model]`이므로 마지막 B의 의미를 확인할 수 있다. “4건 전부 이제 1로 잘 바뀌어 있다”는 해석에는 동의하지 않는다. **세 종류의 singleton은 1, 마지막은 B/B*T가 맞다.**

**gpt-oss-120b: 6건 중 5건은 배치 접힘, 1건은 sink singleton이다.**

prefill raw `148/163`은 QK/PV, decode raw `129/144`는 sliding, `264/279`는 full QK/PV다. 발행 양 phase `5/7`, `29/31`에서 선행축 `B*n_h`가 맞다. decode query는 1, key 길이는 sliding에서 `w_local`, full에서 `T+1`이다.

나머지 sink 확인은 prefill raw `152`, decode raw `133`, view nth 5다. `[n_h]→[1,n_h,1,1]`이므로 옛 `[B,n_h,1,1]`와 맞지 않는 것이 정상이다. head 축은 여전히 `n_h`다. **현재 라벨 수정 불필요.**

**gpt-oss-20b: 이 모델 파일에서도 별도로 같은 의미를 확인했다.**

prefill raw QK/PV는 **`142/157`**, sink는 **`146`**이다. 120b의 prefill raw 번호를 그대로 적용하면 안 된다. decode raw `129/144`, `264/279`, sink `133`은 확인 결과 동일하다. 발행 `5/7`, `29/31`의 `B*n_h`, query 1, `w_local`/`T+1`, sink의 고정 singleton이 각각 맞다. **현재 라벨 수정 불필요.**

두 모델 각각 config와 결과를 대조했다. source는 `modeling_gpt_oss.py:261–278,296–330`: `repeat_kv` 후 Q/K/V의 head는 n_h, 마지막 채널은 d_head, sink parameter는 n_h개다. 발행본 두 phase의 input/output 축에서 `B*n_h`는 모델당 24자리, 맨 `n_h` 선행축은 0자리였다. V4는 각각 48/0자리로 요청서의 계수도 맞다. 이 계수만으로 나머지 selector의 의미까지 증명되는 것은 아니다.

**K3:** 이번 Q0 대상의 dead confirmation/override는 각각 0건이다. 기존 736건의 인용 품질과 bare 증가는 요청대로 새 오라벨 근거로 사용하지 않았다.

**3. Q1 — 모델별 접기 판정**

| 모델 | 실제로 대조한 구성 | 판정 |
|---|---|---|
| V4-Pro | `0–1=HCA+hash`, `2=CSA+hash`, `3,5,…59=HCA+동적 MoE`, `4,6,…60=CSA+동적 MoE` | 61층 중복/누락 없음. 첫 세 층의 hash routing 때문에 0–1과 2를 뒤의 같은 attention 종류에 합치지 않은 것이 맞다. |
| Llama-4 | 짝수 24층 dense+chunked+RoPE; `1 mod 4` 12층 MoE+chunked+RoPE; `3 mod 4` 12층 MoE+full+NoPE | 48층 중복/누락 없음. `no_rope_layers`의 **1은 실제 use_rope=True**다. |
| K3 | 69 KDA, 24 MLA; 0만 dense, 1–92 MoE; 12층마다 AttnRes 경계 | 93층 중복/누락 없음. MLA 끝 `87,91,92`와 마지막 인접 두 MLA가 맞다. 경계 KDA 12,24,…84를 따로 접은 것도 맞다. |
| gpt-oss-120b | even 18 sliding, odd 18 full | 36층 중복/누락 없음. |
| gpt-oss-20b | even 12 sliding, odd 12 full | 24층 중복/누락 없음. |

V4 pinned `config.json:66`의 `compress_ratios`는 62개지만 본체 `num_hidden_layers=61`까지가 표 대상이다. 첫 61개가 위 스케줄이며, 마지막 추가 항목을 본체 60번 층에 잘못 대입하면 안 된다. 정규화는 `configuration_deepseek_v4.py:266–282`, hash 선택은 modeling `1088–1102`다.

Llama-4는 pinned config의 `interleave_moe_layer_step=2`, `no_rope_layer_interval=4`를 `configuration_llama4.py:179–199`가 전개하고, modeling `337,418–422`가 사용한다. K3는 pinned `linear_attn_config.full_attn_layers`와 `kda_layers`를 대조했으며 `configuration_kimi_k3.py:152–155`가 `(layer_idx+1)`을 검사한다. modeling `883–900,995–997`이 attention/FFN/AttnRes 분기를 결정한다. gpt-oss는 **각 pinned config의 layer_types 전체**를 각각 대조했고 modeling `288,308`이 그 스케줄을 사용한다.

층 번호가 보존되므로 접기 자체는 맞다. 다만 `block_type`을 정확한 attention 종류로 읽으면 오해한다. V4의 `MLA`는 특히 **V2/V3식 latent-KV의 축소·재확장 경로를 뜻하지 않는다**. 실제 V4는 single-head shared K=V와 토큰 압축기, low-rank Q/grouped output이다. K3의 `attn`은 KDA라는 구체 종류를 숨긴다. Llama-4와 gpt-oss의 같은 `attn+MoE`도 mask 종류를 숨긴다. 기존 요약 범위의 한계이지만, 소비자 편의를 위해 `attention_type`/`ffn_type`를 따로 제공하는 편이 낫다. 스택 재구성에는 `repeat`만 곱하지 말고 `layers`를 풀어 번호순으로 배치해야 한다.

**4. Q2 — 충돌 값의 자리별 판정**

| 모델 | 확인한 발행 자리 | 판정과 소스 |
|---|---|---|
| V4-Pro | prefill `17`: `[B,T/m_hca,m_hca,d_head]` | 128은 압축 창 길이 `m_hca`. V4 `386–416`. |
| V4-Pro | prefill `19/21`, decode `17/19` 및 CSA 대응 행 | `B*n_h`는 head batch. decode key 길이의 `w_local`은 local cache 128. 둘을 바꾼 흔적 없음. V4 `717–744,821–832`. |
| V4-Pro | prefill `69–76`, decode `65–70`의 indexer | 투영/수축 채널 `c_I`, head `n_h_I`가 맞다. `c_I=128`을 n_h나 m_hca로 부르면 안 된다. V4 `493–503,529–565,455–459`. |
| V4-Pro | prefill `65/66` | 1024는 **`2*d_head`**. `d_g`나 `k_I`가 아니다. 현재 맞다. V4 `589–600` 및 CSA projection 선언. |
| V4-Pro | prefill `22/80/130/188` | `g_o=16`, 그룹당 출력 `d_g=1024`가 맞다. indexer 선택 개수 `k_I`가 아니다. V4 `322–332,792–795`. |
| V4-Pro | prefill `36–46` 및 대응 그룹 | routed/shared FFN 폭 `d_moe=3072`, residual `d_model=7168`이 맞다. `d_ff`를 별개 dense 단계로 더 세면 안 된다. V4 `974–1003,1085–1102`. |
| Llama-4 | 양 phase attention `5/7,21/23,46/48`; experts `30/33,55/58` | attention의 128은 마지막 `d_head`; expert bmm의 선행 128은 `E`. 현재 맞다. Llama `58–82,338–348,361–366`. |
| Llama-4 | dense `11–15`, routed `30–33`, shared `34–38` | dense 16384=`d_ff`, expert/shared 8192=`d_moe`. 8192짜리 FFN 채널을 `chunk_size`로 쓰지 않았다. Llama `59–63,89–103,164,422`. |
| gpt-oss-120b | 양 phase `13/20,37/44`, `15–19,39–43`, norm/residual | gate_up 저장 `[E,d_model,2*d_moe]`, down `[E,d_moe,d_model]`, GLU 중간 d_moe, down 뒤 d_model이 맞다. 둘 다 2880이어도 역할이 유지된다. GPT `77–92,113–116`. |
| gpt-oss-120b | 양 phase `5/7,29/31`, softmax `6/30` | head 위치 `n_h`, 채널 위치 `d_head`가 맞다. decode sliding 128은 `w_local`, weight 전문가 축 128은 `E`. GPT `261–278,296–309`. |
| gpt-oss-20b | 이 모델의 양 phase `13/20,37/44` 및 attention `5/7,29/31` | d_model/d_moe와 n_h/d_head가 각각 맞다. E는 **32**이며 120b의 E=128 충돌을 이 모델에 옮길 수 없다. 동일 소스의 parameter 정의를 이 모델 config로 따로 대조했다. |
| K3 | 양 phase `12/13/14`, conv `6/8/10` | forget projection bottleneck은 d_head_kda, beta projection은 n_h_kda; conv의 kernel 축 d_conv=4가 맞다. K3 `504–529`. |
| K3 | prefill `1764–1766`, decode `156–158` | KV latent `c_kv`, payload `c_kv+d_rope`, 확장 `n_h*(d_nope+d_v)`가 맞다. K3 `378–394,426–440`. |
| K3 | prefill `1784–1835`, decode `176–227` 중 latent/shared projections | routed 폭 d_moe_lat=3584와 expert 내부 d_moe=3072, shared E_shared*d_moe=6144가 구분되어 있다. K3 `776–812,818–837`. |

K3에서 이미 지적한 MLA PV의 d_nope/d_v open tie는 새 발견으로 반복하지 않았다. 요청서에 적힌 q_pass/value_states/g_proj/KDA head의 이전 판정도 이번 새 결함 수에 넣지 않았다.

K3의 `d_head=74`도 심볼표에 기록된 값 자체는 요청서와 같다. 다만 이것은 공통 config가 `7168//96`으로 채운 기본값(`configuration_kimi_k3.py:63–65`)이지 실제 K3 attention의 head 폭이 아니다. 현재 실행 폭은 MLA Q/K 192, V 128, KDA 128이다. 기존 검토 내용의 해석을 유지해야 하며, 74로 attention FLOPs를 계산하면 안 된다.

**“단일 config 값이 겹치는 곳만 위험하다”는 전제는 좁다.** V4 `c_I-d_rope=d_rope=64`, 과거 `T/m_csa=d_head`, `T/m_hca=g_o=n_hc*n_hc`처럼 실행 길이·유도식·broadcast singleton도 충돌한다. 이번 발행본의 해당 후속 이름은 위와 같이 올바르지만, 단순 값 충돌표만으로 검토를 끝낼 수는 없다.

**5. Q3 — 세 MoE 표현은 현재 실행 경로와 맞는다**

| 모델 | 판정 |
|---|---|
| V4-Pro | `S=B*T*k`개의 token-expert pair에 grouped GEMM을 적용하는 것이 맞다. 세 번째 `[E]` 입력은 **전문가별 누적 offset 벡터**다. `[E,…]` weight 전체가 전달되어도 계산량에 E를 한 번 더 곱하지 않는다. 저장 gate_up `[E,2*d_moe,d_model]`과 operand `[E,d_model,2*d_moe]`의 전치는 정상이다. |
| Llama-4 | 이 `Llama4TextMoe`는 실제로 `hidden_states.repeat(E,1)` 후 routing score를 곱하고, `[E,B*T,d_model]`로 bmm한다. 선택 안 된 전문가에는 0이 들어가지만 dense bmm가 생략되지는 않는다. **현재 CPU/reference 구현에 충실한 표**다. 최적화된 sparse serving의 top-k FLOPs로 읽으면 E/k배 과대평가할 수 있다. |
| K3 | 전문가마다 w1/w3/w2를 기록한 것은 소스 모듈 구성과 맞다. 다만 이미 공개한 even-split 4-expert 대체 경로의 shape다. prefill 3840=`3*320*16/4`, decode 12=`3*16/4`가 맞고, 전체 896개를 실제 순회한 결과는 아니다. shared와 latent down/up은 별도 모듈로 보인다. |
| gpt-oss-120b | E=128, k=4의 grouped 경로가 맞다. 발행 `13/20,37/44`를 이 모델에서 확인했다. |
| gpt-oss-20b | E=32, k=4의 grouped 경로가 맞다. 이 모델의 같은 행들을 별도로 확인했다. |

grouped 경로의 근거는 Transformers `integrations/moe.py:383–409,429–482`. GPT class decorator `modeling_gpt_oss.py:73`, V4 decorator `modeling_deepseek_v4.py:992`가 이 경로를 지원한다. Llama-4는 자기 소스 `169–173,79–83`의 경로이므로 gpt-oss의 grouped 수식을 적용하면 안 된다.

`2*d_moe`는 V4/Llama-4/gpt-oss 모두 gate+up 폭으로 맞다. **layout은 같지 않다**: V4와 Llama는 `.chunk(2,-1)`로 두 반을 나누고, gpt-oss는 `[...,::2]`와 `[...,1::2]`로 **교차 배치**를 나눈다(GPT `88`, V4 `1027`, Llama `81`). K3는 w1/w3 별도 matmul 뒤 concat하므로 “단일 fused GEMM”으로 바꾸어 읽으면 안 된다.

**6. Q4 — 보이는 구조, 생략된 구조, decode 해석**

V4-Pro는 HCA compressor prefill `15–18`, CSA `65–68`, indexer `69–76`, mHC `1–7/27–33`, grouped output `22/23`이 있어 요청한 주요 구성은 보인다. K=V는 `kv_proj→kv_norm` 하나에서 오며, **V4에 K3식 kv_a/kv_b latent 확장이 빠졌다고 지적할 수 없다**. 해당 모듈이 이 소스에 없다. indexer는 544개(`2176/4`) 압축 토큰을 다루며 현재 표본에서는 `min(k_I,544)=544`; 1024개의 선택 결과가 반드시 보여야 하는 것이 아니다. 소스 `568–584`가 min과 causal mask를 적용한다. 이미 공개한 Sinkhorn, compressor/indexer reduction, hash/top-k 구분, inverse RoPE 생략은 새 지적에서 제외한다. 위에서 추가 확인한 clamp도 major-op 범위의 생략이다.

Llama-4의 chunk 경계는 마스크에 있고 표에 없다. 현재 T=17과 decode 길이 18은 chunk_size=8192보다 작아 full과 같은 attention shape가 맞다. 이 checkpoint는 **use_qk_norm=false**여서 QK-norm 부재는 누락이 아니다. NoPE 층의 temperature 연산은 소스 `378–385`에 있고, floor_scale=8192인 현재 짧은 위치에서는 배율 1이다. 미표시는 이미 공개한 major-op 생략이다.

gpt-oss-120b의 sink는 prefill softmax `6/30`의 `T+1`, decode `6`의 `w_local+n_sink`, `30`의 `T+1+n_sink`에 드러난다. PV `7/31`에서는 sink를 뺀 길이로 돌아간다. 여기서 **prefill의 +1은 새 cache token이 아니라 sink**다. sliding은 decode `5/7`에서 128, full은 `29/31`에서 265로 구별된다. router `12/36`의 입력 `[B*T,k]` / `[B,k]`는 **top-k 후 softmax**이므로 맞다(GPT `131–135`).

gpt-oss-20b에서도 해당 softmax/PV/cache/router 행들을 별도로 대조했고 같은 의미가 맞았다. 이 모델의 전문가 수 32를 120b의 128로 해석하면 안 된다. 두 모델의 **MXFP4 unpack/scale/dequant 연산은 표에 없다**. config에서 weight 없는 모델을 만드는 추적(`src/loader.py:55–74`)이며 양자화 checkpoint 로딩 경로를 실행한 것이 아니다. GEMM의 논리 폭을 설명하는 표로는 사용할 수 있지만 MXFP4 저장 바이트·실제 kernel 트래픽을 이 표로 직접 산출할 수 없다.

K3는 conv `6/8/10`, forget 관련 `12–17`, chunk 계산 prefill `18–823`, recurrent decode `18/19`, output `824–826` / `20–22`로 경로가 구별된다. 표는 세부 elementwise 전부를 보여주는 것은 아니지만 conv/forget/chunk 계산이 통째로 없지는 않다. 라우터와 per-expert, latent, shared 모듈도 구별된다. AttnRes 마지막 집계 prefill `15189–15195`, decode `2325–2331`의 비층 7행도 존재한다. KDA는 고정 크기 recurrent state를 사용하는 경로라 **그 모든 축을 T+1 KV cache로 설명하면 틀린다**. MLA 부분만 prefill T, decode key T+1을 따른다. 또한 pinned 설정 `mla_use_nope=true`이고 소스 `396–403,418–440`에는 이 경로의 실제 RoPE 회전이 없다. d_rope라는 split 이름이 남았다는 사실과 회전 실행은 다르다.

| 모델 | decode key/state 해석 |
|---|---|
| V4-Pro | local cache `w_local` + 기존 압축 `T/m_hca` 또는 `T/m_csa`. 이번 한 토큰에서는 새 압축 창이 닫히지 않아 compressor softmax/norm 행 수가 줄어든다. 일반 T+1 치환이 아니다. |
| Llama-4 | query 1, KV 길이 T+1=18. 현재 길이에서는 chunk 경계를 넘지 않는다. |
| K3 | MLA는 T+1=321; KDA는 recurrent state와 conv cache다. |
| gpt-oss-120b | sliding 128 / full T+1=265; softmax에만 sink 1 추가. |
| gpt-oss-20b | 이 모델에서도 sliding 128 / full 265; softmax에만 sink 1 추가. |

**7. Q5 — weight_pos만으로 바이트를 분류하면 틀릴 수 있다**

저장형 weight와 전치 operand를 함께 적는 표기는 타당하다. `weight_shape`를 별도 연산 입력으로 한 번 더 더하면 중복 집계다. 다만 다음은 문서에서 더 명확해야 한다.

* **`weight_pos=-1`은 “weight operand가 없다”와 동의어가 아니다.** V4 prefill `22/80/130/188`, decode `20/74/122/176`은 norm이 아닌 grouped output bmm인데 -1이다. 실제 input 1은 `[g_o,n_h*d_head/g_o,d_g]`의 weight view, 저장형은 `[g_o*d_g,n_h*d_head/g_o]`다. V4 `326–332`가 `weight.view(...).transpose(...)`를 넘긴다. 저장/operand 원소 수는 같다. `01-main.md:141`은 이 예외를 이미 말하지만, **171의 “weight_pos>=0 나머지는 activation”, “-1은 weight_shape가 유일한 출처”를 그대로 적용하면 이 경우 weight를 activation으로 잘못 세거나 두 번 센다.**
* gpt-oss의 linear `2/3/4/8/11`은 입력 0이 **bias**, 입력 1이 activation, 입력 2가 weight다. weight_pos=2만 제거하고 나머지를 전부 activation으로 분류하면 bias를 잘못 분류한다. `params`에는 bias와 weight가 함께 있다.
* grouped matmul의 `[E]` offset은 activation/weight가 아닌 metadata다. weight 전체의 저장량, 선택 전문가가 읽는 양, 실제 메모리 트래픽은 서로 다른 수치다.

추천 규약은 `input_shape`를 **`operand_shapes`**, `weight_shape`를 **`parameter_storage_shape`**로 설명하고, `weight_pos`의 부호에 역할 분류를 모두 맡기지 않는 것이다. `parameter_operand_index`와 `parameter_transform`(identity/transpose/reshape/slice/absorbed), 필요하면 bias/offset을 포함한 `operand_roles`를 따로 제공하면 된다. 현 스키마를 유지한다면 위 예외를 규약에 명시해야 한다.

FLOPs는 실제 GEMM operands와 contraction을 사용하고, storage parameter 수는 parameter identity로 중복 제거한다. fused norm은 별도 취급한다. gpt-oss addmm는 input 0과 1을 곱하는 연산이 아니다. Llama의 E-batched 계산과 V4/GPT의 routed-pair grouped 계산도 다른 수식을 쓴다. dtype/quantization/alias 정보 없이 정확한 바이트나 GPU 트래픽을 보장하는 표현은 피하는 편이 맞다.

**8. 재확인한 사실과 검토 근거 위치**

| 모델 | prefill / decode 행 | 열 | layer coverage | csv/jsonl 내용 차이 |
|---|---:|---:|---|---:|
| V4-Pro | 224 / 212 | 25 | 61 전부, 중복 없음 | 0 |
| Llama-4 | 69 / 69 | 23 | 48 전부, 중복 없음 | 0 |
| K3 | 15,198 / 2,334 | 24 | 93 전부, 중복 없음 | 0 |
| gpt-oss-120b | 51 / 51 | 22 | 36 전부, 중복 없음 | 0 |
| gpt-oss-20b | 51 / 51 | 22 | 24 전부, 중복 없음 | 0 |

K3 caveat는 양 phase 각각 1,472행, 합계 2,944행이다. 요청서의 계수가 맞다. 20개 csv/jsonl 모두 로컬 main의 대응 발행 파일과 byte 단위로도 같았다. 통과했다고 알려 준 fleet gate나 재추적은 다시 실행하지 않았다.

발행 폴더의 최상위 `README.md`는 아직 “17개 모델”, “외부 검증 완료” 등의 옛 설명이다. 이 snapshot에는 5개 모델만 있고 각 UNKNOWNS는 현재 ③ 평가가 STALE임을 밝힌다. 요청서의 공개 상태와 맞추려면 이 첫 안내도 갱신해야 한다. 이는 산출물 라벨 오류와 별도의 안내 문제다.

소스 줄 번호는 다음 **실제 읽은 로컬 파일**을 기준으로 한다. HF revision만으로 라이브러리 구현까지 고정되는 것은 아니며, 이번 provenance의 Transformers 버전은 모두 5.14.1이다.

* V4: `.venv/Lib/site-packages/transformers/models/deepseek_v4/modeling_deepseek_v4.py`, `configuration_deepseek_v4.py`.
* Llama: `.venv/Lib/site-packages/transformers/models/llama4/modeling_llama4.py`, `configuration_llama4.py`.
* GPT: `.venv/Lib/site-packages/transformers/models/gpt_oss/modeling_gpt_oss.py`.
* grouped experts: `.venv/Lib/site-packages/transformers/integrations/moe.py`.
* K3: `C:/Users/99ktx/.cache/huggingface/hub/models--moonshotai--Kimi-K3/snapshots/f831ab66814297da540d832a5235f8e904f29d06/modeling_kimi_linear.py`, 같은 폴더 `configuration_kimi_k3.py`, `config.json`.
* FLA: `.venv/Lib/site-packages/fla/ops/kda/naive.py`.
* 나머지 config: HF cache의 요청서에 지정된 각 revision `config.json`; 실행 시 전개된 config는 `models/<모델>/full/provenance.json`과 함께 대조했다.

재현 자료: [읽기 전용 수집 스크립트](review_fleet5_artifacts.py), [파일 해시·스케줄·원래 앵커·현재 후보](fleet5_artifacts_evidence.json). 스크립트는 자동으로 소스 의미를 판정하지 않으며, 이 문서의 소스 대조 결론은 별도 검토 결과다.
