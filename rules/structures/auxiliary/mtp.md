# MTP — Multi-Token Prediction head (보조 진입점)

## 정의
메인 next-token 예측에 더해, 여러 미래 토큰을 예측하는 **보조 헤드**. 학습 신호 강화 및
추론 시 speculative decoding에 쓰인다. 메인 forward 밖에서 별도로 호출되는 보조 모듈이라
`01-main.md` Step 3의 "추가 진입점(extra entrypoint)" 대상.

## 관련 심볼 / config
`num_nextn_predict_layers`(= `num_mtp_layers`) > 0 이면 config상 MTP가 선언된 것.
DeepSeek-V3 계열이 대표.

## 트레이스에서 식별하는 방법 — 중요한 한계 (Phase 8 실측)
- **네이티브 transformers 구현(`deepseek_v3` 등)은 표준 `AutoModelForCausalLM`에 MTP 헤드를
  만들지 않는다.** config에 `num_mtp_layers=1`이 있어도 `model.named_modules()`에 mtp/nextn
  모듈이 0개 → introspect가 발견할 게 없다.
- 우리는 native impl을 우선 로드하므로([../attention/mla.md], [[native-impl-over-remote-code]]
  정책), MTP는 트레이싱되지 않는다. 모델 저자의 remote 코드는 MTP를 빌드하지만 transformers
  5.x에서 import가 깨진다.
- **C15가 이 갭을 WARN으로 표면화한다**: "config declares N MTP layer(s) but no MTP module in
  the traced model (native impl omits MTP head) — MTP NOT traced". 메인 forward는 완전히
  트레이싱되므로 FAIL이 아니라 WARN(P8: 숨기지 않고 드러냄).
- 사용자 결정(Phase 8): 네이티브 한계를 수용하고 C15 WARN으로 문서화. MTP를 실제로
  트레이싱하려면 remote 코드 shim 또는 MTP 전용 input_builder를 별도 진입점으로 구성해야
  한다(README 의 프로파일 `extra_entrypoints`) — 현재 범위 밖.

## 확인된 모델
- **`bzantium/tiny-deepseek-v3`** (Phase 8): `num_mtp_layers=1` 선언, native impl에서 MTP 모듈
  0개 → C15 WARN. deepseek_v3 아키텍처(MLA+MoE) 메인 forward는 전부 통과.
- Phase 9(DeepSeek-V3 실사이즈), 예약된 DeepSeek-V4 계열도 동일하게 적용될 전망.

## 참고 소스
- transformers `DeepseekV3Config` docstring (`num_nextn_predict_layers`)
- DeepSeek-V3 Technical Report — MTP 설계(교차검증용)
