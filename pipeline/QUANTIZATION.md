# OpenVLA 양자화 — RTX 5080 (Blackwell GB203, 16GB)

OpenVLA-7B 를 RTX 5080 한 장(16GB)에서 **학습부터 서빙까지** 돌리기 위한 비대칭 혼합
정밀도(Asymmetric Mixed Precision) 구현. 과제 §1~§4 의 매핑과 실제 라이브러리 제약을
함께 정리한다.

GPU 서버는 무시하고 **현재 개발 환경(RTX 5080 + torch 2.12.0+cu130)** 에서 수행한다.

---

## 1. 핵심: 모듈별 비대칭 정밀도 정책

전역(uniform-bit) 양자화는 **금지**. VLA 는 모듈별 양자화 민감도가 극도로 비대칭이다.
정책은 [`prismatic/quant/policy.py`](../Raccoonbot_Openvla/openvla/prismatic/quant/policy.py)
한 곳에 모았고 학습·서빙이 공유한다.

| 모듈 | OpenVLA FQN | 학습(051) | 서빙(070) |
|---|---|---|---|
| Vision Encoder (DINOv2+SigLIP) | `vision_backbone.featurizer` / `fused_featurizer` | **BF16** (+LoRA r=16) | **INT8** PTQ |
| Projector (MLP) | `projector.fc1/fc2/fc3` | **BF16 full FT** | **BF16** (고정) |
| LLM Backbone (Llama-2 7B) | `language_model.model.*` | **FP8 동결** + LoRA r=32 | **FP8 W8A8** |
| **Action Head** | **`language_model.lm_head`** | **BF16 full FT** | **BF16** (고정) |

### ⚠️ 가장 중요한 정정: Action Head = `lm_head`

OpenVLA 에는 **별도 action head 모듈이 없다.** 행동은 `lm_head` 출력 logits 중 **마지막
256개 vocab bin** 을 argmax → 역정규화해 만든다
([`predict_action`](../Raccoonbot_Openvla/openvla/prismatic/extern/hf/modeling_prismatic.py#L506-L536)).
따라서 "Action Head = BF16 고정" 은 곧 **LLM 백본을 FP8 로 양자화할 때 `lm_head` 를
반드시 제외**하라는 뜻이다. 구현은 `quantize_` 를 `language_model.model`(Llama transformer
body)에만 걸어, 형제 모듈 `language_model.lm_head` 가 자동으로 BF16 으로 남게 한다.

---

## 2. 필요 라이브러리 (conda)

학습/서빙이 한 conda 환경에서 돈다. `transformers==4.40.1` 핀(OpenVLA vendored 모델이
요구)을 **유지**하고, 양자화는 **로드 후(post-load)** `torchao.quantize_()` 로 적용해
transformers 버전 의존을 끊었다.

| 패키지 | 용도 | 비고 |
|---|---|---|
| `torch` 2.12+cu130 | (설치됨) | Blackwell sm_120 |
| `torchao` (최신) | FP8/INT 양자화 | Blackwell 네이티브 FP8. 학습·서빙 공통 |
| `bitsandbytes` (≥0.45) | 폴백 QLoRA + 8-bit optimizer | sm_120 커널 포함 버전 |
| `peft`, `accelerate`, `timm`, `draccus` | (설치됨) | LoRA/학습/비전/CLI |
| `fastapi`, `uvicorn`, `pydantic`, `Pillow` | (설치됨) | 서빙 |
| `scipy` | (선택) FAST action tokenizer DCT | `--use-fast-tokenizer` 쓸 때만 |

vLLM / TensorRT-LLM 은 **불필요** — 서빙은 PyTorch 네이티브 FP8(기존 FastAPI 서버 확장).

```bash
# 예 (conda 환경 안에서)
pip install torchao bitsandbytes scipy
```

---

## 3. 파이프라인 1단계: 샌드위치 QLoRA 파인튜닝

[`051_finetune_sandwich.py`](051_finetune_sandwich.py) →
[`vla-scripts/finetune_sandwich.py`](../Raccoonbot_Openvla/openvla/vla-scripts/finetune_sandwich.py).
기존 [`050_finetune.py`](050_finetune.py)(all-linear LoRA + DDP)의 **대안**이며 단일 GPU 전용.

- LLM 백본: **FP8 weight-only 로 동결**(≈7.5GB) + 각 층 Q,V+선형에 **LoRA r=32 (BF16 어댑터)**.
- Vision: **BF16** + **LoRA r=16**(`rank_pattern` 으로 LLM r=32 와 분리). `--no-vision-lora` 로 완전 동결.
- Projector / Action head(lm_head): PEFT `modules_to_save` 로 **full BF16 학습**(저장·병합 시 보존).
- 메모리: gradient checkpointing + bnb 8-bit optimizer + FA2 기본 on. `batch 4 × accum 4`(유효 16).

```bash
# 050 끄고 051 켜기 (또는 pipeline.toml 에서 050.enabled=false / 051.enabled=true)
python pipeline/051_finetune_sandwich.py \
    --backbone-precision fp8 --quant-backend torchao \
    --vision-precision bf16 --lora_rank 32 --vision_lora_rank 16 \
    --batch_size 4 --grad_accumulation_steps 4 --max_steps 11000 --save_steps 1000

# torchao+PEFT 결합이 불안정하면 폴백 (검증된 QLoRA, INT8 ≈7.5GB / NF4 ≈4GB):
python pipeline/051_finetune_sandwich.py --quant-backend bitsandbytes --backbone-precision int8
```

학습은 어댑터(+`projector`/`lm_head` full weights)를 `adapter_dir` 에 저장한다. 서빙용
단일 체크포인트가 필요하면 **BF16 base 에 병합**(무손실, FP8 가 아님):

```bash
python pipeline/055_merge_lora.py          # last_run.json 의 adapter_dir → run_dir 병합
```

### 토크나이저 최적화(§3, 권장) — 현재 범위

`--use-fast-tokenizer` 플래그는 존재하지만, FAST action tokenizer 는 256-bin action head
경로(가변 길이 토큰·collator·decode·`predict_action` 전반)를 바꾸는 **별도 큰 작업**이라
이번 구현에서는 **문서화된 확장점**으로 둔다(기본 256-bin ActionTokenizer 로 진행). OFT
계열 메모리 절감(grad checkpointing, 8-bit optimizer, FA2)은 이미 기본 적용된다.

---

## 4. 파이프라인 2단계: FP8 W8A8 경량 서빙 (Option A 계열)

[`070_serve.py`](070_serve.py) → [`openvla_server.py`](../Raccoonbot_Openvla/openvla/openvla_server.py).
기존 FastAPI 서버를 확장해 **비대칭 FP8 W8A8** 로 적재한다. (OpenVLA 의 이중 vision backbone
Prismatic 아키텍처는 vLLM 모델 레지스트리에 없어, 진짜 vLLM 서빙은 커스텀 모델 등록이
필요하다. PyTorch 네이티브 FP8 서버가 추가 의존성 0 으로 즉시 동작하며 결정론적 디코딩을 보장.)

- 백본 **FP8 W8A8**(`Float8DynamicActivationFloat8Weight`) — Blackwell 5세대 텐서코어 가속.
- Vision **INT8** PTQ, projector/lm_head **BF16 고정**.
- 디코딩: `do_sample=False`(서버 기본) → 결정론적 greedy. 제어주파수 10~30Hz 방어.
- BF16 14.5GB 선적재 OOM 회피: CPU 로드 후 `quantize_(device="cuda")` 가 layer-by-layer 로 변환.

```bash
python pipeline/070_serve.py --host 0.0.0.0 --port 8000 \
    --precision fp8_w8a8 --vision-precision int8
# 포트 없이 검증: 075_offline_infer.py 가 같은 OpenVLAServingModel 을 in-process 로 로드
```

서버 기동 로그에 `summarize_precision` 표가 찍혀 **비대칭 정책이 실제로 먹었는지**(LLM 만
FP8/INT, projector·lm_head 는 BF16) + 총 weight VRAM(목표 ≈9GB, 16GB 안)을 확인할 수 있다.

---

## 5. 스펙 대비 의도적 편차 (정직하게)

| 스펙 | 구현 | 이유 |
|---|---|---|
| 학습 시 백본 "FP8 W8A8 동결" | 학습은 FP8 **weight-only** 동결, 서빙은 **W8A8** | 동결 백본 + BF16 LoRA(QLoRA 패턴)에는 weight-only 가 안전·정석. W8A8 의 활성 FP8 가속은 추론 시 이득이라 서빙에 적용. 가중치는 동일 FP8 |
| Option A: vLLM | PyTorch 네이티브 FP8 FastAPI 서버 | OpenVLA(Prismatic 이중 backbone)는 vLLM 레지스트리 미지원 → 커스텀 등록 필요. 경량 서버가 추가 의존성 0 으로 즉시 동작 + 결정론 보장 |
| Vision INT4/INT8/BF16 | 학습 BF16(+LoRA r16), 서빙 INT8 | 학습 중 vision 미세조정하려면 BF16 필요. 양자화는 서빙 PTQ 로 |
| FAST tokenizer + OFT | OFT(메모리 절감)만 적용, FAST 는 확장점 | FAST 는 action head/데이터 경로 대수술 — §3 "권장" 범위로 분리 |

`--quant-backend bitsandbytes` 폴백은 INT8/NF4(FP8 아님)지만, torchao+PEFT FP8 결합이
설치 스택에서 불안정할 때 **확실히 학습되는** 경로를 제공한다.

---

## 6. 빠른 점검 순서 (GPU 필요)

```bash
# 1) 정책 모듈 단독 import 확인 (torchao 설치 후)
cd Raccoonbot_Openvla/openvla && python -c "from prismatic.quant import QuantPolicy; print('ok')"
# 2) 짧은 샌드위치 학습 (메모리/정책 리포트 확인)
python pipeline/051_finetune_sandwich.py --max_steps 5 --save_steps 5
# 3) 병합 → FP8 서빙 → 오프라인 추론 검증
python pipeline/055_merge_lora.py
python pipeline/070_serve.py --precision fp8_w8a8 &
python pipeline/075_offline_infer.py --mode eval --num-steps 3 --stack
```
