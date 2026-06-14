# pipeline/ — 번호형 OpenVLA 파이프라인 스크립트

`raccoon_lab1.ipynb` 의 각 단계를 GPU 서버 터미널에서 비대화식으로 순차 실행할 수 있도록
독립 실행 가능한 번호형 스크립트로 분리한 것. 각 스크립트는 `Raccoonbot_Openvla/` 하위의
기존 worker 를 그대로 호출하는 **얇은 오케스트레이션 래퍼**다 (worker 코드는 수정하지 않음).

## 구성

| 스크립트 | 단계 | worker / 동작 |
|---|---|---|
| `pipeline_common.py` | (공용) | `RB_ROOT`/env 해석, 경로 헬퍼, `run()` subprocess 래퍼 |
| `000_check_env.py` | 환경 점검 | RB_ROOT/TFDS_DATA_DIR·torch/tf/dlimp·nvidia-smi |
| `010_generate_demos.py` | 데모 생성 | `raccoon_grasp_multicolor_scene_dataset.py` |
| `020_convert_rlds_intermediate.py` | RLDS 변환 | `convert_raw_to_openvla_rlds_intermediate.py` |
| `030_build_tfds.py` | tfds 빌드 | `tfds build --overwrite` |
| `040_visualize_episode.py` | 시각화 | 에피소드 프레임 몽타주 PNG 저장 (headless) |
| `050_finetune.py` | 파인튜닝 | `torchrun … vla-scripts/finetune.py` |
| `055_merge_lora.py` | LoRA 병합 | post-hoc LoRA → 병합 체크포인트 |
| `060_download_checkpoint.py` | (옵션) | `hf download …` 공개 체크포인트 |
| `070_serve.py` | 추론 서버 | `openvla_server.py` |
| `075_offline_infer.py` | 오프라인 추론 검증 | `openvla_offline_infer.py` (포트 없이 L2/predict) |
| `080_evaluate.py` | **closed-loop 평가 (grasp)** | `openvla_closed_loop_eval.py` (MuJoCo 롤아웃 성공률) |
| `085_evaluate_stack.py` | **closed-loop 평가 (stack)** | `openvla_closed_loop_eval.py --task stack` + 결과 막대그래프 PNG |

## RB_ROOT 해석

우선순위: 환경변수 `RB_ROOT` → 스크립트 기준 `../Raccoonbot_Openvla` → cwd 기반 fallback.
`TFDS_DATA_DIR` 은 `$RB_ROOT/tensorflow_datasets` 로 자동 설정된다.
다른 위치를 쓰려면: `RB_ROOT=/data/.../Raccoonbot_Openvla python pipeline/000_check_env.py`

## 실행 예시

```bash
# 환경/경로 확인
python pipeline/000_check_env.py

# 짧은 데이터 경로 점검
python pipeline/010_generate_demos.py --num-episodes 4 --num-workers 2
python pipeline/020_convert_rlds_intermediate.py
python pipeline/030_build_tfds.py
python pipeline/040_visualize_episode.py          # -> pipeline/outputs/episode_vis.png

# (GPU) 짧은 파인튜닝 테스트
python pipeline/050_finetune.py --max_steps 5 --save_steps 5

# README(원본) 재현용 본 학습
python pipeline/050_finetune.py --max_steps 30000 --save_steps 1000

# (GPU) 추론 서버
python pipeline/070_serve.py --model_path <ckpt>
```

### 전체 순차 실행

```bash
for s in 000 010 020 030 040 050; do
  python pipeline/${s}_*.py || break      # 와일드카드는 단계당 파일이 하나여야 안전
done
```

각 스크립트는 실패 시(`check=True`) 즉시 비0 종료하므로 `|| break` 로 순차 중단된다.

## 파라미터

모든 스크립트는 자체 `argparse` 를 가지며, default 는 노트북 값이다. `--help` 로 전체 옵션 확인:

```bash
python pipeline/010_generate_demos.py --help
```

지정하지 않은 worker 인자는 worker 자체 default 를 따른다 (얇은 passthrough).

## 설정 파일 (`pipeline.toml`)

단계별 `enabled` / `args` 를 코드(`run_pipeline.py` 의 `STAGES`) 대신 선언적으로 기술한다.
`run_pipeline.py` 가 시작 시 `tomllib`(Py3.11 표준, 의존성 0)로 읽어 `STAGES` 기본값을
**오버라이드**한다. 파일이 없으면 기존 `STAGES` 를 그대로 쓴다(하위호환).

```toml
[stage.010]
enabled = true
args = ["--num-workers", "auto"]   # 코드 수정 없이 인자만 바꿔 실험

[stage.080]
args = ["--num-episodes", "20", "--max-steps", "30"]
```

우선순위: **CLI(`--only`/`--skip`/`--enable` …) > pipeline.toml > 코드 내 STAGES**.
다른 파일을 쓰려면 `--config <path>`. `--list` 는 오버라이드가 반영된 최종값을 보여준다.

## 실행 매니페스트

매 실행마다 `pipeline/outputs/runs/<UTC-timestamp>/manifest.json` 에 다음을 기록한다
(실패·시그널 경로에서도 `finally` 로 부분 기록):

- 전체: `start`/`end`(ISO), `duration_s`, `git_sha`(`git rev-parse HEAD`), `config`,
  `selected_stages`, `status`(ok/failed/interrupted/dry-run).
- 단계별: `{id, script, args, cmd, exit_code, duration_s, start, end}`.

콘솔에는 단계별 `(▲ 12.3s)` 와 종료 시 총 소요시간이 출력된다. `--no-manifest` 로 끌 수 있다.

## Closed-loop 평가 (080)

`075` 는 데모 프레임 1장의 예측-정답 L2 거리만 본다. `080` 은 정책을 MuJoCo 멀티컬러
씬에서 **N회 롤아웃**하여 실제 grasp 성공률(수집과 동일한 touch-grasp 기준)을 측정한다.
worker(`openvla_closed_loop_eval.py`)는 `OpenVLAServingModel` 을 in-process 로 1회 로드
(포트 불필요)하고, 7D→4DOF 매핑은 `raccoon_env.py` 의 `execute_delta_action7` 을 재사용한다.

```bash
# 050(파인튜닝)→055(병합) 후 짧게 검증
python pipeline/080_evaluate.py --num-episodes 5 --max-steps 30
# 결과: pipeline/outputs/eval/eval_report.json (색상별/전체 success_rate,
#       steps-to-success, 평균 추론 latency). 베이스 vs 파인튜닝 성공률로 학습 효과 정량화.
```

`085_evaluate_stack.py` 는 같은 서버를 띄운 채 `--task stack` worker 를 굴려
`eval_stack_report.json` 을 쓴 뒤, **두 종류의 시각화**를 함께 만든다(둘 다 040 과 동일하게
headless matplotlib `Agg` — 085 메인 프로세스엔 torch/mujoco 없음):

1. **집계 막대그래프** `eval_stack_report.png` — 색상별/전체 success_rate(+ 전체 평균
   기준선), 색상별 평균 steps-to-success, 상단에 전체 성공/평균 steps/추론 latency/모델명.
2. **프레임 몽타주** `stack_rollouts/montages/ep_*.png` — worker 가 015 데이터셋과 동일
   구조(`ep_*/frame_*.png` + `meta.json`)로 남긴 롤아웃 프레임을, `040` 처럼 에피소드당
   몇 장을 균등 샘플해 한 줄로 묶는다(제목: instruction/색/성공/steps). **각 에피소드가
   step 마다 어떻게 움직였는지**를 한 장에 본다.

```bash
python pipeline/085_evaluate_stack.py --num-episodes 4 --max-steps 150
# -> pipeline/outputs/eval/eval_stack_report.{json,png}
# -> pipeline/outputs/eval/stack_rollouts/ep_*/frame_*.png (+ meta.json)
# -> pipeline/outputs/eval/stack_rollouts/montages/ep_*.png  (프레임 몽타주)
# 막대그래프만 끄기: --no-report-vis / 몽타주만 끄기: --no-frame-montage
# 몽타주 프레임 수: --frame-montage-frames 8
# 기존 산출물(JSON·프레임)만 다시 그리기(GPU 불필요, 개발 환경 OK): --vis-only
```

GPU 필요(모델 추론) → GPU 서버에서만 실행/검증. 단 `085 --vis-only` 는 추론 없이 기존
JSON·프레임만 그리므로 비 GPU 환경에서도 동작한다.

## GPU 메모리 절감 (050, 속도·품질 무손실)

`050_finetune.py` 는 batch 를 키우지 않고도 fine-tune peak GPU 메모리를 줄이는 무손실
기법을 **기본 on** 으로 적용한다(품질·속도 하락 없음):

- **Flash Attention 2** (`--attn-implementation`, 기본 `flash_attention_2`): LLM attention 의
  O(seq²) score 버퍼를 만들지 않아 activation 메모리↓ + 커널 융합으로 더 빠름(근사 아닌 exact).
  베이스라인 비교는 `--attn-implementation eager`.
- **expandable_segments 할당자** (`PYTORCH_CUDA_ALLOC_CONF`, 050 가 자동 설정): 단편화로 인한
  reserved-but-unused 메모리↓. 쉘에 이미 설정돼 있으면 존중한다.
- **8-bit optimizer** (`--use-8bit-optimizer`, 기본 off): LoRA optimizer state 를 bitsandbytes
  AdamW8bit 로 → optimizer 메모리↓(무손실). LoRA 파라미터만이라 절감폭은 중간.
- **DDP** (`--no-ddp-find-unused-parameters`): all-linear LoRA 는 모든 adapter 를 매 step 쓰므로
  `find_unused_parameters=False` 가 안전하고 약간 더 가볍다(기본은 현행 보존 True).

학습 로그/W&B/ClearML 에 `gpu_peak_alloc_gb`, `gpu_peak_reserved_gb`, `step_time_s` 가
함께 기록되므로 A/B(eager vs FA2 등)로 메모리·속도를 정량 비교할 수 있다.

```bash
# 베이스라인(eager) vs 기본(FA2) 메모리/속도 비교 — GPU 서버에서만
python pipeline/050_finetune.py --max_steps 50 --save_steps 1000 --attn-implementation eager
python pipeline/050_finetune.py --max_steps 50 --save_steps 1000           # FA2 기본
```

## 주의

- 모든 스크립트는 현재 venv 의 `sys.executable` 을 사용한다 → GPU 서버에 설치된 패키지를 그대로 쓴다.
- `050_finetune.py` 는 포그라운드 `torchrun` 실행이다. 백그라운드가 필요하면
  `nohup python pipeline/050_finetune.py ... > torchrun.log 2>&1 &` 로 감싼다.
- **ClearML 실험 추적**: repo 루트 `.env` 에 `CLEARML_API_HOST/FILES_HOST/WEB_HOST`,
  `CLEARML_API_ACCESS_KEY`, `CLEARML_API_SECRET_KEY` 가 있으면 `050_finetune.py` 가
  자동으로 켠다(자격증명 감지 시 default on). `train_loss`/`action_accuracy`/`l1_loss`
  스칼라와 하이퍼파라미터가 `https://app.clear.ml` 의 `openvla-raccoon` 프로젝트로
  올라간다. `--no-clearml` 로 끄고, `--clearml-project <name>` 으로 프로젝트명을 바꾼다.
  (`.env` 는 `.gitignore` 에 있어 커밋되지 않는다.)
- `040_visualize_episode.py` 는 headless 대응으로 matplotlib `Agg` 백엔드 + `fig.savefig()` 를 쓴다.
- 개발(비 GPU) 환경에서는 `--help` / `000_check_env.py` 의 경로·import 확인까지만 가능, 실제 실행은 GPU 서버에서.
