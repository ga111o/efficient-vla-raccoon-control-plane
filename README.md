# Efficient VLA Raccoon Control Plane

어떠한 환경에서 진행을 하였는지, 무엇을 변경했는지, 그리고 어떤 결과를 얻었는지 등을 작성한 전체 보고서는 [report/report.pdf](https://github.com/ga111o/efficient-vla-raccoon-control-plane/blob/main/report/report.pdf)에 위치합니다. Raccoonbot 동작 영상은 [report/raccoon.mp4](https://github.com/ga111o/efficient-vla-raccoon-control-plane/blob/main/report/raccoon.mp4)에 위치합니다. 

## repository 구조

```text
efficient-vla-raccoon-control-plane/
├── Raccoonbot_Openvla/       # MuJoCo, OpenVLA, RLDS worker
├── pipeline/
│   ├── pipeline.toml         # Configuration file
│   ├── run_pipeline.py       # 진입점
│   ├── 000_*.py ... 085_*.py # 각 파이프라인 단계
│   └── outputs/              # 실행 로그, manifest, 평가 결과 등
└── environment-train.yml     # 학습/평가 conda 환경
```

## 환경

```bash
conda env create -f environment-train.yml
conda activate openvla-train
```

- `python pipeline/000_check_env.py`을 했을 때, 정상적으로 실행이 되어야 합니다.

## 실행

```bash
python pipeline/run_pipeline.py
```

- 실행 결과는 `pipeline/outputs/runs/<UTC-timestamp>/manifest.json`에 기록됩니다.
- 단계별 exit code, 실행 시간, git SHA가 들어갑니다.


## 파이프라인

현재 `pipeline.toml`에서 켜진 기본 흐름은 다음과 같습니다.


| 단계 | 역할 |
|---|---|
| `000` | 경로, 패키지, GPU 환경 점검 |
| `010` | grasp 데모 생성 |
| `015` | stack 데모 생성 |
| `020` | raw 데모를 RLDS 중간 형식으로 변환 |
| `030` | TFDS 데이터셋 빌드 |
| `040` | 에피소드 프레임 시각화 |
| `050` | 표준 LoRA 파인튜닝 |
| `051` | 소형 GPU용 QLoRA 파인튜닝 |
| `055` | LoRA adapter를 추론용 checkpoint로 병합 |
| `060` | 공개 checkpoint 다운로드 |
| `070` | OpenVLA 추론 HTTP 서버 실행 |
| `075` | MuJoCo 없이 오프라인 추론 검증 |
| `080` | grasp closed-loop 평가 |
| `085` | stack closed-loop 평가와 결과 시각화 |
