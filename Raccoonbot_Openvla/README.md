# Raccoonbot_Openvla

⭐ 1~3번은 직접 finetuning을 진행하는 내용이니 체크포인트를 불러와서 사용하는 경우 0번과 4번만 진행<br>

0~3번 server에서 실행, 4번 local-server 실행<br>


## 0. Dependencies

> **경로 기준:** 아래 모든 명령은 절대경로 대신 `$RB_ROOT`(= `Raccoonbot_Openvla` 디렉토리) 기준이다.
> repo 를 clone/압축해제한 곳에서 한 번만 export 하면 위치에 상관없이 동작한다.
> ```bash
> git clone https://github.com/KWU-FAIR-LAB/Raccoonbot_Openvla.git
> cd Raccoonbot_Openvla
> export RB_ROOT=$(pwd)
> ```

시스템 OpenGL 라이브러리 (MuJoCo 렌더링용, Debian/Ubuntu 기준):
```
apt update
apt install -y \
  libegl1 \
  libgl1 \
  libglvnd0 \
  libglx0 \
  libopengl0 \
  libgles2 \
  libegl1-mesa \
  libegl1-mesa-dev \
  mesa-utils
```

Python 환경 (Python 3.10+ 권장, 3.12 에서 검증됨):
```
cd $RB_ROOT
python3 -m venv env
source env/bin/activate

# 1) 본인 GPU 드라이버의 CUDA 버전에 맞는 torch 를 *먼저* 설치한다 (pyproject/lock 에 휠을 고정하지 않음).
#    드라이버의 CUDA 버전은 `nvidia-smi` 우상단에서 확인.
#      CUDA 12.x (예: 12.2):  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
#      CUDA 13.x:             pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
#    ⚠ 인덱스를 생략하고 그냥 `pip install torch` 하면 최신 CUDA(현재 cu130) 빌드가 깔려
#       구형 드라이버에서 "NVIDIA driver is too old" 로 GPU 를 못 쓴다.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2) dlimp 를 상대경로 editable 로 설치 (과거의 file:///data/... 절대경로 의존성 대체)
pip install -e $RB_ROOT/dlimp_openvla

# 3) openvla(본 패키지)를 editable 로 설치
cd $RB_ROOT/openvla
pip install -e .
```

> **정확 재현이 필요하면** 위 `pip install -e .` 대신 검증된 버전 스냅샷을 사용:
> ```
> # 1) torch 는 위와 동일하게 본인 CUDA 인덱스에서 먼저 설치 (lock 에는 torch 가 없음)
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> # 2) 나머지 (CUDA 무관) 의존성
> pip install -r $RB_ROOT/openvla/requirements-lock.txt
> # 3) 로컬 editable 패키지 (dlimp 누락 시 finetune.py 가 `No module named 'dlimp'` 로 실패)
> pip install -e $RB_ROOT/dlimp_openvla
> pip install -e $RB_ROOT/openvla
> ```
> `requirements-lock.txt` 에는 torch/torchvision/`nvidia-*-cuXX` 같은 CUDA 전용 패키지가 빠져 있다 (머신마다 CUDA 가 달라 이식성을 위해 제외). 그래서 torch 는 반드시 1) 단계에서 따로 설치해야 한다.

> **파인튜닝(3번)을 직접 할 경우에만** flash-attention-2 를 *editable 설치 후* 추가로 설치 (GPU + nvcc 필요):
> ```
> pip install flash-attn --no-build-isolation
> ```
> 추론(4번)만 한다면 생략 가능.

## 1. Dataset 생성
MuJoCo 가상환경에서 finetuning을 위한 데이터를 수집 <br>
(main 함수 `num_episodes`으로 dataset sample 수 변경 가능)
```
cd $RB_ROOT/Mujoco
python raccoon_grasp_multicolor_scene_dataset.py
```
실행하면 $RB_ROOT/Mujoco/raccoon_grasp_colored_cylinder 하위에 episode별로 dataset png 확인 가능

## 2. rlds 파일 변환
raw data를 rlds builder에 맞게 변경
아래 명령문 그대로 실행
```
cd $RB_ROOT/Mujoco/raccoon_dataset
python convert_raw_to_openvla_rlds_intermediate.py \
--raw_root $RB_ROOT/Mujoco/raccoon_grasp_colored_cylinder \
--out_root $RB_ROOT/Mujoco/raccoon_dataset/openvla_rlds_intermediate \
--val_ratio 0.1
```

## 2-1. rlds builder
rlds builder 실행
아래 명령문 그대로 실행
```
# tfds 산출물이 repo 안($RB_ROOT/tensorflow_datasets)에 바로 쌓이도록 지정
#  → 과거의 `mv /root/tensorflow_datasets /data/...` 이동 단계가 사라진다.
export TFDS_DATA_DIR=$RB_ROOT/tensorflow_datasets

cd $RB_ROOT/Mujoco/rlds_dataset_builder/raccoon_pick_place
tfds build --overwrite
```
실행하면 `$RB_ROOT/tensorflow_datasets/raccoon_pick_place` 하위에 데이터셋이 생성됨

## 3. Raccoonbot 기반 OpenVLA finetuning
아래 명령어 그대로 실행 <br>
(`max_steps`, `save_steps` 변경 가능)
```
cd $RB_ROOT/openvla
export PYTHONPATH=$RB_ROOT/openvla:$PYTHONPATH

WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0 \
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir $RB_ROOT/tensorflow_datasets \
  --dataset_name raccoon_pick_place \
  --run_root_dir $RB_ROOT/openvla/openvla-runs \
  --adapter_tmp_dir $RB_ROOT/openvla/openvla-adapter-tmp \
  --lora_rank 32 \
  --batch_size 8 \
  --grad_accumulation_steps 2 \
  --learning_rate 5e-4 \
  --max_steps 30000 \
  --save_steps 30000 \
  --run_id_note raccoon-eef-v100
```

## 4. Mujoco 환경 Inference (local-server)
1~3번을 진행했다면 4-1은 건너뛰고 이후 명령어에서 본인이 finetuning한 모델 경로로 modelpath를 변경하여 진행

## 4-1. Hugging Face에서 RaccoonBot finetuned OpenVLA 모델 다운로드
서버에서 terminal에 아래 명령어를 입력하여 모델 다운로드
```
pip install -U huggingface_hub

hf download fair-lab/openvla-7b-finetuned-raccoonbot --local-dir $RB_ROOT/openvla/openvla-runs/openvla-7b-finetuned-raccoonbot
``` 

## 4-2. 서버측 코드 실행
server 실행 명령문<br>
만약 1~3번을 진행하여 직접 finetuning했다면 model path를 openvla-runs/ 아래에 있는 모델 디렉토리로 변경하고 진행<br>
```
cd $RB_ROOT/openvla
CUDA_VISIBLE_DEVICES=0 python openvla_server.py \
  --model_path $RB_ROOT/openvla/openvla-runs/openvla-7b-finetuned-raccoonbot \
  --default-unnorm-key raccoon_pick_place \
  --host 0.0.0.0 \
  --port 8000 \
  --device cuda
```

## 4-3. 클라이언트측에서 실행할 환경 설정
클라이언트측 코드와 MuJoCo xml 파일 [다운로드](https://drive.google.com/drive/folders/1xrH3FoTfKC9CiUE-kDRorxTKMMq0O7Px?usp=sharing) 후 압축 풀기 <br>
파일: openvla_multicolor_client.py, openvla_multicolor_client_real_robot.py, raccoon_env.py, Raccoon_colored_cylinder.xml, RaccoonBot_S.xml, requirements.txt

> `raccoon_env.py`, `Raccoon_colored_cylinder.xml`, `RaccoonBot_S.xml` 는 이미 `$RB_ROOT/Mujoco` 에 있으므로,
> Drive 에서 받은 클라이언트 코드(`openvla_multicolor_client*.py`)를 `$RB_ROOT/Mujoco` 에 풀어 함께 두면 된다.

압축 푼 폴더에서 terminal 환경설정
```
cd $RB_ROOT/Mujoco
pip install -r requirements.txt
```

## 4-4. 클라이언트측 코드 실행
target_color를 **[red, blue, green, yellow]** 로 수정하면 그에 맞게 prompt가 변경됨

⭐ local 실행 명령문
```
python openvla_multicolor_client.py --server_url http://127.0.0.1:8000 --xml_path Raccoon_colored_cylinder.xml --target_color red --use_viewer
```

## 4-5. 실제 라쿤봇을 연결하여 실행
openvla_multicolor_client_real_robot.py를 실행하면 MuJoCo 환경에서 동작하는 Action을 로봇이 동일하게 수행

⭐ local 실행 명령문
```
python openvla_multicolor_client_real_robot.py --server_url http://127.0.0.1:8000 --target_color red --use_real_robot --use_viewer
```
