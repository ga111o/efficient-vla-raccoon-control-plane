"""prismatic.quant.policy — OpenVLA 비대칭 혼합 정밀도(Asymmetric Mixed Precision) 정책.

RTX 5080 (Blackwell GB203, 16GB) 한 장에서 OpenVLA-7B 를 학습/서빙하기 위한 모듈별
양자화 정책을 한 곳에 모은다. **전역(uniform-bit) 양자화는 금지**하고, 모듈별 민감도에
따라 정밀도를 비대칭으로 적용한다(과제 §2).

OpenVLAForActionPrediction 의 서브모듈 ↔ 정책 매핑
--------------------------------------------------------------------------------
  vision_backbone.featurizer        DINOv2  ┐ 노이즈에 강건 → 공격적 양자화 허용
  vision_backbone.fused_featurizer  SigLIP  ┘ (학습 중엔 BF16, 서빙 시 INT8/INT4 PTQ)
  projector.fc1/fc2/fc3             MLP        모달리티 정렬 → **절대 양자화 금지(BF16)**
  language_model.model.*            Llama-2    FP8 (W8A8) — VRAM 절감 핵심 표적
  language_model.lm_head            ⚠️ Action Head     **절대 양자화 금지(BF16)**

핵심(자주 틀리는 부분): OpenVLA 는 별도 action head 모듈이 없다. 행동은 `lm_head` 의
출력 logits 중 **마지막 256개 vocab bin** 을 argmax → 역정규화해서 만든다
(modeling_prismatic.OpenVLAForActionPrediction.predict_action). 따라서 "Action Head =
BF16 고정" 은 곧 **`language_model.lm_head` 를 LLM 백본 FP8 양자화에서 제외**하라는 뜻이다.
편의상 `quantize_` 를 `language_model.model`(=Llama transformer body)에만 걸면 형제
모듈인 `lm_head` 는 자동으로 BF16 으로 남는다.

백엔드 두 가지
--------------------------------------------------------------------------------
  - torchao (기본, 스펙 정확): 로드 후(post-load) `quantize_()` 로 서브트리별 적용.
      Blackwell 5세대 텐서코어의 하드웨어 네이티브 FP8 을 사용한다. transformers
      4.40.1 핀을 건드리지 않으므로 OpenVLA 의 vendored 모델 로딩과 충돌이 없다.
  - bitsandbytes (폴백): `from_pretrained(quantization_config=...)` 로 로드 시 적용.
      INT8(LLM.int8 W8A8) / NF4. `llm_int8_skip_modules` 로 vision·projector·lm_head 를
      건너뛰어 비대칭 정책을 흉내낸다(검증된 QLoRA 경로).

이 모듈은 torch 만 있으면 import 된다. torchao/bitsandbytes 는 실제 적용 함수 안에서
lazy import 하고, 없으면 설치 안내가 담긴 명확한 에러를 던진다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn

# --- 서브모듈 그룹 식별용 FQN 규칙 -------------------------------------------------
VISION_PREFIX = "vision_backbone."
PROJECTOR_PREFIX = "projector."
LLM_BODY_PREFIX = "language_model.model."   # Llama transformer (lm_head 제외)
LM_HEAD_NAME = "language_model.lm_head"      # = OpenVLA 의 Action Head (절대 양자화 금지)

# torchao 정밀도 토큰 → 설명. 스크립트/서버가 이 문자열로 정밀도를 고른다.
#   학습:  backbone 기본 "fp8"(weight-only, 동결 백본 ≈7.5GB) — QLoRA 패턴.
#   서빙:  backbone 기본 "fp8_w8a8"(동적 활성 + FP8 가중치) — Blackwell FP8 가속 최대.
TORCHAO_PRECISIONS = {"fp8_w8a8", "fp8", "fp8_wo", "int8_w8a8", "int8", "int4", "bf16"}
BNB_PRECISIONS = {"int8", "nf4", "bf16"}


@dataclass
class QuantPolicy:
    """모듈별 정밀도 정책. projector / action_head 는 스펙상 BF16 고정(변경 금지)."""

    backend: str = "torchao"            # "torchao" | "bitsandbytes"
    backbone_precision: str = "fp8"     # language_model.model.* (lm_head 제외)
    vision_precision: str = "bf16"      # vision_backbone.* (학습:bf16, 서빙:int8/int4)
    int4_group_size: int = 128          # torchao Int4WeightOnly group size
    # --- 스펙 고정값(투명성을 위해 노출하되 바꾸지 말 것) ---
    projector_precision: str = "bf16"
    action_head_precision: str = "bf16"

    def validate(self) -> "QuantPolicy":
        if self.backend == "torchao":
            allowed = TORCHAO_PRECISIONS
        elif self.backend == "bitsandbytes":
            allowed = BNB_PRECISIONS
        else:
            raise ValueError(f"알 수 없는 backend={self.backend!r} (torchao|bitsandbytes)")
        for field_name in ("backbone_precision", "vision_precision"):
            val = getattr(self, field_name)
            if val not in allowed:
                raise ValueError(
                    f"backend={self.backend} 에서 {field_name}={val!r} 미지원. "
                    f"가능: {sorted(allowed)}"
                )
        if self.projector_precision not in ("bf16", "fp32"):
            raise ValueError("projector 는 BF16/FP32 만 허용(양자화 금지) — 스펙 §2")
        if self.action_head_precision != "bf16":
            raise ValueError("action head(lm_head) 는 BF16 고정 — 스펙 §2")
        return self


# ============================================================================
#  torchao 경로 — 로드 후 서브트리별 quantize_()
# ============================================================================
def _torchao_config(precision: str, group_size: int = 128):
    """정밀도 토큰 → torchao AOBaseConfig 객체. bf16 이면 None(=양자화 안 함)."""
    if precision == "bf16":
        return None
    try:
        from torchao.quantization import (
            Float8DynamicActivationFloat8WeightConfig,
            Float8WeightOnlyConfig,
            Int4WeightOnlyConfig,
            Int8DynamicActivationInt8WeightConfig,
            Int8WeightOnlyConfig,
        )
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "torchao 가 필요합니다 (FP8/INT 양자화). conda 환경에 설치하세요:\n"
            "    pip install torchao   # Blackwell sm_120 FP8 지원 최신 버전\n"
            f"(원본 에러: {e})"
        ) from e

    mapping = {
        "fp8_w8a8": lambda: Float8DynamicActivationFloat8WeightConfig(),
        "fp8": lambda: Float8WeightOnlyConfig(),
        "fp8_wo": lambda: Float8WeightOnlyConfig(),
        "int8_w8a8": lambda: Int8DynamicActivationInt8WeightConfig(),
        "int8": lambda: Int8WeightOnlyConfig(),
        "int4": lambda: Int4WeightOnlyConfig(group_size=group_size),
    }
    return mapping[precision]()


def _is_linear(module: nn.Module, _fqn: str = "") -> bool:
    return isinstance(module, nn.Linear)


def apply_torchao_quantization(
    vla: nn.Module, policy: QuantPolicy, device: Optional[str] = "cuda"
) -> Dict[str, str]:
    """OpenVLA(BF16 로 로드된) 모델에 비대칭 정밀도를 in-place 로 적용한다.

    - language_model.model.*  → backbone_precision (lm_head 는 형제라 자동 제외)
    - vision_backbone.*        → vision_precision
    - projector / lm_head      → 손대지 않음(BF16 유지)

    서브트리(.language_model.model, .vision_backbone)에만 quantize_ 를 걸어
    FQN 필터 시그니처의 torchao 버전차에 견고하게 만든다.

    `device` 를 주면 quantize_ 가 모듈을 하나씩 그 디바이스로 옮기며 양자화한다
    (최신 torchao). 7B 를 BF16 전체로 16GB GPU 에 먼저 올리면 OOM(≈14.5GB, 스펙 §1)
    이므로, CPU 로 로드한 모델을 layer-by-layer 로 FP8 변환하며 GPU 로 올린다. 구버전
    torchao 가 device 인자를 모르면 device 없이 in-place 양자화로 폴백한다(이미 올라간
    디바이스에서 수행). 반환값은 적용 리포트.
    """
    policy.validate()
    from torchao.quantization import quantize_  # lazy

    def _quantize(module: nn.Module, cfg) -> None:
        if cfg is None:
            return
        try:
            quantize_(module, cfg, filter_fn=_is_linear, device=device)
        except TypeError:
            quantize_(module, cfg, filter_fn=_is_linear)  # 구버전 폴백

    report: Dict[str, str] = {
        "projector": "bf16 (locked)",
        "action_head(lm_head)": "bf16 (locked)",
    }

    # LLM 백본 (lm_head 제외) — FP8 W8A8 표적
    llm_cfg = _torchao_config(policy.backbone_precision, policy.int4_group_size)
    _quantize(vla.language_model.model, llm_cfg)
    report["language_model.model.*"] = (
        policy.backbone_precision if llm_cfg is not None else "bf16 (no quant)"
    )

    # Vision encoder (DINOv2 + SigLIP)
    vis_cfg = _torchao_config(policy.vision_precision, policy.int4_group_size)
    _quantize(vla.vision_backbone, vis_cfg)
    report["vision_backbone.*"] = (
        policy.vision_precision if vis_cfg is not None else "bf16 (no quant)"
    )

    return report


# ============================================================================
#  bitsandbytes 경로 — from_pretrained(quantization_config=...) 용 설정 빌더
# ============================================================================
def build_bnb_config(policy: QuantPolicy):
    """비대칭 정책을 흉내내는 BitsAndBytesConfig 를 만든다.

    bnb 는 단일 양자화 스킴만 적용하므로(서브트리별 정밀도 불가), LLM 백본만 양자화하고
    vision·projector·lm_head 는 `llm_int8_skip_modules` 로 건너뛰어 BF16 으로 남긴다.
    (vision 은 이 경로에서 BF16 고정 — 서빙 시 torchao 로 따로 INT8 PTQ 하면 된다.)

    backbone_precision: "int8"(LLM.int8 W8A8) | "nf4" | "bf16"(=None).
    """
    policy.validate()
    if policy.backbone_precision == "bf16":
        return None
    try:
        from transformers import BitsAndBytesConfig
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"transformers/BitsAndBytesConfig import 실패: {e}") from e

    # vision·projector·lm_head 전체 서브트리를 양자화에서 제외 → BF16 유지.
    skip = ["vision_backbone", "projector", "lm_head"]
    if policy.backbone_precision == "int8":
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_skip_modules=skip,
            llm_int8_threshold=6.0,
        )
    if policy.backbone_precision == "nf4":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=skip,   # 4bit 경로에서도 skip 목록으로 동작
        )
    raise ValueError(f"bnb backbone_precision={policy.backbone_precision!r} 미지원")


# ============================================================================
#  Sandwich 학습 보조 — 동결/해제 + LoRA 타깃 수집
# ============================================================================
def collect_lora_targets(
    vla: nn.Module,
    llm_mode: str = "all",
    include_vision: bool = True,
) -> Tuple[List[str], List[str]]:
    """LoRA 를 붙일 Linear 모듈의 **정확한 FQN 리스트**를 (llm, vision) 으로 반환한다.

    - llm_mode="all": language_model.model 의 모든 Linear (q/k/v/o + gate/up/down).
                      OpenVLA 검증 레시피(=all-linear 의 LLM 한정판).
    - llm_mode="qv" : self_attn.q_proj / v_proj 만 (스펙의 'Q,V 투영' 최소 구성).
    - lm_head 는 절대 포함하지 않는다(= action head, full BF16 학습 대상이지 LoRA 아님).
    - projector 도 포함하지 않는다(LoRA 금지; modules_to_save 로 full 학습).

    FQN 전체 경로를 그대로 넘기면 PEFT 가 endswith 매칭으로 정확히 그 모듈만 잡는다
    (projector.fc1 / vision fc1 처럼 짧은 이름이 충돌하는 문제를 피한다).
    """
    llm_targets: List[str] = []
    vision_targets: List[str] = []
    for name, module in vla.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if name == LM_HEAD_NAME:
            continue  # action head — 제외
        if name.startswith(LLM_BODY_PREFIX):
            if llm_mode == "qv" and not (name.endswith(".q_proj") or name.endswith(".v_proj")):
                continue
            llm_targets.append(name)
        elif include_vision and name.startswith(VISION_PREFIX):
            vision_targets.append(name)
    return llm_targets, vision_targets


def build_lora_config(
    vla: nn.Module,
    llm_rank: int = 32,
    vision_rank: int = 16,
    lora_dropout: float = 0.0,
    llm_mode: str = "all",
    lora_alpha: Optional[int] = None,
    use_rslora: bool = False,
    train_vision_lora: bool = True,
    train_projector: bool = True,
    train_action_head: bool = True,
):
    """스펙 §3 의 샌드위치 구성을 그대로 담은 PEFT LoraConfig 를 만든다.

    - LLM Q,V+선형: LoRA rank=32 (BF16 어댑터 — 언더플로 방지).
    - Vision: LoRA rank=16 (옵션). rank_pattern/alpha_pattern 으로 LLM 과 별도 rank.
    - Projector / Action head(lm_head): LoRA 가 아니라 **modules_to_save** 로 전체
      가중치를 BF16 로 학습(동결 해제). PEFT 가 저장/병합 시 함께 보존한다.

    capacity 레버(design_v9_lr_finetune §"Capacity Tune Plan"):
    - `lora_alpha`(None=현행 min(rank,16) 보존): LLM LoRA 의 유효 스케일 γ=α/r 을
      키워 under-fitting(동결 FP8 백본 위 LoRA-only 적응의 절반 감쇠)을 푼다.
    - `use_rslora`: 스케일을 α/√r 로 바꿔(rsLoRA) 같은 α 로도 γ 를 더 키운다.
      둘 다 **파라미터·VRAM 증가 0**(상수만 바뀜) — rank↑ 만 VRAM 비용.
      (vision 은 alpha_pattern=vision_rank 로 둬 LLM 경로만 조정; rsLoRA 시 vision
      유효 스케일은 √vision_rank 가 되지만 병목이 아니라 그대로 둔다.)
    """
    try:
        from peft import LoraConfig
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"peft import 실패: {e}") from e

    llm_targets, vision_targets = collect_lora_targets(
        vla, llm_mode=llm_mode, include_vision=train_vision_lora
    )
    target_modules = list(llm_targets)
    rank_pattern: Dict[str, int] = {}
    alpha_pattern: Dict[str, int] = {}
    if train_vision_lora and vision_targets:
        target_modules += vision_targets
        # vision FQN 들만 rank=16 으로. (정확 FQN 을 key 로 — endswith 매칭과 일관)
        for t in vision_targets:
            rank_pattern[t] = vision_rank
            alpha_pattern[t] = vision_rank

    modules_to_save: List[str] = []
    if train_projector:
        modules_to_save.append("projector")
    if train_action_head:
        modules_to_save.append("lm_head")

    # 기본은 현행값 보존(min(rank,16)) — 명시 시 capacity 레버로 올린다.
    alpha = lora_alpha if lora_alpha is not None else min(llm_rank, 16)

    return LoraConfig(
        r=llm_rank,
        lora_alpha=alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        rank_pattern=rank_pattern or None,
        alpha_pattern=alpha_pattern or None,
        modules_to_save=modules_to_save or None,
        use_rslora=use_rslora,
        init_lora_weights="gaussian",
    )


# ============================================================================
#  검증/리포트 — 정책이 실제로 적용됐는지 + VRAM 추정
# ============================================================================
def _module_group(fqn: str) -> str:
    # fqn 은 모듈 FQN("language_model.lm_head") 또는 파라미터 FQN
    # ("language_model.lm_head.weight") 둘 다 들어온다. 경로 컴포넌트로 판정해
    # 파라미터의 .weight/.bias 접미사에 영향받지 않게 한다.
    # PEFT 로 감싸면 모든 이름 앞에 "base_model.model." 이 붙으므로 먼저 떼어낸다
    # (get_peft_model 이후에 불러도 그룹이 무너지지 않게).
    for prefix in ("base_model.model.", "base_model.", "module."):
        if fqn.startswith(prefix):
            fqn = fqn[len(prefix):]
    parts = fqn.split(".")
    if "lm_head" in parts:
        return "action_head(lm_head)"
    if parts[0] == "vision_backbone":
        return "vision_backbone"
    if parts[0] == "projector":
        return "projector"
    if parts[0] == "language_model":
        return "language_model.model"
    return "other"


def _real_bytes(t) -> int:
    """텐서가 실제로 차지하는 저장 바이트. torchao/양자화 subclass 는 element_size() 가
    논리 dtype(예: bf16=2B)을 돌려줘 fp8/int 저장을 과대평가하므로, tensor subclass 의
    표준 프로토콜(__tensor_flatten__)로 내부 텐서(packed data + scale)를 재귀 합산한다."""
    flatten = getattr(t, "__tensor_flatten__", None)
    if callable(flatten):
        try:
            names, _ = t.__tensor_flatten__()
            return sum(_real_bytes(getattr(t, n)) for n in names)
        except Exception:  # noqa: BLE001
            pass
    try:
        return t.untyped_storage().nbytes()
    except Exception:  # noqa: BLE001
        try:
            return t.numel() * t.element_size()
        except Exception:  # noqa: BLE001
            return t.numel()


def _dtype_label(t) -> str:
    """plain 텐서는 dtype 명("bfloat16"), 양자화 subclass 는 클래스명("Float8Tensor" 등)을
    돌려줘 어떤 그룹이 실제로 양자화됐는지 한눈에 보이게 한다."""
    cls = type(t)
    if cls is torch.Tensor or cls.__name__ in ("Tensor", "Parameter"):
        return str(t.dtype).replace("torch.", "")
    return cls.__name__


def summarize_precision(vla: nn.Module) -> str:
    """모듈 그룹별 (param 수, 추정 바이트, 대표 dtype/양자화)를 표로 만든다.

    torchao 양자화 가중치는 tensor subclass 라 element_size() 가 실제 저장 바이트를
    반영하므로 그대로 합산하면 양자화 후 메모리에 근접한다. 정책이 비대칭으로 먹었는지
    (LLM 만 FP8/INT, projector·lm_head 는 BF16) 한눈에 확인하는 용도.
    """
    n_params: Dict[str, int] = {}
    n_bytes: Dict[str, int] = {}
    dtypes: Dict[str, Set[str]] = {}

    def _accum(fqn: str, t) -> None:
        g = _module_group(fqn)
        n_params[g] = n_params.get(g, 0) + t.numel()
        n_bytes[g] = n_bytes.get(g, 0) + _real_bytes(t)
        dtypes.setdefault(g, set()).add(_dtype_label(t))

    for fqn, p in vla.named_parameters():
        # subclass(torchao fp8/int)는 .data 가 양자화 텐서다 — 실제 저장량을 .data 로 잰다.
        _accum(fqn, getattr(p, "data", p))
    # quantize_ 가 weight 를 버퍼/subclass 로 옮기면 named_parameters 에서 빠질 수 있어
    # named_buffers 도 합산한다.
    for fqn, b in vla.named_buffers():
        _accum(fqn, b)

    order = ["vision_backbone", "projector", "language_model.model",
             "action_head(lm_head)", "other"]
    lines = ["", "=== OpenVLA 비대칭 정밀도 리포트 ==========================",
             f"{'group':24s} {'params':>13s} {'mem(GB)':>9s}  dtypes"]
    total_bytes = 0
    for g in order:
        if g not in n_params:
            continue
        gb = n_bytes[g] / 1e9
        total_bytes += n_bytes[g]
        dt = ",".join(sorted(dtypes[g]))
        lines.append(f"{g:24s} {n_params[g]:>13,d} {gb:>9.3f}  {dt}")
    lines.append("-" * 58)
    lines.append(f"{'TOTAL (weights)':24s} {'':>13s} {total_bytes / 1e9:>9.3f}  GB")
    lines.append("=" * 58)
    return "\n".join(lines)
