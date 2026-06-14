"""
merge_lora.py — post-hoc merge of a trained LoRA adapter into the base OpenVLA model.

When `finetune.py` is run with `--merge_during_training False`, training only writes
the (small) LoRA adapter to `adapter_dir`, plus the processor and dataset statistics to
`run_dir`. This skips the slow in-loop step that reloads the 7B backbone and rewrites
~14 GB at every checkpoint. Run this script ONCE after training to perform the merge and
write the full, servable model into `run_dir`.

Paths are read from `<run_root_dir>/last_run.json` (written by finetune.py) unless they
are given explicitly.

Run with (single process, no torchrun needed):
    python vla-scripts/merge_lora.py --run_root_dir <PATH/TO/openvla-runs>
    python vla-scripts/merge_lora.py --vla_path openvla/openvla-7b \
        --adapter_dir <ADAPTER_DIR> --run_dir <OUTPUT_DIR>
"""

import argparse
import json
import time
from pathlib import Path

import torch
# NOTE: keep timm imported before tensorflow-pulling deps, matching finetune.py's import-order guard.
import timm  # noqa: F401
from peft import PeftModel
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor


def resolve_paths(args: argparse.Namespace) -> tuple[str, Path, Path]:
    """Return (vla_path, adapter_dir, run_dir), preferring explicit args over last_run.json."""
    vla_path, adapter_dir, run_dir = args.vla_path, args.adapter_dir, args.run_dir

    if not (vla_path and adapter_dir and run_dir):
        if not args.run_root_dir:
            raise SystemExit(
                "[merge_lora] --run_root_dir 가 필요합니다 (last_run.json 위치). "
                "또는 --vla_path/--adapter_dir/--run_dir 를 모두 직접 지정하세요."
            )
        pointer = Path(args.run_root_dir) / "last_run.json"
        if not pointer.is_file():
            raise SystemExit(f"[merge_lora] pointer 파일 없음: {pointer} (먼저 050_finetune 을 실행하세요)")
        info = json.loads(pointer.read_text())
        if not info.get("use_lora", True):
            raise SystemExit("[merge_lora] last_run.json: use_lora=False — 병합할 어댑터가 없습니다.")
        vla_path = vla_path or info["vla_path"]
        adapter_dir = adapter_dir or info["adapter_dir"]
        run_dir = run_dir or info["run_dir"]

    return vla_path, Path(adapter_dir), Path(run_dir)


def main():
    p = argparse.ArgumentParser(description="Post-hoc merge of LoRA adapter into base OpenVLA")
    p.add_argument("--run_root_dir", type=str, default=None,
                   help="run_root_dir used in training; last_run.json is read from here")
    p.add_argument("--vla_path", type=str, default=None, help="base model path/repo (overrides pointer)")
    p.add_argument("--adapter_dir", type=str, default=None, help="LoRA adapter dir (overrides pointer)")
    p.add_argument("--run_dir", type=str, default=None, help="output dir for merged model (overrides pointer)")
    args = p.parse_args()

    vla_path, adapter_dir, run_dir = resolve_paths(args)
    if not adapter_dir.is_dir():
        raise SystemExit(f"[merge_lora] adapter dir 없음: {adapter_dir}")

    # Register OpenVLA model with HF Auto classes (needed for local/custom checkpoints).
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    print(f"[merge_lora] base    = {vla_path}", flush=True)
    print(f"[merge_lora] adapter = {adapter_dir}", flush=True)
    print(f"[merge_lora] out     = {run_dir}", flush=True)

    print("[merge_lora] loading base model ...", flush=True)
    t = time.time()
    base_vla = AutoModelForVision2Seq.from_pretrained(
        vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
    )
    print(f"[merge_lora] base loaded in {time.time() - t:.1f}s; loading adapter ...", flush=True)

    merged_vla = PeftModel.from_pretrained(base_vla, str(adapter_dir))
    print("[merge_lora] merging (merge_and_unload) ...", flush=True)
    t = time.time()
    merged_vla = merged_vla.merge_and_unload()
    print(f"[merge_lora] merged in {time.time() - t:.1f}s; writing merged model to {run_dir} ...", flush=True)

    run_dir.mkdir(parents=True, exist_ok=True)
    t = time.time()
    merged_vla.save_pretrained(str(run_dir))
    print(f"[merge_lora] model saved in {time.time() - t:.1f}s", flush=True)

    # Ensure the processor sits next to the merged weights (training already saves it to run_dir,
    #   but re-save as a safe fallback when run_dir was given explicitly).
    if not (run_dir / "preprocessor_config.json").exists():
        try:
            AutoProcessor.from_pretrained(vla_path, trust_remote_code=True).save_pretrained(str(run_dir))
        except Exception as e:  # noqa: BLE001
            print(f"[merge_lora] (warn) processor save skipped: {e}", flush=True)

    print(f"[merge_lora] ✓ done -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()
