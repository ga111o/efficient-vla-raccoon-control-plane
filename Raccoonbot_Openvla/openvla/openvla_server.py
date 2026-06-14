import argparse
import base64
import io
import json
import os
import traceback
from pathlib import Path

# Reduce extra TensorFlow/backend noise.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
import uvicorn


from typing import Optional


class PredictRequest(BaseModel):
    instruction: str
    image_b64: str
    unnorm_key: Optional[str] = None
    do_sample: bool = False

class OpenVLAServingModel:
    def __init__(self, model_path: str, device: str = "cuda", default_unnorm_key: str = "bridge_orig",
                 load_in_4bit: bool = False, precision: str = "bf16", vision_precision: str = "int8"):

        self.model_path = model_path
        self.device = device
        self.default_unnorm_key = default_unnorm_key
        self.load_in_4bit = load_in_4bit
        self.precision = precision
        self.vision_precision = vision_precision

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

        load_kwargs: dict[str, object] = dict(
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            load_kwargs["quantization_config"] = quantization_config
            load_kwargs["device_map"] = {"": device}
            print("[INFO] Loading model with 4-bit (NF4) quantization")
            self.vla = AutoModelForVision2Seq.from_pretrained(model_path, **load_kwargs)
        elif precision != "bf16":
            from prismatic.quant.policy import (
                QuantPolicy, apply_torchao_quantization, summarize_precision,
            )
            print(f"[INFO] Asymmetric mixed precision: backbone={precision}, vision={vision_precision}, "
                  f"projector/lm_head=bf16 (locked)")
            self.vla = AutoModelForVision2Seq.from_pretrained(model_path, **load_kwargs)
            policy = QuantPolicy(
                backend="torchao", backbone_precision=precision, vision_precision=vision_precision,
            ).validate()
            report = apply_torchao_quantization(self.vla, policy, device=device)
            self.vla = self.vla.to(device)
            print("[INFO] quantization report:", report)
            print(summarize_precision(self.vla))
        else:
            self.vla = AutoModelForVision2Seq.from_pretrained(model_path, **load_kwargs).to(device)

        stats_path = Path(model_path) / "dataset_statistics.json"
        if stats_path.exists():
            with open(stats_path, "r", encoding="utf-8") as f:
                self.vla.norm_stats = json.load(f)
            print(f"[INFO] Loaded dataset statistics from: {stats_path}")
            print(f"[INFO] Available norm_stats keys: {list(self.vla.norm_stats.keys())}")
        else:
            print(f"[WARN] dataset_statistics.json not found at: {stats_path}")

    @torch.inference_mode()
    def predict(self, req: PredictRequest):
        image_bytes = base64.b64decode(req.image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        prompt = f"In: What action should the robot take to {req.instruction}?\nOut:"
        inputs = self.processor(prompt, image).to(self.device, dtype=torch.bfloat16)

        unnorm_key = req.unnorm_key or self.default_unnorm_key

        action = self.vla.predict_action(
            **inputs,
            unnorm_key=unnorm_key,
            do_sample=req.do_sample,
        )

        if hasattr(action, "tolist"):
            action = action.tolist()

        action = [float(x) for x in action]
        if len(action) < 4:
            raise ValueError(f"Predicted action is too short: len={len(action)}, action={action}")

        print(f"[PREDICT] instruction={req.instruction}")
        print(f"[PREDICT] unnorm_key={unnorm_key}")
        print(f"[PREDICT] action={action}", flush=True)

        return {
            "action": action,
            "unnorm_key": unnorm_key,
            "prompt": prompt,
        }


def build_app(serving_model: OpenVLAServingModel):
    app = FastAPI()

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.post("/predict")
    def predict(req: PredictRequest):
        try:
            return serving_model.predict(req)
        except Exception as exc:
            traceback.print_exc()
            raise HTTPException(status_code=400, detail=str(exc))

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--default-unnorm-key", type=str, default="bridge_orig")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--precision", type=str, default="bf16")
    parser.add_argument("--vision-precision", dest="vision_precision", type=str, default="int8")
    args = parser.parse_args()

    serving_model = OpenVLAServingModel(
        model_path=args.model_path,
        device=args.device,
        default_unnorm_key=args.default_unnorm_key,
        load_in_4bit=args.load_in_4bit,
        precision=args.precision,
        vision_precision=args.vision_precision,
    )
    app = build_app(serving_model)

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", access_log=False)


if __name__ == "__main__":
    main()