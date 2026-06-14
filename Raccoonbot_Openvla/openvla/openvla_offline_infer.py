import argparse
import base64
import json
from pathlib import Path

from openvla_server import OpenVLAServingModel, PredictRequest


def _b64(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("ascii")


def _fmt(vec):
    return "[" + ", ".join(f"{float(x):+.4f}" for x in vec) + "]"


def _l2(a, b):
    n = min(len(a), len(b))
    return sum((float(a[i]) - float(b[i])) ** 2 for i in range(n)) ** 0.5


def iter_samples(args):
    if args.episode_dir:
        ep = Path(args.episode_dir)
        meta = json.loads((ep / "meta.json").read_text(encoding="utf-8"))
        instruction = args.instruction or meta.get("instruction", "")
        steps = meta.get("steps", [])
        if not steps:
            raise ValueError(f"meta.json 에 steps 가 없음: {ep}")
        for step in steps[: args.num_steps]:
            img = ep / step["image_file"]
            yield img, instruction, step.get("action")
    else:
        if not args.image:
            raise ValueError("--image 또는 --episode-dir 중 하나는 필수")
        yield Path(args.image), args.instruction, None


def make_request(model, img, instruction, do_sample, unnorm_key):
    return PredictRequest(
        instruction=instruction,
        image_b64=_b64(img),
        unnorm_key=unnorm_key,
        do_sample=do_sample,
    )


def run_predict(model, req):
    return model.predict(req)


def run_eval(model, req, gt_action):
    out = model.predict(req)
    pred = out["action"]
    return {
        "pred": pred,
        "gt": gt_action,
        "l2": _l2(pred, gt_action) if gt_action is not None else None,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["predict", "eval"], default="predict")
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--default-unnorm-key", type=str, default="raccoon_pick_place")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--image", type=str, default=None)
    p.add_argument("--episode-dir", type=str, default=None)
    p.add_argument("--instruction", type=str, default=None)
    p.add_argument("--num-steps", type=int, default=3)
    p.add_argument("--do-sample", action="store_true")
    args = p.parse_args()

    if args.mode == "eval" and not args.episode_dir:
        p.error("--mode eval requires --episode-dir with ground truth action")

    model = OpenVLAServingModel(
        model_path=args.model_path,
        device=args.device,
        default_unnorm_key=args.default_unnorm_key,
    )

    n = 0
    for img, instruction, gt in iter_samples(args):
        req = make_request(model, img, instruction, args.do_sample,
                           args.default_unnorm_key)

        if args.mode == "predict":
            out = run_predict(model, req)
            print(f"\n=== sample {n} : {img.name} ===")
            print(json.dumps(out, ensure_ascii=False))
        else:  # eval
            res = run_eval(model, req, gt)
            print(f"\n=== sample {n} : {img.name} ===")
            print(f"  instruction : {instruction}")
            print(f"  pred  action: {_fmt(res['pred'])}")
            print(f"  gt    action: {_fmt(res['gt'])}")
            print(f"  L2(pred,gt) : {res['l2']:.4f}")
        n += 1

    if args.mode == "predict":
        print(f"\n[OK] {n} samples completed (identical to server output)")
    else:
        print(f"\n[OK] {n} samples completed (accuracy evaluation)")


if __name__ == "__main__":
    main()
