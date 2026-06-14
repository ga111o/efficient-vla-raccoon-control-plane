import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_common import resolve_rb_root, setup_env, raw_demos_dir, stack_demos_dir  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episode-index", type=int, default=0)
    p.add_argument("--num-frames", type=int, default=5)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--dataset-root", type=str, default=None)
    p.add_argument("--stack", action="store_true")
    args = p.parse_args()

    rb = resolve_rb_root()
    setup_env(rb)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    if args.dataset_root:
        root = Path(args.dataset_root).resolve()
    else:
        root = stack_demos_dir(rb) if args.stack else raw_demos_dir(rb)
    episodes = sorted(d for d in root.iterdir() if d.is_dir() and d.name.startswith("episode_"))
    if not episodes:
        raise SystemExit(f"episode_* 디렉토리를 찾을 수 없음: {root}")
    if not (0 <= args.episode_index < len(episodes)):
        raise SystemExit(f"--episode-index {args.episode_index} 범위 밖 (0..{len(episodes)-1})")

    ep = episodes[args.episode_index]
    frames = sorted(ep.glob("frame_*.png"))
    if not frames:
        raise SystemExit(f"frame_*.png 를 찾을 수 없음: {ep}")

    with open(ep / "meta.json", "r") as f:
        meta = json.load(f)
    instruction = meta.get("instruction", meta.get("language_instruction", "N/A"))
    target_color = meta.get("target_color", "N/A")
    success = meta.get("success", "N/A")

    n_show = min(args.num_frames, len(frames))
    fig, axes = plt.subplots(1, n_show, figsize=(3 * n_show, 3))
    if n_show == 1:
        axes = [axes]
    for ax, frame_path in zip(axes, frames[:n_show]):
        ax.imshow(Image.open(frame_path))
        ax.set_title(frame_path.name, fontsize=10)
        ax.axis("off")

    fig.suptitle(
        f"Instruction: {instruction}\nTarget Color: {target_color}, Success: {success}",
        fontsize=12,
    )
    fig.tight_layout()

    out = Path(args.out).resolve() if args.out else (Path(__file__).resolve().parent / "outputs" / "episode_vis.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[saved] {out}  (episode={ep.name}, frames={n_show})")


if __name__ == "__main__":
    main()
