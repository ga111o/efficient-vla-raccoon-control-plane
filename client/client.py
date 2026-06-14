import io
import base64
import time
import requests
import numpy as np
import cv2
from PIL import Image
from roboid import Raccoon

SERVER_URL = "http://ga111o-desktop:8005"
INSTRUCTION = "grasp the yellow cylinder"
UNNORM_KEY = "raccoon_pick_place"
MAX_STEPS = 300
GRIPPER_OPEN_THRESHOLD = 0.5
XYZ_SCALE = 1.0


def _img_to_b64(image_rgb: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(image_rgb).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def capture_frame(cap: cv2.VideoCapture) -> np.ndarray:
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("ret not true")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def predict_action(image_rgb: np.ndarray) -> list[float]:
    payload = {
        "instruction": INSTRUCTION,
        "image_b64": _img_to_b64(image_rgb),
        "unnorm_key": UNNORM_KEY,
        "do_sample": False,
    }
    resp = requests.post(f"{SERVER_URL}/predict", json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    action = data["action"]
    if len(action) < 4:
        raise ValueError(f"action value err: {action}")
    return action


def execute_delta_action7(bot: Raccoon, action: list[float]) -> None:
    """7D delta [dx, dy, dz, droll, dpitch, dyaw, gripper] → 4DOF 실행."""
    dx, dy, dz = [v * XYZ_SCALE for v in action[:3]]
    gripper = action[6]

    if bot.can_move_to(*[a + b for a, b in zip(bot.xyz(), [dx, dy, dz])]):
        bot.move_by(dx, dy, dz, speed=40)
    else:
        print(f"can not move: dx={dx:.3f} dy={dy:.3f} dz={dz:.3f}")

    if gripper > GRIPPER_OPEN_THRESHOLD:
        bot.open_gripper(wait=False)
    else:
        bot.close_gripper(wait=False)


def run_episode(bot: Raccoon) -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("camera not opened")

    try:
        for step in range(MAX_STEPS):
            frame = capture_frame(cap)

            action = predict_action(frame)
            print(
                f"  step {step+1:03d} | "
                f"xyz=({action[0]:.3f},{action[1]:.3f},{action[2]:.3f}) "
                f"rpy=({action[3]:.3f},{action[4]:.3f},{action[5]:.3f}) "
                f"grip={action[6]:.3f}"
            )

            execute_delta_action7(bot, action)
            time.sleep(0.1)
    finally:
        cap.release()


def main() -> None:
    bot = Raccoon()
    try:
        run_episode(bot)
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        bot.stop()
        bot.dispose()


if __name__ == "__main__":
    main()

