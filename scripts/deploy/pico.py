"""Reuse the PICO processes from GR00T launch_data_collection, without robot control."""
import argparse
from pathlib import Path
import shlex
import subprocess
import time


def commands(a):
    root = a.groot_root.resolve()
    stream = [str(root / ".venv_teleop/bin/python"),
              "gear_sonic/scripts/pico_manager_thread_server.py",
              "--input-source", "xrt", "--manager", "--num_frames_to_send", "1",
              "--port", str(a.port),
              "--zmq_feedback_host", a.feedback_host,
              "--zmq_feedback_port", str(a.feedback_port)]
    if a.visualize:
        print("Diagnostic visualization can reduce pose streaming rate; omit for live control.", flush=True)
        stream += ["--vis_vr3pt", "--vis_smpl"]
    result = [("PICO tracking / SMPL / 3PT", stream)]
    common = ["--listen-host", a.listen_host, "--headset", a.headset]
    if a.video == "usb":
        video = [str(root / ".venv_data_collection/bin/python"),
                 "-m", "gear_sonic.scripts.run_pico_usb_stereo",
                 "--camera-host", a.camera_host, "--camera-port", str(a.camera_port),
                 *common, "--view", "right"]
        result.append(("PICO USB right-eye video", video))
    elif a.video == "test":
        video = [str(root / ".venv_data_collection/bin/python"),
                 "-m", "gear_sonic.scripts.run_pico_video_bridge",
                 "--source", "test", *common, "--layout", "mono"]
        result.append(("PICO TEST IMAGE (not simulation camera)", video))
    return root, result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--groot-root", type=Path, required=True)
    p.add_argument("--listen-host", default="192.168.11.224")
    p.add_argument("--headset", default="192.168.11.228")
    p.add_argument("--video", choices=["none", "test", "usb"], default="none")
    p.add_argument("--camera-host", default="192.168.123.164")
    p.add_argument("--camera-port", type=int, default=15556)
    p.add_argument("--port", type=int, default=5556)
    p.add_argument("--feedback-host", default="127.0.0.1")
    p.add_argument("--feedback-port", type=int, default=5557)
    p.add_argument("--visualize", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    root, jobs = commands(a)
    for name, cmd in jobs:
        print(name + ":\n  cd " + shlex.quote(str(root)) + " && " + shlex.join(cmd), flush=True)
    print("Tracking uses XRoboToolkit PC Service. On PICO: PC Service IP = "
          + a.listen_host + "; Head/Controller enabled, Data/Control Send, Full body.", flush=True)
    print("listen-host/headset configure VIDEO only; they do not configure tracking.", flush=True)
    if a.dry_run:
        return
    for _, cmd in jobs:
        if not Path(cmd[0]).is_file():
            p.error("Missing original GR00T environment: " + cmd[0])
    children = []
    try:
        for name, cmd in jobs:
            children.append((name, subprocess.Popen(cmd, cwd=root)))
        while True:
            for name, child in children:
                status = child.poll()
                if status is not None:
                    raise SystemExit(f"{name} exited ({status}); closing launcher children.")
            time.sleep(.2)
    except KeyboardInterrupt:
        pass
    finally:
        for _, child in children:
            if child.poll() is None:
                child.terminate()
        for _, child in children:
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


if __name__ == "__main__":
    main()
