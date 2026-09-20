"""Run upstairs and downstairs G1 mimic training sequentially in the background."""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TASKS = ("g1_mimic_stairs_up", "g1_mimic_stairs_down")


def timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save_status(queue_dir, status):
    temporary = queue_dir / "status.tmp"
    temporary.write_text(json.dumps(status, indent=2) + "\n")
    temporary.replace(queue_dir / "status.json")


def run_queue(queue_dir):
    status = json.loads((queue_dir / "status.json").read_text())
    status.update(pid=os.getpid(), state="waiting", started_at=timestamp())
    save_status(queue_dir, status)
    child = None
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True
        if child is not None and child.poll() is None:
            child.terminate()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    env = os.environ.copy()
    for name in ("PYTHONPATH", "LEGGEDLAB_MOTION_DIR", "LEGGEDLAB_MOTION_SELECTION"):
        env.pop(name, None)
    env["OMNI_KIT_ACCEPT_EULA"] = "YES"
    runtime = Path(status["python"]).parent.parent / "lib/libstdc++.so.6"
    if runtime.is_file():
        env["LD_PRELOAD"] = str(runtime) + (":" + env["LD_PRELOAD"] if env.get("LD_PRELOAD") else "")
    env["MPLCONFIGDIR"] = str(queue_dir / "matplotlib_cache")
    lock_path = REPOSITORY_ROOT / "logs/mimic_queue" / (status["device"].replace(":", "_") + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        status["state"] = "running"
        save_status(queue_dir, status)
        for job in status["jobs"]:
            if stopping:
                job["state"] = "cancelled"
                continue
            job.update(state="running", started_at=timestamp())
            print(f"[{timestamp()}] START {job['task']}", flush=True)
            with Path(job["console_log"]).open("ab", buffering=0) as console:
                child = subprocess.Popen(job["command"], cwd=REPOSITORY_ROOT, env=env,
                                         stdin=subprocess.DEVNULL, stdout=console, stderr=subprocess.STDOUT)
                job["pid"] = child.pid
                save_status(queue_dir, status)
                returncode = child.wait()
                child = None
            runs = sorted((REPOSITORY_ROOT / "logs" / job["task"]).glob("*" + status["run_name"]))
            run_dir = runs[-1] if runs else None
            checkpoint = run_dir / f"model_{status['max_iterations'] - 1}.pt" if run_dir else None
            job.update(exit_code=returncode, finished_at=timestamp(),
                       run_dir=str(run_dir) if run_dir else None,
                       final_checkpoint=str(checkpoint) if checkpoint and checkpoint.is_file() else None)
            if stopping:
                job["state"] = "cancelled"
            else:
                job["state"] = "completed" if returncode == 0 and job["final_checkpoint"] else "failed"
                if returncode == 0 and not job["final_checkpoint"]:
                    job["error"] = "Training exited without the expected final checkpoint. Inspect console_log."
            save_status(queue_dir, status)
            print(f"[{timestamp()}] END {job['task']}: {job['state']} (exit {returncode})", flush=True)
        status["state"] = ("cancelled" if stopping else
                           "completed" if all(j["state"] == "completed" for j in status["jobs"]) else "failed")
        status["finished_at"] = timestamp()
        save_status(queue_dir, status)
    return 0 if status["state"] == "completed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=10000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--detach", action="store_true", help="Keep the queue running after the calling terminal exits.")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return run_queue(args.worker.resolve())
    if args.num_envs <= 0 or args.max_iterations <= 0:
        parser.error("--num_envs and --max_iterations must be positive")
    queue_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{os.getpid()}"
    queue_dir = REPOSITORY_ROOT / "logs/mimic_queue" / queue_id
    queue_dir.mkdir(parents=True)
    run_name = "queue_" + queue_id
    status = {"state": "queued", "created_at": timestamp(), "python": sys.executable,
              "num_envs": args.num_envs, "max_iterations": args.max_iterations,
              "device": args.device, "run_name": run_name, "jobs": []}
    for task in TASKS:
        status["jobs"].append({
            "task": task, "state": "pending", "console_log": str(queue_dir / f"{task}.log"),
            "command": [sys.executable, "-u", str(REPOSITORY_ROOT / "legged_lab/scripts/train.py"),
                        f"--task={task}", "--headless", "--logger=tensorboard",
                        f"--device={args.device}", f"--num_envs={args.num_envs}",
                        f"--max_iterations={args.max_iterations}", f"--run_name={run_name}"],
        })
    save_status(queue_dir, status)
    if args.detach:
        with (queue_dir / "queue.log").open("ab", buffering=0) as console:
            worker = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()),
                                       "--worker", str(queue_dir)], cwd=REPOSITORY_ROOT,
                                      stdin=subprocess.DEVNULL, stdout=console, stderr=subprocess.STDOUT,
                                      start_new_session=True)
        print(json.dumps({"pid": worker.pid, "queue_dir": str(queue_dir),
                          "status_file": str(queue_dir / "status.json")}, indent=2))
        return 0
    print(f"Queue status: {queue_dir / 'status.json'}", flush=True)
    return run_queue(queue_dir)


if __name__ == "__main__":
    sys.exit(main())
