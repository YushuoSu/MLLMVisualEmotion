# =============================================================================
# FaceEmotion — Batch runner for all models
# =============================================================================
import os
import sys
import subprocess
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

VLLM_SCRIPTS = [
    "qwen2_test.py",
    "qwen2p5_test.py",
    "qwen3_test.py",
    "internvl2_test.py",
    "internvl3_test.py",
    "llava1p5_test.py",
    "llava1p6_test.py",
]

API_SCRIPTS = [
    "gpt4o_test.py",
    "gpt5mini_test.py",
    "gemini_flash_test.py",
]

ALL_SCRIPTS = VLLM_SCRIPTS + API_SCRIPTS

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(SCRIPT_DIR, "script_execution_log.txt")


def log_message(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_script(script_name):
    script_path = os.path.join(SCRIPT_DIR, script_name)
    if not os.path.exists(script_path):
        log_message(f"ERROR: {script_path} not found!")
        return False

    log_message(f"Running: {script_name}")
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=SCRIPT_DIR,
            capture_output=True,
            text=True,
        )
        if result.stdout:
            log_message(f"[{script_name}] stdout:\n{result.stdout[-2000:]}")
        if result.stderr:
            log_message(f"[{script_name}] stderr:\n{result.stderr[-2000:]}")
        ok = result.returncode == 0
        log_message(f"{'OK' if ok else 'FAILED (rc=' + str(result.returncode) + ')'}")
        return ok
    except Exception as e:
        log_message(f"EXCEPTION: {e}")
        return False


def main():
    log_message("=" * 50)
    log_message("FaceEmotion — Batch evaluation start")
    log_message("=" * 50)

    success, fail = 0, 0
    failed_scripts = []

    for i, script in enumerate(ALL_SCRIPTS, 1):
        log_message(f"\n[{i}/{len(ALL_SCRIPTS)}] {script}")
        if run_script(script):
            success += 1
        else:
            fail += 1
            failed_scripts.append(script)

    log_message("\n" + "=" * 50)
    log_message(f"FaceEmotion batch complete: {success} OK, {fail} FAILED")
    if failed_scripts:
        log_message(f"Failed: {', '.join(failed_scripts)}")
    log_message("=" * 50)


if __name__ == "__main__":
    main()