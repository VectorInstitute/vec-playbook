#!/usr/bin/env python3
from datetime import datetime
import json
from pathlib import Path
import signal
import sys
import time

loop_count = 0

# If a checkpoint exists, update loop_count with the value stored there
ckpt_path = Path("checkpoint.json")
if ckpt_path.exists():
    try:
        with ckpt_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        loop_count = int(data.get("loop_count", loop_count))
    except:
        pass

# Signal handler should write a timestamp, output a checkpoint, then exit
def handle_sigusr1(signum, frame):
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Caught SIGUSR1, writing checkpoint...", flush=True)
    with open("checkpoint.json", "w", encoding="utf-8") as f:
        json.dump({"loop_count": loop_count}, f, indent=2)
    sys.exit(0)

# Register handler
signal.signal(signal.SIGUSR1, handle_sigusr1)


while loop_count < 20:
    now = datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    time.sleep(60)
    loop_count = loop_count + 1

print("Completed 20 iterations of the loop, job is now finished", flush=True)
