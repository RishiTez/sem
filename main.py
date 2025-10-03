import os
import subprocess
import time

def run_cmd(cmd):
    print(f"\nRunning: {cmd}\n")
    ret = subprocess.call(cmd, shell=True)
    if ret != 0:
        raise RuntimeError(f"Command failed: {cmd}")

def main():
    os.makedirs("checkpoints", exist_ok=True)

    run_cmd( f"python train_teacher.py ")
    run_cmd( f"python train_student.py --no-teacher")
    run_cmd( f"python train_student.py ")

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    time_taken = end_time - start_time
    hours_elapsed = int(time_taken // 3600)
    minutes_elapsed = int((time_taken % 3600) // 60)
    seconds_elapsed = int(time_taken % 60)
    print(f"Total time taken: {hours_elapsed}h {minutes_elapsed}m {seconds_elapsed}s")

