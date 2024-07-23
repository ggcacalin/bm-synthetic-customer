import psutil
import time
from datetime import datetime

def get_cpu_memory_usage():
    process = psutil.Process()
    cpu_usage = process.cpu_percent(interval=1)
    memory_info = process.memory_info()
    memory_usage = memory_info.rss / (1024 ** 2)  # Convert bytes to MB
    return cpu_usage, memory_usage

def monitor_usage():
    runtime = 0
    with open("usage_logs.csv", "a") as f:
          f.write(datetime.now(), "\n")
    while True:
        cpu, memory = get_cpu_memory_usage()
        with open("usage_logs.csv", "a") as f:
          f.write(str(runtime) + "," + str(cpu) + "," + str(memory) + "\n")
        runtime += 3
        time.sleep(3)  # Wait for 3 seconds before the next check