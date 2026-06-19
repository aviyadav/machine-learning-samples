import os
import subprocess
import time
import tracemalloc


def _get_process_rss_mb():
    """Return process RSS memory in MB when psutil is available."""
    try:
        import psutil  # Optional dependency

        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except Exception:
        return None


def _detect_gpu():
    """Detect NVIDIA GPU availability using nvidia-smi when present."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        output = result.stdout.strip()
        if result.returncode == 0 and output:
            first_line = output.splitlines()[0]
            return True, first_line
    except Exception:
        pass
    return False, None


def profile_execution(func):
    def wrapper(*args, **kwargs):
        usable_processors = os.process_cpu_count() or os.cpu_count()
        total_processors = os.cpu_count()
        gpu_available, gpu_info = _detect_gpu()

        tracemalloc.start()
        rss_before = _get_process_rss_mb()
        start_time = time.perf_counter()
        start_cpu = time.process_time()

        result = func(*args, **kwargs)

        end_cpu = time.process_time()
        end_time = time.perf_counter()
        rss_after = _get_process_rss_mb()
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print("\n--- Execution Profile ---")
        print(f"Elapsed time: {end_time - start_time:.6f} seconds")
        print(f"CPU time: {end_cpu - start_cpu:.6f} seconds")
        print(f"Processors available to process: {usable_processors} / {total_processors}")

        if rss_before is not None and rss_after is not None:
            print(f"Process memory (RSS) before: {rss_before:.2f} MB")
            print(f"Process memory (RSS) after: {rss_after:.2f} MB")
            print(f"Process memory delta (RSS): {rss_after - rss_before:.2f} MB")
        else:
            print("Process memory (RSS): unavailable (install psutil for RSS metrics)")

        print(f"Peak Python memory (tracemalloc): {peak_bytes / (1024 * 1024):.2f} MB")
        print(f"GPU available: {'Yes' if gpu_available else 'No'}")
        if gpu_info:
            print(f"Detected GPU: {gpu_info}")

        # sklearn LinearRegression uses CPU backend by default.
        print("GPU used by this method: No")
        print("--- End Profile ---")

        return result

    return wrapper
