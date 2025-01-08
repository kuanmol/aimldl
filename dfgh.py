import psutil

def close_programs(excluded_processes=None):
    """Close all non-essential programs."""
    if excluded_processes is None:
        excluded_processes = ["python.exe", "explorer.exe", "System", "svchost.exe", "winlogon.exe"]

    for process in psutil.process_iter(['pid', 'name']):
        try:
            process_name = process.info['name']
            process_pid = process.info['pid']

            # Skip excluded processes
            if process_name in excluded_processes:
                continue

            # Terminate the process safely
            psutil.Process(process_pid).terminate()
            print(f"Closed: {process_name} (PID: {process_pid})")
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

if __name__ == "__main__":
    print("Closing all non-essential programs...")
    close_programs()
    print("Done!")
