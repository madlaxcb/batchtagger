import os
import sys
import time
import subprocess
import signal

# Assert executable exists before running test
exe_check = os.path.join("dist", "BatchTagger", "BatchTagger.exe")
if not os.path.isfile(exe_check):
    print(f"CRITICAL ERROR: {exe_check} not found; build step likely failed or output path is wrong.")
    # Verify dist folder content to help debugging
    if os.path.exists("dist"):
        print(f"Contents of dist: {os.listdir('dist')}")
        bt_dir = os.path.join("dist", "BatchTagger")
        if os.path.exists(bt_dir):
            print(f"Contents of dist/BatchTagger: {os.listdir(bt_dir)}")
    sys.exit(1)

def run_smoke_test():
    """
    Smoke test to verify the built executable launches and runs for a few seconds.
    """
    exe_path = os.path.join("dist", "BatchTagger", "BatchTagger.exe")
    
    if not os.path.exists(exe_path):
        print(f"Error: Executable not found at {exe_path}")
        sys.exit(1)
        
    print(f"Launching {exe_path}...")
    
    # Launch the process
    try:
        process = subprocess.Popen([exe_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception as e:
        print(f"Failed to launch process: {e}")
        sys.exit(1)
        
    # Wait for 5 seconds
    time.sleep(5)
    
    # Check if process is still running
    if process.poll() is None:
        print("Success: Application is running.")
        # Terminate the process
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
        sys.exit(0)
    else:
        # Process exited prematurely
        stdout, stderr = process.communicate()
        print(f"Error: Application exited prematurely with code {process.returncode}")
        print(f"Stdout: {stdout.decode('utf-8', errors='ignore')}")
        print(f"Stderr: {stderr.decode('utf-8', errors='ignore')}")
        sys.exit(1)

if __name__ == "__main__":
    run_smoke_test()
