import subprocess
from pathlib import Path

# CONFIGURATION 
VOLUME_NAME = "gemma4-sft-volume"
LOCAL_DATA_FILE = "gemma_training_data.jsonl"
REMOTE_PATH = "/" # Root of the volume

def run_command(cmd):
    print(f"Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    else:
        print(f"Success: {result.stdout}")

def upload_data():
    if not Path(LOCAL_DATA_FILE).exists():
        print(f"File {LOCAL_DATA_FILE} not found!")
        return

    # Check if volume exists (this will error if it doesn't, but that's fine)
    # run_command(["modal", "volume", "ls", VOLUME_NAME])

    # Upload command
    print(f"Uploading {LOCAL_DATA_FILE} to Modal Volume '{VOLUME_NAME}'...")
    run_command(["python3", "-m", "modal", "volume", "put", VOLUME_NAME, LOCAL_DATA_FILE, REMOTE_PATH])

if __name__ == "__main__":
    upload_data()
