import os
import sys
import json
import subprocess
import platform
import psutil
from pathlib import Path
import time

# Only import Windows-specific libraries if on Windows
if platform.system() == 'Windows':
    try:
        import ctypes
        import win32gui
        import win32con
        import win32process
    except ImportError:
        print("The 'pywin32' library is required on Windows. Install it using 'pip install pywin32'.")
        sys.exit(1)

def get_config_path() -> Path:
    """
    Get the path to the configuration file for PurplePuppy Stamps.

    Returns:
        Path: The full path to the configuration file.
    """
    # Start with the base path of the executable or script
    if getattr(sys, 'frozen', False):
        # If the application is frozen, use the executable's directory
        base_path = Path(sys.executable).parent
    else:
        # If not frozen, use the script's directory
        base_path = Path(__file__).parent

    # Navigate up until we reach 'GDWeave'
    while base_path.name in ["mods", "PurplePuppy-Stamps"]:
        base_path = base_path.parent

    # Ensure the resolved base path is correct
    if base_path.name != "GDWeave":
        raise ValueError(f"Base path resolution error: {base_path} is not GDWeave.")

    # Navigate to the sibling 'configs' directory
    config_dir = (base_path / "configs").resolve()

    # Ensure the configs directory exists
    config_dir.mkdir(parents=True, exist_ok=True)

    # Define the specific configuration file name
    config_file = config_dir / "PurplePuppy.Stamps.json"

    return config_file

def load_config() -> dict:
    """
    Load the configuration from the JSON file, creating a default if necessary.
    """
    config_path = get_config_path()

    # Default configuration data
    default_config_data = {
        "open_menu": 16777252,
        "spawn_stamp": 61,
        "ctrl_z": 16777220,
        "toggle_playback": 45,
        "gif_ready": False,
        "pid": -1,
        "walky_talky_webfish": "nothing new!",
        "walky_talky_menu": "nothing new!"
    }

    # Create the file with default data if it does not exist
    if not config_path.exists():
        with open(config_path, 'w') as file:
            json.dump(default_config_data, file, indent=4)
        print(f"Created default config at {config_path}")
        return default_config_data

    # Load existing configuration
    with open(config_path, 'r') as file:
        return json.load(file)

def save_pid_to_config(pid: int):
    """
    Save the current process PID to the configuration file.
    """
    config_path = get_config_path()
    config = load_config()
    config['pid'] = pid

    with open(config_path, 'w') as file:
        json.dump(config, file, indent=4)

    print(f"Updated 'pid' to {pid} in config file.")

def is_process_running_by_pid(pid: int) -> bool:
    """
    Check if a process with the given PID is running.
    """
    try:
        proc = psutil.Process(pid)
        return proc.is_running()
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return False

def bring_to_front_windows(proc):
    """
    Bring the window of the given process to the front on Windows.
    """
    try:
        def enum_windows_callback(hwnd, pid):
            try:
                _, found_pid = win32process.GetWindowThreadProcessId(hwnd)
                if found_pid == pid:
                    if win32gui.IsWindowVisible(hwnd):
                        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                        win32gui.SetForegroundWindow(hwnd)
                        return False  # Stop enumeration
            except:
                pass
            return True  # Continue enumeration

        win32gui.EnumWindows(enum_windows_callback, proc.pid)
    except Exception as e:
        print(f"Failed to bring window to front: {e}")

def bring_to_front(pid):
    """
    Bring the application's window to the front based on the OS.
    """
    current_os = platform.system()
    if current_os == 'Windows':
        try:
            proc = psutil.Process(pid)
            bring_to_front_windows(proc)
        except psutil.NoSuchProcess:
            print("Process no longer exists.")
    else:
        print("Bringing window to front is not implemented for this OS.")

def launch_process(launch_cmd):
    """
    Launch the target process.
    """
    try:
        proc = subprocess.Popen(launch_cmd)
        print("Process launched successfully.")
        return proc.pid
    except Exception as e:
        print(f"Failed to launch process: {e}")
        sys.exit(1)

def main():
    # Load configuration
    config = load_config()
    stored_pid = config.get("pid", -1)

    # Verify if the stored PID is running
    if stored_pid != -1 and is_process_running_by_pid(stored_pid):
        print(f"Process already running with PID {stored_pid}. Bringing to front.")
        bring_to_front(stored_pid)
        return

    # Determine the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define paths
    image_pawcessor_dir = os.path.join(script_dir, 'imagePawcessor')
    image_pawcess_exe = os.path.join(image_pawcessor_dir, 'imagePawcess.exe')
    image_pawcess_script = os.path.join(script_dir, 'imagePawcessorScript', 'imagePawcess.py')

    # Determine what to launch
    if os.path.exists(image_pawcess_exe):
        launch_cmd = [image_pawcess_exe]
    else:
        launch_cmd = [sys.executable, image_pawcess_script]

    # Launch the process and update PID in config
    print("Launching process...")
    new_pid = launch_process(launch_cmd)
    save_pid_to_config(new_pid)

if __name__ == '__main__':
    main()
