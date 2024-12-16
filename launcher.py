import os
import sys
import json
import subprocess
import platform
import psutil
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    filename='launcher.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Only import Windows-specific libraries if on Windows
if platform.system() == 'Windows':
    try:
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
    def get_base_path() -> Path:
        if getattr(sys, 'frozen', False):
            # If the application is frozen, use the executable's directory
            base_path = Path(sys.executable).parent
        else:
            # If not frozen, use the script's directory
            base_path = Path(__file__).parent

        # Trim just the current script directory
        return base_path.parent
    
    base_path = get_base_path()

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
    config_path = get_config_path()
    if not config_path.exists():
        with open(config_path, 'w') as file:
            json.dump({"pid": -1}, file, indent=4)
        logging.info(f"Created default config at {config_path}")
        return {"pid": -1}
    with open(config_path, 'r') as file:
        config = json.load(file)
        logging.debug(f"Loaded config: {config}")
        return config


def is_process_running_by_pid(pid: int) -> bool:
    try:
        proc = psutil.Process(pid)
        running = proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
        logging.debug(f"Process with PID {pid} is_running: {running}")
        return running
    except (psutil.NoSuchProcess, psutil.ZombieProcess, psutil.AccessDenied) as e:
        logging.warning(f"Process with PID {pid} check failed: {e}")
        return False


def bring_to_front_windows(pid):
    try:
        def callback(hwnd, pid_to_match):
            _, found_pid = win32process.GetWindowThreadProcessId(hwnd)
            if found_pid == pid_to_match and win32gui.IsWindowVisible(hwnd):
                win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                win32gui.SetForegroundWindow(hwnd)
                return False  # Stop enumeration
            return True  # Continue enumeration

        win32gui.EnumWindows(callback, pid)
        logging.info(f"Brought process with PID {pid} to front.")
    except Exception as e:
        logging.error(f"Failed to bring window to front: {e}")


def bring_to_front(pid):
    if platform.system() == 'Windows':
        bring_to_front_windows(pid)


def launch_process(launch_cmd):
    """
    Launch the target process and fully detach it so the parent launcher can exit safely.
    """
    try:
        if platform.system() == 'Windows':
            proc = subprocess.Popen(
                launch_cmd,
                creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        else:
            proc = subprocess.Popen(
                launch_cmd,
                start_new_session=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True
            )

        logging.info(f"Process launched successfully with PID {proc.pid}")

        # Update config with new PID
        config_path = get_config_path()
        with open(config_path, 'w') as file:
            json.dump({"pid": proc.pid}, file, indent=4)
        logging.debug(f"Updated config with PID {proc.pid}")

        print(f"Process launched successfully with PID {proc.pid}")
        return proc.pid
    except Exception as e:
        logging.error(f"Failed to launch process: {e}")
        print(f"Failed to launch process: {e}")
        sys.exit(1)


def main():
    logging.debug("Starting main function")
    # Load the existing config
    config = load_config()
    stored_pid = config.get("pid", -1)

    # Check if the stored PID is valid and running
    if stored_pid != -1 and is_process_running_by_pid(stored_pid):
        print(f"Process already running with PID {stored_pid}. Bringing it to the front.")
        logging.info(f"Process already running with PID {stored_pid}. Bringing it to the front.")
        bring_to_front(stored_pid)
        return

    print("No valid process found. Launching a new process...")
    logging.info("No valid process found. Launching a new process...")

    # Resolve base path
    try:
        if getattr(sys, 'frozen', False):
            base_path = Path(sys.executable).parent
            logging.debug(f"Running as a frozen executable. Base path: {base_path}")
        else:
            base_path = Path(__file__).parent
            logging.debug(f"Running as a script. Base path: {base_path}")

        # Since the user mentioned the executables/scripts are in the same directory,
        # we do not navigate to the parent directory
        # base_path = base_path.parent

        image_pawcess_exe = base_path / 'imagePawcess.exe'
        image_pawcess_script = base_path / 'imagePawcess.py'

        logging.debug(f"Looking for executable at: {image_pawcess_exe}")
        logging.debug(f"Looking for script at: {image_pawcess_script}")

        if image_pawcess_exe.exists():
            print(f"Executable found: {image_pawcess_exe}")
            logging.info(f"Executable found: {image_pawcess_exe}")
            launch_cmd = [str(image_pawcess_exe)]
        elif image_pawcess_script.exists():
            print(f"Script found: {image_pawcess_script}")
            logging.info(f"Script found: {image_pawcess_script}")
            launch_cmd = [sys.executable, str(image_pawcess_script)]
        else:
            print("Neither executable nor script found. Check the paths!")
            logging.error("Neither executable nor script found. Check the paths!")
            sys.exit(1)

        # Launch the process
        launch_process(launch_cmd)
    except Exception as e:
        logging.error(f"Error in main: {e}")
        print(f"Error in main: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
