# setup.py

import sys
import os
from cx_Freeze import setup, Executable

build_exe_options = {
    "packages": [
        "os",
        "sys",
        "random",
        "threading",
        "tempfile",
        "json",
        "time",
        "math",
        "pathlib",
        "concurrent.futures",
        "webbrowser",
        "cv2",
        "PIL",
        "numpy",
        "logging",
        "shutil",
        "sklearn",
        "joblib",
        "PySide6",
    ],
    "include_files": [
        "imagePawcess.ico",
        "assets/",
    ],
    "include_msvcr": True,
    "excludes": [
    ],
    "optimize": 2,
    "build_exe": "build",
}


base = None
if sys.platform == "win32":
    base = "Win32GUI"

# Executable settings
executables = [
    Executable(
        script="imagePawcess.py",
        base=base,
        target_name="imagePawcess.exe",
        icon="imagePawcess.ico",  # Set the application icon
    )
]

# Setup configuration
setup(
    name="ImagePawcess",
    version="1.0",
    description="A description of your application",
    options={"build_exe": build_exe_options},
    executables=executables,
)
