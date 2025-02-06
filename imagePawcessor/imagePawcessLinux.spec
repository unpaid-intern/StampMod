# -*- mode: python ; coding: utf-8 -*-

# Import collect_submodules to gather all submodules of Numba
from PyInstaller.utils.hooks import collect_submodules

# Collect all submodules of Numba to ensure none are missed
numba_hidden_imports = collect_submodules('numba')

a = Analysis(
    ['imagePawcessLinux.py'],  # Your main script
    pathex=[],  # Add any additional search paths if necessary
    binaries=[],  # Add any additional binary files if necessary
    datas=[],  # Add any additional data files if necessary
    hiddenimports=[
        'sklearn.tree._partitioner',  # Existing hidden import
    ] + numba_hidden_imports,  # Add all Numba submodules
    hookspath=[],  # Paths to additional hooks if any
    hooksconfig={},  # Configuration for hooks if any
    runtime_hooks=[],  # Any runtime hooks if necessary
    excludes=[],  # Modules to exclude if any
    noarchive=True,  # Set to False if you want to bundle Python files in a ZIP
    optimize=0,  # Optimization level (0 = none)
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,  # Include the main script in the EXE
    a.binaries,  # Include any binaries
    a.datas,  # Include any data files
    [],
    name='imagePawcess',  # Name of the final executable
    debug=False,  # Set to True for debug mode
    bootloader_ignore_signals=False,  # Set to True to ignore signals during boot
    strip=False,  # Strips symbols from binaries (leave False if unsure)
    upx=True,  # Enable UPX compression (requires UPX installed)
    upx_exclude=[],  # Exclude certain files from UPX compression
    runtime_tmpdir=None,  # Temporary directory for runtime files
    console=False,  # Set to True for a console window, False for GUI-only
    disable_windowed_traceback=False,  # Disable traceback for GUI apps
    argv_emulation=False,  # Mac-specific (leave False for Linux)
    target_arch=None,  # Architecture (leave None for auto-detect)
    codesign_identity=None,  # Mac-specific (leave None for Linux)
    entitlements_file=None,  # Mac-specific (leave None for Linux)
)
