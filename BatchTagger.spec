# -*- mode: python ; coding: utf-8 -*-
import os
import sys
from PyInstaller.utils.hooks import collect_dynamic_libs
from PyInstaller.utils.hooks import collect_all

block_cipher = None
base_dir = os.path.abspath(os.getcwd())

datas = []
binaries = []
hiddenimports = []

# Collect onnxruntime
binaries += collect_dynamic_libs('onnxruntime')
tmp_ret = collect_all('onnxruntime')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

# Add libiomp5md.dll if exists in environment
# Check common locations relative to base_dir
libiomp_paths = [
    os.path.join(base_dir, 'python_env', 'Library', 'bin', 'libiomp5md.dll'),
    os.path.join(sys.prefix, 'Library', 'bin', 'libiomp5md.dll')
]

for libiomp in libiomp_paths:
    if os.path.isfile(libiomp):
        binaries += [(libiomp, '.')]
        break

a = Analysis(
    ['batch_tagger.py'],
    pathex=[base_dir],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='BatchTagger',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
