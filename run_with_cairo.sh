#!/bin/bash

# Crystal Therapy con Cairo configurato
export DYLD_LIBRARY_PATH=/opt/homebrew/opt/cairo/lib:$DYLD_LIBRARY_PATH
export PKG_CONFIG_PATH=/opt/homebrew/opt/cairo/lib/pkgconfig:$PKG_CONFIG_PATH

echo "🔧 Variabili Cairo configurate"
echo "🚀 Avviando Crystal Therapy con rendering professionale..."

cd /Users/daniele/CrystalPython2
/Users/daniele/CrystalPython2/.venv/bin/python crystal_fiume_funziona_bello.py
