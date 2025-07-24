#!/usr/bin/env python3
"""
Test completo con debug per trovare dove crashia
"""

import cv2
import numpy as np
import datetime
from scipy.interpolate import splprep, splev
from noise import pnoise2
import multiprocessing
from functools import partial
import time
import os
from collections import deque
import xml.etree.ElementTree as ET
from svgpathtools import svg2paths2
import re
import subprocess
import sys
from io import BytesIO

# Import extract_contours_from_svg dalla main
sys.path.append('.')

def test_main_extraction():
    """Test dell'estrazione come nel main"""
    
    print("🎨 Test estrazione contorni dal main...")
    
    # Simula i parametri del main
    width, height = 1920, 1080
    padding = 50
    svg_path = "input/logo.svg"
    
    try:
        # Importa la funzione dal file principale
        from crystal_fiume_funziona_bello import extract_contours_from_svg
        
        print("📝 Chiamando extract_contours_from_svg...")
        contours, hierarchy = extract_contours_from_svg(svg_path, width, height, padding)
        
        print(f"✅ Estrazione completata!")
        print(f"📐 Trovati {len(contours)} contorni")
        
        if hierarchy is not None:
            external_count = sum(1 for h in hierarchy[0] if h[3] == -1)
            internal_count = len(contours) - external_count
            print(f"   🔹 {external_count} contorni esterni, {internal_count} buchi interni")
        
        print("🎯 Test estrazione: SUCCESSO")
        return True
        
    except Exception as e:
        print(f"❌ Errore durante estrazione: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_main_extraction()
