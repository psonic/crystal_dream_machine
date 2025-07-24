#!/usr/bin/env python3
"""
Test rendering di un singolo frame per debug Cairo crash
"""

import cv2
import numpy as np
import os
import sys

# Aggiungi il path per importare le funzioni
sys.path.append('.')

def test_single_frame():
    """Test rendering di un singolo frame"""
    
    try:
        # Importa le configurazioni e funzioni dal main
        from crystal_fiume_funziona_bello import (
            Config, extract_contours_from_svg, create_unified_mask,
            apply_organic_deformation
        )
        
        print("🎨 Test rendering singolo frame con CairoSVG...")
        
        # Parametri
        width, height = Config.WIDTH, Config.HEIGHT
        svg_path = "input/logo.svg"
        
        # Estrai contorni
        print("📝 Estraendo contorni...")
        contours, hierarchy = extract_contours_from_svg(svg_path, width, height, 50)
        print(f"📐 Estratti {len(contours)} contorni")
        
        # Crea maschera unificata
        print("🎯 Creando maschera unificata...")
        unified_mask = create_unified_mask(contours, hierarchy, width, height, True, 0.01)
        print(f"🎯 Maschera creata: {unified_mask.shape}")
        
        # Crea frame base
        print("🖼️ Creando frame...")
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Applica deformazione organica (frame 0)
        print("🌊 Applicando deformazione organica...")
        deformed_mask = apply_organic_deformation(unified_mask, 0)  # frame 0
        print(f"🌊 Deformazione applicata: {deformed_mask.shape}")
        
        # Applica la maschera al frame
        print("✨ Applicando maschera al frame...")
        logo_color = (255, 192, 203)  # Rosa
        frame[deformed_mask > 0] = logo_color
        
        # Salva il frame di test
        output_path = "test_single_frame.png"
        cv2.imwrite(output_path, frame)
        print(f"💎 Frame salvato: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Errore durante test frame: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Configura Cairo
    os.environ['DYLD_LIBRARY_PATH'] = '/opt/homebrew/opt/cairo/lib:' + os.environ.get('DYLD_LIBRARY_PATH', '')
    test_single_frame()
