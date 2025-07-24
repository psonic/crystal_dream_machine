#!/usr/bin/env python3
"""
Test CairoSVG standalone per verificare il rendering
"""

import cv2
import numpy as np
import cairosvg
from PIL import Image
import io

def test_cairo_svg():
    svg_path = "input/logo.svg"
    width, height = 1000, 400
    
    try:
        print("🎨 Testando CairoSVG...")
        
        # Leggi SVG
        with open(svg_path, 'r') as f:
            svg_content = f.read()
        
        # Renderizza
        png_data = cairosvg.svg2png(
            bytestring=svg_content.encode('utf-8'), 
            output_width=width * 2, 
            output_height=height * 2,
            background_color='transparent'
        )
        
        # Converti in PIL
        pil_image = Image.open(io.BytesIO(png_data)).convert('RGBA')
        img_array = np.array(pil_image)
        
        # Estrai alpha channel
        alpha_channel = img_array[:, :, 3]
        mask = np.where(alpha_channel > 128, 255, 0).astype(np.uint8)
        
        # Ridimensiona
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_AREA)
        
        # Trova contorni
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        print(f"✅ CairoSVG funziona!")
        print(f"📐 Trovati {len(contours)} contorni")
        
        if hierarchy is not None:
            external_count = sum(1 for h in hierarchy[0] if h[3] == -1)
            internal_count = len(contours) - external_count
            print(f"   🔹 {external_count} contorni esterni, {internal_count} buchi interni")
        
        # Salva immagine di test
        cv2.imwrite('test_cairo_output.png', mask)
        print("💎 Immagine salvata come test_cairo_output.png")
        
        return True
        
    except Exception as e:
        print(f"❌ Errore: {e}")
        return False

if __name__ == "__main__":
    test_cairo_svg()
