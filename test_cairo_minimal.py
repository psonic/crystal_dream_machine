#!/usr/bin/env python3
"""
Test rendering video minimale con maschera CairoSVG pre-generata
"""

import cv2
import numpy as np
import os

def test_cairo_video():
    """Test rendering video con maschera Cairo"""
    
    try:
        # Carica la maschera pre-generata
        mask_path = "cairo_mask_960x540_p1.png" 
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            print(f"❌ Impossibile caricare {mask_path}")
            return False
            
        print(f"✅ Maschera caricata: {mask.shape}")
        
        # Parametri video
        width, height = 960, 540
        fps = 30
        total_frames = 30  # Solo 1 secondo
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter('test_cairo_minimal.mp4', fourcc, fps, (width, height))
        
        print(f"🎬 Rendering {total_frames} frame...")
        
        for i in range(total_frames):
            # Crea frame nero
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Applica logo rosa
            logo_color = (255, 192, 203)  # Rosa BGR
            frame[mask > 0] = logo_color
            
            # Scrivi frame
            out.write(frame)
            
            if i % 10 == 0:
                print(f"📐 Frame {i+1}/{total_frames}")
        
        out.release()
        print("✅ Video test completato: test_cairo_minimal.mp4")
        return True
        
    except Exception as e:
        print(f"❌ Errore: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_cairo_video()
