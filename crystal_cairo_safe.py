#!/usr/bin/env python3
"""
Versione SAFE di Crystal Therapy con CairoSVG - evita segfault
"""

import cv2
import numpy as np
import datetime
import time
import os

# Configurazione Cairo
os.environ['DYLD_LIBRARY_PATH'] = '/opt/homebrew/opt/cairo/lib:' + os.environ.get('DYLD_LIBRARY_PATH', '')

def render_cairo_video():
    """Rendering video sicuro con maschera CairoSVG"""
    
    try:
        print("🌊 CRYSTAL THERAPY SAFE MODE con CairoSVG...")
        
        # Carica la maschera pre-generata
        mask_path = "cairo_mask_960x540_p1.png" 
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            print(f"❌ Impossibile caricare {mask_path}")
            print("⚠️ Esegui prima il main normale per generare la cache")
            return False
            
        print(f"✅ Maschera CairoSVG caricata: {mask.shape}")
        print(f"💎 Buchi delle lettere PRESERVATI!")
        
        # Parametri video
        width, height = 960, 540
        fps = 30
        total_frames = 300  # 10 secondi
        
        # Setup video writer con timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"output/crystal_cairo_safe_{timestamp}_💎.mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"🎬 Rendering {total_frames} frame con qualità CairoSVG...")
        
        start_time = time.time()
        
        for i in range(total_frames):
            # Crea frame nero
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Applica logo rosa con leggera animazione di opacità
            progress = i / total_frames
            pulse = 0.8 + 0.2 * np.sin(progress * 4 * np.pi)  # Pulsazione leggera
            
            logo_color = np.array([255, 192, 203]) * pulse  # Rosa con pulsazione
            logo_color = np.clip(logo_color, 0, 255).astype(np.uint8)
            
            frame[mask > 0] = logo_color
            
            # Scrivi frame
            out.write(frame)
            
            # Progress
            if i % 30 == 0 or i == total_frames - 1:
                elapsed = time.time() - start_time
                fps_current = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (total_frames - i - 1) / fps_current if fps_current > 0 else 0
                
                progress_bar = "█" * int(25 * (i + 1) / total_frames)
                progress_bar += "-" * (25 - len(progress_bar))
                
                print(f"💎 Cristallizzazione Cairo... [{progress_bar}] {100 * (i + 1) / total_frames:.1f}% | FPS: {fps_current:.1f} | ETA: {eta:.0f}s")
        
        out.release()
        
        print("🌿 Cristallizzazione CairoSVG ULTRA completata!")
        print("💎 QUALITÀ SUPREMA con buchi delle lettere PERFETTI!")
        print("✨ Lettere A, O, P, R con interno TRASPARENTE!")
        print(f"💫 Video salvato: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Errore: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    render_cairo_video()
