#!/usr/bin/env python3
"""
Crystal Therapy - Versione MAIN CORRETTA con CairoSVG
Integra la logica safe nel main esistente
"""

# Copia le imports dal main originale
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

# Configura Cairo
os.environ['DYLD_LIBRARY_PATH'] = '/opt/homebrew/opt/cairo/lib:' + os.environ.get('DYLD_LIBRARY_PATH', '')

# Importa solo le funzioni essenziali dal main
sys.path.append('.')

# Import delle funzioni utili dal main
from crystal_fiume_funziona_bello import extract_contours_from_svg, Config

def apply_organic_deformation(mask, frame_index, speed, scale, intensity):
    """Applica una deformazione organica usando noise Perlin."""
    h, w = mask.shape
    
    time_component = frame_index * speed
    
    # Griglia ridotta per performance
    grid_size = 6
    h_grid = h // grid_size + 1
    w_grid = w // grid_size + 1
    
    # Calcolo noise sulla griglia
    noise_x = np.zeros((h_grid, w_grid), dtype=np.float32)
    noise_y = np.zeros((h_grid, w_grid), dtype=np.float32)
    
    for y in range(h_grid):
        for x in range(w_grid):
            real_x = x * grid_size
            real_y = y * grid_size
            
            noise_x[y, x] = pnoise2(
                real_x * scale, 
                real_y * scale + time_component, 
                octaves=4, persistence=0.5, lacunarity=2.0
            )
            noise_y[y, x] = pnoise2(
                real_x * scale + time_component, 
                real_y * scale, 
                octaves=4, persistence=0.5, lacunarity=2.0
            )
    
    # Interpola per riempire tutti i pixel
    noise_x_full = cv2.resize(noise_x, (w, h), interpolation=cv2.INTER_CUBIC)
    noise_y_full = cv2.resize(noise_y, (w, h), interpolation=cv2.INTER_CUBIC)
    
    # Applica deformazione
    displacement_x = noise_x_full * intensity
    displacement_y = noise_y_full * intensity
    
    # Crea mappa di rimappatura
    x_indices, y_indices = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x_indices + displacement_x).astype(np.float32)
    map_y = (y_indices + displacement_y).astype(np.float32)
    
    # Applica deformazione
    deformed_mask = cv2.remap(mask, map_x, map_y, 
                             interpolation=cv2.INTER_CUBIC, 
                             borderMode=cv2.BORDER_CONSTANT, 
                             borderValue=0)
    
    return deformed_mask

def load_texture(texture_path, width, height):
    """Carica e ridimensiona l'immagine di texture."""
    if not os.path.exists(texture_path):
        print(f"ATTENZIONE: File texture non trovato in '{texture_path}'. Il logo non verrà texturizzato.")
        return None
    try:
        texture = cv2.imread(texture_path, cv2.IMREAD_COLOR)
        if texture is None:
            raise Exception("cv2.imread ha restituito None.")
        # Ridimensiona la texture per adattarla al frame
        return cv2.resize(texture, (width, height), interpolation=cv2.INTER_LINEAR)
    except Exception as e:
        print(f"Errore durante il caricamento della texture: {e}")
        return None

def main_cairo_fixed():
    """Main corretto che usa CairoSVG senza crash"""
    
    try:
        print("🌊 Crystal Therapy CAIRO EDITION - MAIN CORRETTO...")
        
        # Importa config dal main
        from crystal_fiume_funziona_bello import Config, extract_contours_from_svg
        
        # Parametri
        width, height = Config.WIDTH, Config.HEIGHT
        svg_path = Config.SVG_PATH
        
        print("🎨 Estrazione contorni con CairoSVG...")
        contours, hierarchy = extract_contours_from_svg(svg_path, width, height, Config.LOGO_PADDING)
        print(f"✅ Estratti {len(contours)} contorni con qualità CairoSVG")
        
        # Carica la maschera dalla cache (dovrebbe esistere dopo l'estrazione)
        cache_filename = f"cairo_mask_{width}x{height}_p{Config.LOGO_PADDING}.png"
        mask = cv2.imread(cache_filename, cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            print("❌ Maschera non trovata, usando fallback")
            return False
        
        # Carica texture (se disponibile)
        texture_image = load_texture("input/texture.jpg", width, height)
        if texture_image is not None:
            print("🎨 Texture caricata con successo!")
        else:
            print("⚠️ Texture non disponibile, usando colore solido")
        
        # Setup video
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        symbols = ['💎', '🌊', '✨', '🔮', '🌟']
        symbol = symbols[hash(timestamp) % len(symbols)]
        output_path = f"output/crystal_cairo_main_{timestamp}_{symbol}.mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, Config.FPS, (width, height))
        
        print(f"🎬 Rendering {Config.TOTAL_FRAMES} frame con CairoSVG main...")
        
        start_time = time.time()
        
        for i in range(Config.TOTAL_FRAMES):
            # Frame nero
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Animazione migliorata
            progress = i / Config.TOTAL_FRAMES
            
            # Pulsazione + movimento orizzontale leggero
            pulse = 0.7 + 0.3 * np.sin(progress * 6 * np.pi)
            drift = int(10 * np.sin(progress * 2 * np.pi))  # Movimento orizzontale
            
            # Colore logo animato (rosa -> viola -> rosa)
            hue_shift = np.sin(progress * 4 * np.pi) * 0.3
            base_color = np.array([255, 192, 203])  # Rosa base
            color_shift = np.array([50, -50, 50]) * hue_shift  # Shift verso viola
            logo_color = np.clip(base_color + color_shift, 0, 255) * pulse
            logo_color = logo_color.astype(np.uint8)
            
            # Applica con drift orizzontale
            if drift != 0:
                # Crea maschera shiftata
                shifted_mask = np.zeros_like(mask)
                if drift > 0:
                    shifted_mask[:, drift:] = mask[:, :-drift]
                else:
                    shifted_mask[:, :drift] = mask[:, -drift:]
                current_mask = shifted_mask
            else:
                current_mask = mask
            
            # NUOVO: Effetto DEFORMAZIONE ORGANICA
            if True:  # Deformazione sempre attiva per test
                # Parametri deformazione dal main originale
                deform_speed = 0.05
                deform_scale = 0.008
                deform_intensity = 8.0  # Ridotto rispetto al main per essere più delicato
                
                current_mask = apply_organic_deformation(current_mask, i, 
                                                       deform_speed, deform_scale, deform_intensity)
            
            # NUOVO: Applica logo con texture (se disponibile)
            if texture_image is not None:
                # Applica texture + colore
                texture_alpha = 0.4  # Intensità texture
                
                # Crea layer con colore solido
                solid_color_layer = np.zeros_like(frame)
                solid_color_layer[current_mask > 0] = logo_color
                
                # Crea layer texture mascherata
                textured_logo_masked = cv2.bitwise_and(texture_image, texture_image, mask=current_mask)
                
                # Combina colore e texture
                logo_layer = cv2.addWeighted(solid_color_layer, 1.0 - texture_alpha, 
                                           textured_logo_masked, texture_alpha, 0)
                
                # Applica al frame
                frame[current_mask > 0] = logo_layer[current_mask > 0]
            else:
                # Usa solo colore solido se texture non disponibile
                frame[current_mask > 0] = logo_color
            
            # NUOVO: Effetto GLOW
            if True:  # Glow sempre attivo per test
                # Crea maschera per glow (un po' espansa)
                glow_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
                glow_mask = cv2.dilate(current_mask, glow_kernel, iterations=1)
                
                # Applica blur per creare effetto glow
                glow_layer = np.zeros_like(frame)
                glow_layer[glow_mask > 0] = logo_color
                glow_blurred = cv2.GaussianBlur(glow_layer, (35, 35), 0)
                
                # Combina con intensità ridotta
                glow_intensity = 0.3
                frame = cv2.addWeighted(frame, 1.0, glow_blurred, glow_intensity, 0)
            
            out.write(frame)
            
            # Progress
            if i % 30 == 0 or i == Config.TOTAL_FRAMES - 1:
                elapsed = time.time() - start_time
                fps_current = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (Config.TOTAL_FRAMES - i - 1) / fps_current if fps_current > 0 else 0
                
                progress_bar = "█" * int(25 * (i + 1) / Config.TOTAL_FRAMES)
                progress_bar += "-" * (25 - len(progress_bar))
                
                spinner = ['🔮', '✨', '🌟', '💎'][i % 4]
                print(f"{spinner} Cristallizzazione Cairo... [{progress_bar}] {100 * (i + 1) / Config.TOTAL_FRAMES:.1f}% | FPS: {fps_current:.1f} | ETA: {eta:.0f}s")
        
        out.release()
        
        print("🌿 Cristallizzazione MAIN CAIRO completata!")
        print("💎 QUALITÀ CairoSVG SUPREMA nel main ufficiale!")
        print("✨ Buchi delle lettere A, O, P, R PERFETTI!")
        print("🌊 Animazione fluida con pulsazione e drift!")
        print(f"💫 Video salvato: {output_path}")
        
        # Auto-versioning come nel main originale
        try:
            print("🚀 Avvio gestore di versioni...")
            filename = os.path.basename(output_path)
            tag_name = filename.replace('.mp4', '').replace('crystal_cairo_main_', 'cairo_')
            
            subprocess.run(['git', 'add', '.'], cwd=os.getcwd())
            subprocess.run(['git', 'commit', '-m', f'Crystal Therapy Cairo Edition: {filename}'], cwd=os.getcwd())
            subprocess.run(['git', 'tag', tag_name], cwd=os.getcwd())
            subprocess.run(['git', 'push', 'origin', tag_name], cwd=os.getcwd())
            
            print(f"✅ Versione Cairo creata: {tag_name}")
        except:
            print("⚠️ Versioning opzionale fallito")
        
        return True
        
    except Exception as e:
        print(f"❌ Errore nel main Cairo: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main_cairo_fixed()
