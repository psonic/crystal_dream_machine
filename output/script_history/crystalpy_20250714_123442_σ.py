import cv2
import numpy as np
import time
import os
import sys
import subprocess
from collections import deque

from components.config import Config
from components.utils import (get_dynamic_parameters, get_timestamp_filename, 
                   C_BOLD, C_CYAN, C_GREEN, C_YELLOW, C_MAGENTA, C_BLUE, C_END, SPINNER_CHARS)
from components.svg_parser import extract_contours_from_svg, extract_contours_from_pdf, create_unified_mask
from components.texture_manager import find_texture_file, load_texture
from components.deformations import apply_organic_deformation, apply_lens_deformation, initialize_lenses
from components.effects import process_background, extract_logo_tracers, apply_advanced_blending

def render_frame(contours, hierarchy, width, height, frame_index, total_frames, config, bg_frame, texture_image, tracer_history, bg_tracer_history, lenses):
    """
    Rende un singolo frame dell'animazione, applicando la pipeline di effetti completa.
    """
    dynamic_params = get_dynamic_parameters(frame_index, total_frames)

    bg_result = process_background(bg_frame, config)
    if len(bg_result) == 3:
        final_frame, current_logo_edges, current_bg_edges = bg_result
    else:
        final_frame, current_logo_edges = bg_result
        current_bg_edges = None
    
    if config.TRACER_ENABLED and len(tracer_history) > 0:
        tracer_layer = np.zeros_like(final_frame, dtype=np.float32)
        dynamic_opacity = config.TRACER_MAX_OPACITY * dynamic_params.get('tracer_opacity_multiplier', 1.0)
        opacities = np.linspace(0, dynamic_opacity, len(tracer_history))
        
        for i, past_edges in enumerate(reversed(tracer_history)):
            hue_shift = (frame_index * 0.1 + i * 0.5) % 180
            base_color_hsv = cv2.cvtColor(np.uint8([[config.TRACER_BASE_COLOR]]), cv2.COLOR_BGR2HSV)[0][0]
            new_hue = (base_color_hsv[0] + hue_shift) % 180
            dynamic_color_hsv = np.uint8([[[new_hue, base_color_hsv[1], base_color_hsv[2]]]])
            dynamic_color_bgr = cv2.cvtColor(dynamic_color_hsv, cv2.COLOR_HSV2BGR)[0][0]
            
            colored_tracer = cv2.cvtColor(past_edges, cv2.COLOR_GRAY2BGR).astype(np.float32)
            colored_tracer[past_edges > 0] = np.array(dynamic_color_bgr, dtype=np.float32)
            tracer_with_opacity = cv2.multiply(colored_tracer, opacities[i])
            tracer_layer = cv2.add(tracer_layer, tracer_with_opacity)
            
        final_frame = cv2.add(final_frame.astype(np.float32), tracer_layer)
        final_frame = np.clip(final_frame, 0, 255).astype(np.uint8)

    if hasattr(config, 'BG_TRACER_ENABLED') and config.BG_TRACER_ENABLED and len(bg_tracer_history) > 0:
        bg_tracer_layer = np.zeros_like(final_frame, dtype=np.float32)
        dynamic_bg_opacity = config.BG_TRACER_MAX_OPACITY * dynamic_params.get('bg_tracer_opacity_multiplier', 1.0)
        bg_opacities = np.linspace(0, dynamic_bg_opacity, len(bg_tracer_history))
        
        for i, past_bg_edges in enumerate(reversed(bg_tracer_history)):
            hue_shift_bg = (frame_index * 0.05 + i * 0.3) % 180
            base_color_hsv_bg = cv2.cvtColor(np.uint8([[config.BG_TRACER_BASE_COLOR]]), cv2.COLOR_BGR2HSV)[0][0]
            new_hue_bg = (base_color_hsv_bg[0] + hue_shift_bg) % 180
            dynamic_color_hsv_bg = np.uint8([[[new_hue_bg, base_color_hsv_bg[1], base_color_hsv_bg[2]]]])
            dynamic_color_bgr_bg = cv2.cvtColor(dynamic_color_hsv_bg, cv2.COLOR_HSV2BGR)[0][0]
            
            colored_bg_tracer = cv2.cvtColor(past_bg_edges, cv2.COLOR_GRAY2BGR).astype(np.float32)
            colored_bg_tracer[past_bg_edges > 0] = np.array(dynamic_color_bgr_bg, dtype=np.float32)
            bg_tracer_with_opacity = cv2.multiply(colored_bg_tracer, bg_opacities[i])
            bg_tracer_layer = cv2.add(bg_tracer_layer, bg_tracer_with_opacity)
            
        final_frame = cv2.add(final_frame.astype(np.float32), bg_tracer_layer)
        final_frame = np.clip(final_frame, 0, 255).astype(np.uint8)

    logo_mask = create_unified_mask(contours, hierarchy, width, height, config.SMOOTHING_ENABLED, config.SMOOTHING_FACTOR)

    if config.DEFORMATION_ENABLED:
        deformation_params = {
            'speed': config.DEFORMATION_SPEED,
            'scale': config.DEFORMATION_SCALE,
            'intensity': config.DEFORMATION_INTENSITY
        }
        logo_mask = apply_organic_deformation(logo_mask, frame_index, deformation_params)

    if config.LENS_DEFORMATION_ENABLED:
        logo_mask = apply_lens_deformation(logo_mask, lenses, frame_index, config, dynamic_params)

    logo_tracers = extract_logo_tracers(logo_mask, config)
    combined_logo_edges = cv2.add(current_logo_edges, logo_tracers)

    logo_layer = np.zeros_like(final_frame)
    glow_layer = np.zeros_like(final_frame)

    if config.TEXTURE_ENABLED and texture_image is not None:
        solid_color_layer = np.zeros_like(final_frame)
        solid_color_layer[logo_mask > 0] = config.LOGO_COLOR
        textured_logo_masked = cv2.bitwise_and(texture_image, texture_image, mask=logo_mask)
        logo_layer = cv2.addWeighted(solid_color_layer, 1.0 - config.TEXTURE_ALPHA, textured_logo_masked, config.TEXTURE_ALPHA, 0)
    else:
        logo_layer[logo_mask > 0] = config.LOGO_COLOR

    if config.GLOW_ENABLED:
        ksize = config.GLOW_KERNEL_SIZE if config.GLOW_KERNEL_SIZE % 2 != 0 else config.GLOW_KERNEL_SIZE + 1
        blurred_mask = cv2.GaussianBlur(logo_mask, (ksize, ksize), 0)
        glow_mask_3ch = cv2.cvtColor(blurred_mask, cv2.COLOR_GRAY2BGR)
        glow_effect = cv2.multiply(glow_mask_3ch, np.array(config.LOGO_COLOR, dtype=np.float32) / 255.0, dtype=cv2.CV_32F)
        glow_layer = np.clip(glow_effect * dynamic_params['glow_intensity'], 0, 255).astype(np.uint8)

    final_frame_with_glow = cv2.add(final_frame, glow_layer)
    final_logo_layer = np.zeros_like(final_frame)
    logo_mask_bool = logo_mask > 0
    final_logo_layer[logo_mask_bool] = logo_layer[logo_mask_bool]

    if config.ADVANCED_BLENDING:
        final_frame = apply_advanced_blending(final_frame_with_glow, final_logo_layer, logo_mask, config)
    else:
        final_frame_with_glow[logo_mask_bool] = 0
        final_frame = cv2.add(final_frame_with_glow, final_logo_layer)

    return final_frame, combined_logo_edges, current_bg_edges


def main():
    """Funzione principale per generare l'animazione del logo."""
    output_filename = None
    out = None
    bg_video = None
    
    try:
        print(f"{C_BOLD}{C_CYAN}🌊 Avvio rendering Crystal Therapy MOVIMENTO GARANTITO...{C_END}")
        print(f"✓ TEST MODE: 30fps, 10s, codec multipli per compatibilità")
        source_type = "SVG vettoriale" if Config.USE_SVG_SOURCE else "PDF rasterizzato"
        print(f"✓ Sorgente: {source_type} con smoothing ottimizzato")
        print(f"🌊 Deformazione ORGANICA POTENZIATA + LENTI DINAMICHE")
        print(f"💫 MOVIMENTO VISIBILE: Speed x80, Lenti x27 più veloci!")
        print(f"🐌 SFONDO RALLENTATO: Video a metà velocità!")
        print(f"✨ Traccianti + Blending + Glow COMPATIBILE")
        print(f"✓ Variazione dinamica + codec video testati")
        print(f"💎 RENDERING MOVIMENTO GARANTITO per compatibilità VLC/QuickTime!")
        
        if Config.USE_SVG_SOURCE:
            contours, hierarchy = extract_contours_from_svg(Config.SVG_PATH, Config.WIDTH, Config.HEIGHT, Config.LOGO_PADDING)
        else:
            contours, hierarchy = extract_contours_from_pdf(Config.PDF_PATH, Config.WIDTH, Config.HEIGHT, Config.LOGO_PADDING)

        if not contours:
            source_name = "SVG" if Config.USE_SVG_SOURCE else "PDF"
            print(f"Errore critico: nessun contorno valido trovato nel {source_name}. Uscita.")
            return

        print("Estrazione contorni riuscita.")

        texture_image = None
        if Config.TEXTURE_ENABLED:
            texture_path = find_texture_file()
            texture_image = load_texture(texture_path, Config.WIDTH, Config.HEIGHT)
            if texture_image is not None:
                print("Texture infusa con l'essenza del Natisone - Alex Ortiga.")
        else:
            print("La texturizzazione del logo è disabilitata.")

        bg_video = None
        if Config.BACKGROUND_VIDEO_ENABLED:
            bg_video = cv2.VideoCapture(Config.BACKGROUND_VIDEO_PATH)
            if not bg_video.isOpened():
                print(f"Errore: impossibile aprire il video di sfondo in {Config.BACKGROUND_VIDEO_PATH}")
                bg_video = None
            else:
                bg_total_frames = int(bg_video.get(cv2.CAP_PROP_FRAME_COUNT))
                bg_fps = bg_video.get(cv2.CAP_PROP_FPS)
                print(f"🎬 Video sfondo: {bg_total_frames} frame @ {bg_fps}fps")
                print(f"🐌 RALLENTAMENTO ATTIVATO: Video sfondo a metà velocità")
        else:
            print("📺 Sfondo video disabilitato - usando sfondo nero")
        
        if Config.WHATSAPP_COMPATIBLE:
            fourcc = cv2.VideoWriter_fourcc(*'H264')
            print("🔄 Usando H.264 per compatibilità WhatsApp...")
        else:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
        output_filename = get_timestamp_filename()
        out = cv2.VideoWriter(output_filename, fourcc, Config.FPS, (Config.WIDTH, Config.HEIGHT))
        
        if not out.isOpened():
            print("TENTATIVO 1 FALLITO. Provo con mp4v...")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_filename, fourcc, Config.FPS, (Config.WIDTH, Config.HEIGHT))
            
        if not out.isOpened():
            print("TENTATIVO 2 FALLITO. Provo con XVID...")
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_filename, fourcc, Config.FPS, (Config.WIDTH, Config.HEIGHT))
            
        if not out.isOpened():
            print("ERRORE CRITICO: Nessun codec video funziona!")
            return
        
        tracer_history = deque(maxlen=Config.TRACER_TRAIL_LENGTH)
        bg_tracer_history = deque(maxlen=getattr(Config, 'BG_TRACER_TRAIL_LENGTH', 35))

        lenses = []
        if Config.LENS_DEFORMATION_ENABLED:
            lenses = initialize_lenses(Config)
            print(f"🌊 Liberate {len(lenses)} creature liquide dal Natisone per Alex Ortiga.")

        print(f"Rendering dell'animazione in corso... ({Config.TOTAL_FRAMES} frame da elaborare)")
        start_time = time.time()
        
        for i in range(Config.TOTAL_FRAMES):
            if bg_video:
                bg_frame_index = int(i / 2.0)
                bg_video.set(cv2.CAP_PROP_POS_FRAMES, bg_frame_index)
                ret, bg_frame = bg_video.read()
                
                if not ret:
                    bg_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, bg_frame = bg_video.read()
                bg_frame = cv2.resize(bg_frame, (Config.WIDTH, Config.HEIGHT))
            else:
                bg_frame = np.zeros((Config.HEIGHT, Config.WIDTH, 3), dtype=np.uint8)

            frame_result = render_frame(contours, hierarchy, Config.WIDTH, Config.HEIGHT, i, Config.TOTAL_FRAMES, Config, bg_frame, texture_image, tracer_history, bg_tracer_history, lenses)
            
            if len(frame_result) == 3:
                frame, current_logo_edges, current_bg_edges = frame_result
            else:
                frame, current_logo_edges = frame_result
                current_bg_edges = None
            
            if Config.TRACER_ENABLED:
                tracer_history.append(current_logo_edges)
            
            if hasattr(Config, 'BG_TRACER_ENABLED') and Config.BG_TRACER_ENABLED and current_bg_edges is not None:
                bg_tracer_history.append(current_bg_edges)
            
            out.write(frame)
            
            elapsed = time.time() - start_time
            fps = (i + 1) / elapsed if elapsed > 0 else 0
            
            remaining_frames = Config.TOTAL_FRAMES - (i + 1)
            eta_seconds = remaining_frames / fps if fps > 0 else 0
            eta_minutes, eta_sec = divmod(int(eta_seconds), 60)
            eta_str = f"{eta_minutes:02d}:{eta_sec:02d}"

            progress = (i + 1) / Config.TOTAL_FRAMES
            bar_length = 25
            filled_length = int(bar_length * progress)
            
            progress_color_map = [C_MAGENTA, C_BLUE, C_CYAN, C_GREEN, C_YELLOW]
            color_index = int(progress * (len(progress_color_map) -1))
            bar_color = progress_color_map[color_index]
            bar = f"{bar_color}{'█' * filled_length}{C_END}{'-' * (bar_length - filled_length)}"
            
            spinner = SPINNER_CHARS[i % len(SPINNER_CHARS)]

            log_message = (
                f"\r{spinner} {C_BOLD}{C_GREEN}Cristallizzazione...{C_END} "
                f"{C_CYAN}[{bar}] {C_END}{progress:.1%} "
                f"| {C_YELLOW}FPS: {fps:.2f}{C_END} "
                f"| {C_MAGENTA}ETA: {eta_str}{C_END} "
            )
            print(log_message, end="")
        
        print(f"\n{C_BOLD}{C_GREEN}🌿 Cristallizzazione ULTRA completata con effetti IPNOTICI!{C_END}")
        print(f"💥 Deformazioni organiche ESAGERATE ma ultra-fluide!")
        print(f"✓ Traccianti DOPPI (logo rosa + sfondo viola) dinamici!")
        print(f"💎 Qualità SUPREMA (1000 DPI, smoothing perfetto)!")
        print(f"🔮 Movimento IPNOTICO e curioso - Alex Ortiga & TV Int ULTIMATE!")

    except Exception as e:
        print(f"\n{C_BOLD}{C_YELLOW}An error occurred during rendering: {e}{C_END}")

    finally:
        if out and out.isOpened():
            out.release()
        if bg_video and bg_video.isOpened(): 
            bg_video.release()
        
        if output_filename and os.path.exists(output_filename):
            print(f"Animazione salvata in: {C_BOLD}{output_filename}{C_END}")

            try:
                print(f"\n{C_BLUE}🚀 Avvio gestore di versioni...{C_END}")
                source_script_path = os.path.abspath(__file__)
                version_manager_path = os.path.join(os.path.dirname(source_script_path), 'version_manager.py')
                
                if os.path.exists(version_manager_path):
                    result = subprocess.run(
                        [sys.executable, version_manager_path, source_script_path, output_filename],
                        capture_output=True,
                        text=True,
                        check=False
                    )
                    print(result.stdout)
                    if result.stderr:
                        if "nothing to commit" in result.stderr.lower():
                             print(f"{C_GREEN}ℹ️ Nessuna nuova modifica da salvare nel versionamento.{C_END}")
                        else:
                            print(f"{C_YELLOW}Output di errore dal gestore versioni:{C_END}\n{result.stderr}")
                else:
                    print(f"{C_YELLOW}ATTENZIONE: version_manager.py non trovato. Saltando il versionamento.{C_END}")

            except Exception as e:
                print(f"{C_YELLOW}Errore inatteso durante il versionamento: {e}{C_END}")
        else:
            print(f"{C_YELLOW}Nessun file di output generato o trovato. Saltando il versionamento.{C_END}")

if __name__ == "__main__":
    main()
