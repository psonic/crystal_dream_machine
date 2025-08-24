import cv2
import numpy as np
import datetime
from scipy.interpolate import splprep, splev
from noise import pnoise2
import multiprocessing
from functools import partial
import time
import os
import argparse
from collections import deque
import subprocess
import sys

# Import dei nuovi moduli
# Configurazione caricata dinamicamente dal file config
Config = type('Config', (), {})()
from components.preview import run_preview_mode
from components.audio import (
    load_audio_analysis, 
    get_audio_reactive_factors, 
    get_organic_deformation_factors, 
    add_audio_to_video,
    load_audio_wrapper,
    AUDIO_AVAILABLE
)
from components.svg_pdf import (
    get_svg_dimensions,
    extract_contours_from_svg,
    extract_contours_from_svg_fallback,
    extract_contours_from_pdf,
    smooth_contour,
    create_unified_mask,
    create_gap_free_mask,
    PDF_AVAILABLE,
    SVG_PATHTOOLS_AVAILABLE
)
from components.lenses import (
    generate_cinematic_path,
    initialize_lenses,
    apply_lens_deformation
)
from components.deformations import (
    apply_organic_deformation,
    get_organic_deformation_params,
    apply_deformation_wrapper,
    NOISE_AVAILABLE
)
from components.blending import (
    apply_blending_preset,
    apply_texture_blending,
    apply_advanced_blending,
    print_blending_options,
    load_texture,
    find_texture_file,
    validate_blending_config,
    load_texture_wrapper,
    extract_logo_tracers
)

# CAIROSVG verrà importato solo se necessario (gestito nel componente svg_pdf)
CAIROSVG_AVAILABLE = None

# Disabilita il warning PIL per le immagini ad alta risoluzione
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # Rimuove il limite di sicurezza PIL
# --- FUNZIONI DI SUPPORTO ---

def get_dynamic_parameters(frame_index, total_frames):
    """
    Calcola parametri che cambiano automaticamente nel tempo per creare variazioni.
    """
    t = frame_index / total_frames  # Progresso animazione (0.0 a 1.0)
    params = {}

    # Pulsazione del glow
    glow_pulse = np.sin(t * np.pi)
    params['glow_intensity'] = Config.GLOW_INTENSITY + (glow_pulse * 0.2)

    # Variazioni automatiche dei parametri principali
    if Config.DYNAMIC_VARIATION_ENABLED:
        base_seed = frame_index * 0.001
        
        # Usa il nuovo componente deformazioni per parametri dinamici
        # (compatibilità mantenuta con i nomi della Config)
        enable_variation = Config.DYNAMIC_VARIATION_ENABLED
        deformation_params = get_organic_deformation_params(Config, enable_variation)
        
        params['deformation_speed'] = deformation_params['speed']
        params['deformation_scale'] = deformation_params['scale']
        params['deformation_intensity'] = deformation_params['intensity']
        
        # Variazioni medie per lenti
        lens_var_x = np.sin(base_seed * Config.VARIATION_SPEED_MEDIUM + 3.0) * Config.VARIATION_AMPLITUDE
        lens_var_y = np.cos(base_seed * Config.VARIATION_SPEED_MEDIUM + 5.5) * Config.VARIATION_AMPLITUDE
        
        params['lens_speed_factor'] = Config.LENS_SPEED_FACTOR * (1.0 + lens_var_x)
        params['lens_strength_multiplier'] = 1.0 + lens_var_y
        
        # Variazioni veloci per traccianti
        tracer_var_x = np.sin(base_seed * Config.VARIATION_SPEED_FAST + 2.0) * Config.VARIATION_AMPLITUDE
        tracer_var_y = np.cos(base_seed * Config.VARIATION_SPEED_FAST + 6.0) * Config.VARIATION_AMPLITUDE
        
        params['tracer_opacity_multiplier'] = 1.0 + tracer_var_x
        params['bg_tracer_opacity_multiplier'] = 1.0 + tracer_var_y
    else:
        # Usa valori fissi se le variazioni sono disabilitate
        deformation_params = get_organic_deformation_params(Config, False)
        params['deformation_speed'] = deformation_params['speed']
        params['deformation_scale'] = deformation_params['scale']
        params['deformation_intensity'] = deformation_params['intensity']
        params['lens_speed_factor'] = Config.LENS_SPEED_FACTOR
        params['lens_strength_multiplier'] = 1.0
        params['tracer_opacity_multiplier'] = 1.0
        params['bg_tracer_opacity_multiplier'] = 1.0
    
    return params

def get_timestamp_filename():
    """Genera nome file con timestamp e carattere decorativo."""
    now = datetime.datetime.now()
    magic_chars = ['α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'ι', 'κ', 'λ', 'μ', 'ν', 'ξ', 'ο', 'π', 'ρ', 'σ', 'τ', 'υ', 'φ', 'χ', 'ψ', 'ω', 'ॐ', '☯', '✨', 'Δ', 'Σ', 'Ω']
    magic_char = np.random.choice(magic_chars)
    
    # File di test vanno nella sottocartella test/
    if Config.TEST_MODE:
        return f"output/test/crystalpy_{now.strftime('%Y%m%d_%H%M%S')}_TEST_{magic_char}.mp4"
    else:
        return f"output/crystalpy_{now.strftime('%Y%m%d_%H%M%S')}_{magic_char}.mp4"

# --- FUNZIONI DI SUPPORTO ---

    if texture_image is None or alpha <= 0:
        return base_image.copy()
    
    # Converti in float32 per calcoli precisi
    base_float = base_image.astype(np.float32) / 255.0
    texture_float = texture_image.astype(np.float32) / 255.0
    
    # Applica blending mode
    if blending_mode == 'normal':
        # Normal: sovrapposizione diretta
        blended = texture_float
    
    elif blending_mode == 'overlay':
        # Overlay: moltiplica se base < 0.5, altrimenti screen
        condition = base_float < 0.5
        blended = np.where(condition, 
                          2 * base_float * texture_float,
                          1 - 2 * (1 - base_float) * (1 - texture_float))
    
    elif blending_mode == 'multiply':
        # Multiply: moltiplica i valori
        blended = base_float * texture_float
    
    elif blending_mode == 'screen':
        # Screen: inverso del multiply
        blended = 1 - (1 - base_float) * (1 - texture_float)
    
    elif blending_mode == 'soft_light':
        # Soft Light: versione più morbida di overlay
        condition = texture_float <= 0.5
        blended = np.where(condition,
                          base_float - (1 - 2 * texture_float) * base_float * (1 - base_float),
                          base_float + (2 * texture_float - 1) * (np.sqrt(base_float) - base_float))
    
    elif blending_mode == 'hard_light':
        # Hard Light: overlay invertito
        condition = texture_float < 0.5
        blended = np.where(condition,
                          2 * base_float * texture_float,
                          1 - 2 * (1 - base_float) * (1 - texture_float))
    
    elif blending_mode == 'color_dodge':
        # Color Dodge: schiarisce drasticamente
        blended = np.where(texture_float >= 1.0, 
                          1.0, 
                          np.minimum(1.0, base_float / (1.0 - texture_float + 1e-10)))
    
    elif blending_mode == 'color_burn':
        # Color Burn: scurisce drasticamente
        blended = np.where(texture_float <= 0.0,
                          0.0,
                          1.0 - np.minimum(1.0, (1.0 - base_float) / (texture_float + 1e-10)))
    
    elif blending_mode == 'darken':
        # Darken: prende il più scuro
        blended = np.minimum(base_float, texture_float)
    
    elif blending_mode == 'lighten':
        # Lighten: prende il più chiaro
        blended = np.maximum(base_float, texture_float)
    
    elif blending_mode == 'difference':
        # Difference: differenza assoluta
        blended = np.abs(base_float - texture_float)
    
    elif blending_mode == 'exclusion':
        # Exclusion: simile a difference ma più morbido
        blended = base_float + texture_float - 2 * base_float * texture_float
    
    else:
        # Default overlay
        condition = base_float < 0.5
        blended = np.where(condition, 
                          2 * base_float * texture_float,
                          1 - 2 * (1 - base_float) * (1 - texture_float))
    
    # Miscela con alpha
    result = base_float * (1 - alpha) + blended * alpha
    
    # Applica maschera se fornita
    if mask is not None:
        mask_norm = mask.astype(np.float32) / 255.0
        if len(mask_norm.shape) == 2:
            mask_norm = cv2.cvtColor(mask_norm, cv2.COLOR_GRAY2BGR)
        result = base_float * (1 - mask_norm) + result * mask_norm
    
    # Converti back a uint8
    return np.clip(result * 255, 0, 255).astype(np.uint8)

# Rimuovo la vettorizzazione che rallentava invece di velocizzare

def process_background(bg_frame, config):
    """
    Processa il frame di sfondo: lo adatta alle dimensioni video senza crop,
    lo scurisce e ne estrae i contorni per i traccianti.
    """
    h, w, _ = bg_frame.shape
    
    # 1. NUOVO: Usa video originale senza crop, adattalo alle dimensioni target
    if hasattr(config, 'BG_USE_ORIGINAL_SIZE') and config.BG_USE_ORIGINAL_SIZE:
        # Scala il video originale mantenendo le proporzioni
        target_width = config.WIDTH
        target_height = config.HEIGHT
        
        # Calcola scaling per coprire tutto il frame (come background)
        scale_x = target_width / w
        scale_y = target_height / h
        scale = max(scale_x, scale_y)  # Usa il maggiore per coprire tutto
        
        # Applica lo zoom configurabile moltiplicando il fattore di scala
        zoom_factor = getattr(config, 'BG_ZOOM_FACTOR', 1.0)
        scale = scale * zoom_factor
        
        # Nuove dimensioni scalate (ora con zoom)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Ridimensiona
        scaled_bg = cv2.resize(bg_frame, (new_w, new_h))
        
        # Centro-crop per adattare alle dimensioni esatte (il crop sarà più stretto con zoom > 1)
        start_x = (new_w - target_width) // 2
        start_y = (new_h - target_height) // 2
        final_bg = scaled_bg[start_y:start_y + target_height, start_x:start_x + target_width]
        
    else:
        # Metodo crop personalizzato per video verticali
        h, w, _ = bg_frame.shape
        
        # Calcola le dimensioni del crop basandosi sui ratio
        crop_width = int(w * config.BG_CROP_WIDTH_RATIO)
        crop_height = int(h * config.BG_CROP_HEIGHT_RATIO)
        
        # Calcola le coordinate di inizio
        crop_x_start = int(config.BG_CROP_X_START * (w - crop_width))
        crop_y_start = int(config.BG_CROP_Y_START * (h - crop_height))
        
        # Calcola le coordinate di fine
        crop_x_end = crop_x_start + crop_width
        crop_y_end = crop_y_start + crop_height
        
        # Esegue il crop
        cropped_bg = bg_frame[crop_y_start:crop_y_end, crop_x_start:crop_x_end]
        
        # Ridimensiona alla dimensione target
        final_bg = cv2.resize(cropped_bg, (config.WIDTH, config.HEIGHT))
    
    # 2. Scurisce e contrasta
    if config.BG_DARKEN_FACTOR < 1.0:
        # Applica lo scurimento in modo più "morbido"
        final_bg = cv2.addWeighted(final_bg, config.BG_DARKEN_FACTOR, np.zeros_like(final_bg), 1 - config.BG_DARKEN_FACTOR, 0)
    if config.BG_CONTRAST_FACTOR > 1.0:
        final_bg = cv2.convertScaleAbs(final_bg, alpha=config.BG_CONTRAST_FACTOR, beta=0)

    # 3. Estrae i contorni (bordi) per l'effetto tracciante del logo con soglie ottimizzate
    gray_bg = cv2.cvtColor(final_bg, cv2.COLOR_BGR2GRAY)  # Usa il frame processato
    # Applica un leggero blur per ridurre il rumore prima di Canny
    gray_bg = cv2.GaussianBlur(gray_bg, (3, 3), 0)
    logo_edges = cv2.Canny(gray_bg, config.TRACER_THRESHOLD1, config.TRACER_THRESHOLD2)
    
    # 4. NUOVO: Estrae traccianti separati per lo sfondo con soglie diverse
    bg_edges = None
    if hasattr(config, 'BG_TRACER_ENABLED') and config.BG_TRACER_ENABLED:
        # Usa soglie ottimizzate per catturare i contorni del video di sfondo
        bg_edges = cv2.Canny(gray_bg, config.BG_TRACER_THRESHOLD1, config.BG_TRACER_THRESHOLD2)
        # Dilata leggermente per renderli più visibili e organici
        kernel = np.ones((2,2), np.uint8)
        bg_edges = cv2.dilate(bg_edges, kernel, iterations=1)
    
    return final_bg, logo_edges, bg_edges

def render_frame(contours, hierarchy, width, height, frame_index, total_frames, config, bg_frame, texture_image, tracer_history, bg_tracer_history, lenses, audio_data=None):
    """
    Rende un singolo frame dell'animazione, applicando la pipeline di effetti completa.
    """
    # --- 0. Ottieni Parametri Dinamici ---
    dynamic_params = get_dynamic_parameters(frame_index, total_frames)
    
    # --- 0.5. Calcola Fattori Audio-Reattivi ---
    audio_factors = get_audio_reactive_factors(audio_data, frame_index, config)

    # --- 1. Preparazione Sfondo e Traccianti ---
    bg_result = process_background(bg_frame, config)
    if len(bg_result) == 3:
        final_frame, current_logo_edges, current_bg_edges = bg_result
    else:
        final_frame, current_logo_edges = bg_result
        current_bg_edges = None
    
    # --- 2. Creazione Layer Traccianti del Logo (CON PARAMETRI DINAMICI) ---
    if config.TRACER_ENABLED and len(tracer_history) > 0:
        tracer_layer = np.zeros_like(final_frame, dtype=np.float32)
        # Applica moltiplicatore dinamico all'opacità
        dynamic_opacity = config.TRACER_MAX_OPACITY * dynamic_params.get('tracer_opacity_multiplier', 1.0)
        opacities = np.linspace(0, dynamic_opacity, len(tracer_history))
        
        for i, past_edges in enumerate(reversed(tracer_history)):
            # --- NUOVO: Colore dinamico per i traccianti ---
            hue_shift = (frame_index * 0.1 + i * 0.5) % 180
            base_color_hsv = cv2.cvtColor(np.uint8([[config.TRACER_BASE_COLOR]]), cv2.COLOR_BGR2HSV)[0][0]
            new_hue = (base_color_hsv[0] + hue_shift) % 180
            dynamic_color_hsv = np.uint8([[[new_hue, base_color_hsv[1], base_color_hsv[2]]]])
            dynamic_color_bgr = cv2.cvtColor(dynamic_color_hsv, cv2.COLOR_HSV2BGR)[0][0]
            
            # Colora i bordi e applica l'opacità dinamica
            colored_tracer = cv2.cvtColor(past_edges, cv2.COLOR_GRAY2BGR).astype(np.float32)
            colored_tracer[past_edges > 0] = np.array(dynamic_color_bgr, dtype=np.float32)
            tracer_with_opacity = cv2.multiply(colored_tracer, opacities[i])
            tracer_layer = cv2.add(tracer_layer, tracer_with_opacity)
            
        final_frame = cv2.add(final_frame.astype(np.float32), tracer_layer)
        final_frame = np.clip(final_frame, 0, 255).astype(np.uint8)

    # --- 2.5. NUOVO: Creazione Layer Traccianti Sfondo (CON PARAMETRI DINAMICI) ---
    if hasattr(config, 'BG_TRACER_ENABLED') and config.BG_TRACER_ENABLED and len(bg_tracer_history) > 0:
        bg_tracer_layer = np.zeros_like(final_frame, dtype=np.float32)
        # Applica moltiplicatore dinamico all'opacità dello sfondo
        dynamic_bg_opacity = config.BG_TRACER_MAX_OPACITY * dynamic_params.get('bg_tracer_opacity_multiplier', 1.0)
        bg_opacities = np.linspace(0, dynamic_bg_opacity, len(bg_tracer_history))
        
        for i, past_bg_edges in enumerate(reversed(bg_tracer_history)):
            # Colore dinamico per traccianti sfondo (diverso dal logo)
            hue_shift_bg = (frame_index * 0.05 + i * 0.3) % 180  # Velocità diversa
            base_color_hsv_bg = cv2.cvtColor(np.uint8([[config.BG_TRACER_BASE_COLOR]]), cv2.COLOR_BGR2HSV)[0][0]
            new_hue_bg = (base_color_hsv_bg[0] + hue_shift_bg) % 180
            dynamic_color_hsv_bg = np.uint8([[[new_hue_bg, base_color_hsv_bg[1], base_color_hsv_bg[2]]]])
            dynamic_color_bgr_bg = cv2.cvtColor(dynamic_color_hsv_bg, cv2.COLOR_HSV2BGR)[0][0]
            
            # Colora i bordi dello sfondo e applica l'opacità dinamica
            colored_bg_tracer = cv2.cvtColor(past_bg_edges, cv2.COLOR_GRAY2BGR).astype(np.float32)
            colored_bg_tracer[past_bg_edges > 0] = np.array(dynamic_color_bgr_bg, dtype=np.float32)
            bg_tracer_with_opacity = cv2.multiply(colored_bg_tracer, bg_opacities[i])
            bg_tracer_layer = cv2.add(bg_tracer_layer, bg_tracer_with_opacity)
            
        final_frame = cv2.add(final_frame.astype(np.float32), bg_tracer_layer)
        final_frame = np.clip(final_frame, 0, 255).astype(np.uint8)

    # --- 3. Creazione Maschera del Logo ---
    logo_mask = create_unified_mask(contours, hierarchy, width, height, config.SMOOTHING_ENABLED, config.SMOOTHING_FACTOR)

    # --- 4. Applica Deformazione Organica (per movimento di base CON AUDIO REATTIVO) ---
    if config.DEFORMATION_ENABLED:
        # Parametri base per il "respiro" costante
        deformation_params = {
            'speed': config.DEFORMATION_SPEED,
            'scale': config.DEFORMATION_SCALE,
            'intensity': config.DEFORMATION_INTENSITY
        }
        
        # Calcola parametri dinamici basati sull'audio per movimento delicato
        dynamic_deformation_params = get_organic_deformation_factors(audio_data, frame_index, config)
        
        logo_mask = apply_organic_deformation(logo_mask, frame_index, deformation_params, dynamic_deformation_params)

    # --- 5. Applica Deformazione a Lenti (sovrapposta alla prima) ---
    if config.LENS_DEFORMATION_ENABLED:
        logo_mask = apply_lens_deformation(logo_mask, lenses, frame_index, config, dynamic_params, audio_factors)

    # --- 5.5. Estrai Traccianti del Logo (NUOVO per maggiore aderenza) ---
    logo_tracers = extract_logo_tracers(logo_mask, config)
    # Combina i traccianti del logo con quelli dello sfondo per un effetto più ricco
    combined_logo_edges = cv2.add(current_logo_edges, logo_tracers)

    # --- 6. Applicazione Texture Dinamica (NUOVO SISTEMA) ---
    # Applica texture secondo la modalità configurata PRIMA di creare i layer del logo
    if config.TEXTURE_ENABLED and texture_image is not None:
        if config.TEXTURE_TARGET in ['background', 'both']:
            # Applica texture allo sfondo
            
            final_frame = apply_texture_blending(
                final_frame, 
                texture_image, 
                config.TEXTURE_BACKGROUND_ALPHA, 
                config.TEXTURE_BLENDING_MODE
            )
    
    # --- 7. Creazione Layer Logo e Glow ---
    logo_layer = np.zeros_like(final_frame)
    glow_layer = np.zeros_like(final_frame)

    # Applica texture al logo (se configurato)
    if config.TEXTURE_ENABLED and texture_image is not None and config.TEXTURE_TARGET in ['logo', 'both']:        
        # Crea base di colore solido
        solid_color_layer = np.zeros_like(final_frame)
        solid_color_layer[logo_mask > 0] = config.LOGO_COLOR
        
        # Applica texture usando il nuovo sistema di blending
        logo_layer = apply_texture_blending(
            solid_color_layer,
            texture_image,
            config.TEXTURE_ALPHA,
            config.TEXTURE_BLENDING_MODE,
            logo_mask
        )
    else:
        # Usa colore solido se la texture è disabilitata o non per il logo
        logo_layer[logo_mask > 0] = config.LOGO_COLOR

    # Applica l'effetto Glow (se abilitato)
    if config.GLOW_ENABLED:
        ksize = config.GLOW_KERNEL_SIZE if config.GLOW_KERNEL_SIZE % 2 != 0 else config.GLOW_KERNEL_SIZE + 1
        blurred_mask = cv2.GaussianBlur(logo_mask, (ksize, ksize), 0)
        glow_mask_3ch = cv2.cvtColor(blurred_mask, cv2.COLOR_GRAY2BGR)
        glow_effect = cv2.multiply(glow_mask_3ch, np.array(config.LOGO_COLOR, dtype=np.float32) / 255.0, dtype=cv2.CV_32F)
        glow_layer = np.clip(glow_effect * dynamic_params['glow_intensity'], 0, 255).astype(np.uint8)

    # --- 6. Composizione Finale con BLENDING AVANZATO SCRITTA-SFONDO ---
    
    # A. Aggiungi il glow allo sfondo in modo additivo
    final_frame_with_glow = cv2.add(final_frame, glow_layer)

    # B. Crea una versione "pulita" del logo (senza glow)
    final_logo_layer = np.zeros_like(final_frame)
    
    # Crea una maschera booleana per un'applicazione precisa
    logo_mask_bool = logo_mask > 0
    
    # Applica il logo (texturizzato o a colore solido) alla sua area
    final_logo_layer[logo_mask_bool] = logo_layer[logo_mask_bool]

    # C. NUOVO: Applica il Blending Avanzato se abilitato
    if config.ADVANCED_BLENDING:
        final_frame = apply_advanced_blending(final_frame_with_glow, final_logo_layer, logo_mask, config)
    else:
        # Metodo tradizionale: sovrapponi il logo pulito allo sfondo con glow
        final_frame_with_glow[logo_mask_bool] = 0
        final_frame = cv2.add(final_frame_with_glow, final_logo_layer)

    return final_frame, combined_logo_edges, current_bg_edges



def extract_logo_tracers(logo_mask, config):
    """
    Estrae i contorni dal logo stesso per creare traccianti più aderenti.
    """
    # Estrae i bordi della maschera del logo
    logo_edges = cv2.Canny(logo_mask, 50, 150)
    
    # Dilata leggermente i bordi per renderli più visibili
    kernel = np.ones((2,2), np.uint8)
    logo_edges = cv2.dilate(logo_edges, kernel, iterations=1)
    
    return logo_edges



def get_background_frame(bg_video, frame_index, bg_start_frame=0):
    """Funzione helper per ottenere un frame di sfondo con offset casuale"""
    if bg_video and bg_video.isOpened():
        # Calcola il frame considerando il rallentamento e l'offset casuale
        bg_frame_index = int(frame_index / Config.BG_SLOWDOWN_FACTOR) + bg_start_frame
        bg_video.set(cv2.CAP_PROP_POS_FRAMES, bg_frame_index)
        ret, bg_frame = bg_video.read()
        
        if ret:
            return bg_frame
    
    # Fallback: frame nero
    return np.zeros((Config.HEIGHT, Config.WIDTH, 3), dtype=np.uint8)



def setup_config_defaults():
    """Imposta i valori di default per la configurazione"""
    # Modalità e Qualità
    Config.TEST_MODE = False
    Config.PREVIEW_MODE = False
    
    # Formato Video
    Config.VIDEO_FORMAT = "INPUT_VIDEO_SIZE"  # "IG_STORY", "IG_POST", "INPUT_VIDEO_SIZE"
    
    # Compatibilità WhatsApp
    Config.WHATSAPP_COMPATIBLE = True
    Config.CREATE_WHATSAPP_VERSION = True
    
    # Sorgente Logo e Texture
    Config.USE_SVG_SOURCE = True
    Config.SVG_PATH = 'input/logo.svg'
    Config.PDF_PATH = 'input/logo.pdf'
    Config.SVG_LEFT_PADDING = 50
    Config.TEXTURE_AUTO_SEARCH = True
    Config.TEXTURE_FALLBACK_PATH = 'input/texture.jpg'
    
    # Sistema Texture Avanzato
    Config.TEXTURE_ENABLED = True
    Config.TEXTURE_TARGET = 'logo'
    Config.TEXTURE_ALPHA = 0.6
    Config.TEXTURE_BACKGROUND_ALPHA = 0.1
    Config.TEXTURE_BLENDING_MODE = 'lighten'
    
    # Parametri Video
    Config.SVG_PADDING = 20
    Config.FPS = 20
    Config.DURATION_SECONDS = 10
    Config.TOTAL_FRAMES = Config.DURATION_SECONDS * Config.FPS
    
    # Colore e Stile
    Config.LOGO_COLOR = (255, 255, 255)
    Config.LOGO_ALPHA = 0.7
    Config.LOGO_ZOOM_FACTOR = 1.0
    
    # Video di Sfondo
    Config.BACKGROUND_VIDEO_PATH = 'input/sfondo.MOV'
    Config.BG_USE_ORIGINAL_SIZE = True
    Config.BG_ZOOM_FACTOR = 1.4
    Config.BG_SLOWDOWN_FACTOR = 1.0
    Config.BG_DARKEN_FACTOR = 0.7
    Config.BG_CONTRAST_FACTOR = 1.0
    Config.BG_RANDOM_START = True
    
    # Parametri Crop Video Verticale
    Config.BG_CROP_Y_START = 0.0
    Config.BG_CROP_X_START = 0.0
    Config.BG_CROP_WIDTH_RATIO = 1.0
    Config.BG_CROP_HEIGHT_RATIO = 1.0
    
    # Sistema Audio Reattivo
    Config.AUDIO_ENABLED = True
    Config.AUDIO_FILES = ['input/audio1.aif', 'input/audio2.aif']
    Config.AUDIO_RANDOM_SELECTION = True
    Config.AUDIO_RANDOM_START = True
    Config.AUDIO_REACTIVE_LENSES = True
    Config.AUDIO_BASS_SENSITIVITY = 0.5
    Config.AUDIO_MID_SENSITIVITY = 0.3
    Config.AUDIO_HIGH_SENSITIVITY = 0.25
    Config.AUDIO_SMOOTHING = 0.5
    Config.AUDIO_BOOST_FACTOR = 4.0
    
    # Parametri Audio Lenti
    Config.AUDIO_SPEED_INFLUENCE = 1.0
    Config.AUDIO_STRENGTH_INFLUENCE = 2
    Config.AUDIO_PULSATION_INFLUENCE = 1.3
    
    # Effetto Glow
    Config.GLOW_ENABLED = True
    Config.GLOW_KERNEL_SIZE = 30
    Config.GLOW_INTENSITY = 0.5
    
    # Altri parametri con valori di default
    Config.DEFORMATION_ENABLED = True
    Config.DEFORMATION_SPEED = 0.01
    Config.DEFORMATION_SCALE = 0.002
    Config.DEFORMATION_INTENSITY = 10.0
    Config.DEFORMATION_AUDIO_REACTIVE = True
    Config.DEFORMATION_BASS_INTENSITY = 0.22
    Config.DEFORMATION_BASS_SPEED = 0.03
    Config.DEFORMATION_MID_SCALE = 0.002
    Config.DEFORMATION_SMOOTHING = 0.85
    Config.DEFORMATION_AUDIO_MULTIPLIER = 1.4
    
    Config.LENS_DEFORMATION_ENABLED = True
    Config.NUM_LENSES = 50
    Config.LENS_MIN_STRENGTH = -1.2
    Config.LENS_MAX_STRENGTH = 1.5
    Config.LENS_MIN_RADIUS = 5
    Config.LENS_MAX_RADIUS = 35
    Config.LENS_SPEED_FACTOR = 0.1
    Config.LENS_PATH_SPEED_MULTIPLIER = 0.1
    Config.LENS_BASE_SPEED_MULTIPLIER = 0.1
    Config.LENS_ROTATION_SPEED_MULTIPLIER = 0.01
    Config.LENS_INERTIA = 0.95
    Config.LENS_ROTATION_SPEED_MIN = -0.02
    Config.LENS_ROTATION_SPEED_MAX = 0.02
    Config.LENS_HORIZONTAL_BIAS = 2
    Config.LENS_PULSATION_ENABLED = True
    Config.LENS_PULSATION_SPEED = 0.0005
    Config.LENS_PULSATION_AMPLITUDE = 0.2
    Config.LENS_FORCE_PULSATION_ENABLED = True
    Config.LENS_FORCE_PULSATION_AMPLITUDE = 0.2
    Config.WORM_SHAPE_ENABLED = True
    Config.WORM_LENGTH = 1.8
    Config.WORM_COMPLEXITY = 5
    
    Config.SMOOTHING_ENABLED = True
    Config.SMOOTHING_FACTOR = 0.0001
    
    Config.TRACER_ENABLED = True
    Config.TRACER_TRAIL_LENGTH = 45
    Config.TRACER_MAX_OPACITY = 0.01
    Config.TRACER_BASE_COLOR = (255, 200, 220)
    Config.TRACER_THRESHOLD1 = 50
    Config.TRACER_THRESHOLD2 = 200
    
    Config.BG_TRACER_ENABLED = True
    Config.BG_TRACER_TRAIL_LENGTH = 45
    Config.BG_TRACER_MAX_OPACITY = 0.01
    Config.BG_TRACER_BASE_COLOR = (200, 170, 200)
    Config.BG_TRACER_THRESHOLD1 = 20
    Config.BG_TRACER_THRESHOLD2 = 100
    
    Config.ADVANCED_BLENDING = True
    Config.BLENDING_PRESET = "cinematic"
    Config.BLENDING_MODE = "color_burn"
    Config.BLENDING_STRENGTH = 0.7
    Config.EDGE_DETECTION_ENABLED = True
    Config.EDGE_BLUR_RADIUS = 21
    Config.ADAPTIVE_BLENDING = False
    Config.COLOR_HARMONIZATION = False
    Config.LUMINANCE_MATCHING = False
    Config.LOGO_BLEND_FACTOR = 0.8
    Config.EDGE_SOFTNESS = 80
    Config.BLEND_TRANSPARENCY = 0.5
    Config.COLOR_BLENDING_STRENGTH = 0.6
    
    Config.DEBUG_MASK = False
    
    Config.DYNAMIC_VARIATION_ENABLED = True
    Config.RANDOM_DEFORMATION_PARAMS = True  # Nuovo attributo per il componente deformazioni
    Config.VARIATION_AMPLITUDE = 0.8
    Config.VARIATION_SPEED_SLOW = 0.01
    Config.VARIATION_SPEED_MEDIUM = 0.025
    Config.VARIATION_SPEED_FAST = 0.005

def load_config_from_file():
    """Carica i parametri dal file config se esiste"""
    # Prima imposta i valori di default
    setup_config_defaults()
    
    config_file = "config"
    if not os.path.exists(config_file):
        print("📄 File config non trovato, uso valori di default")
        return
    
    print("📄 Caricamento parametri dal file config...")
    
    try:
        with open(config_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    try:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        # Separa il valore dal commento
                        if '#' in value:
                            value = value.split('#')[0].strip()
                        else:
                            value = value.strip()
                        
                        # Rimuove le virgolette se presenti
                        value = value.strip('"\'')
                        
                        # Converti il valore nel tipo appropriato e gestisci parametri speciali
                        # Gestione speciale per parametri BGR (prima del controllo hasattr)
                        if key == 'LOGO_COLOR_B':
                            current_color = list(Config.LOGO_COLOR)
                            current_color[0] = int(value)
                            Config.LOGO_COLOR = tuple(current_color)
                        elif key == 'LOGO_COLOR_G':
                            current_color = list(Config.LOGO_COLOR)
                            current_color[1] = int(value)
                            Config.LOGO_COLOR = tuple(current_color)
                        elif key == 'LOGO_COLOR_R':
                            current_color = list(Config.LOGO_COLOR)
                            current_color[2] = int(value)
                            Config.LOGO_COLOR = tuple(current_color)
                        elif key == 'TRACER_BASE_COLOR_B':
                            current_color = list(Config.TRACER_BASE_COLOR)
                            current_color[0] = int(value)
                            Config.TRACER_BASE_COLOR = tuple(current_color)
                        elif key == 'TRACER_BASE_COLOR_G':
                            current_color = list(Config.TRACER_BASE_COLOR)
                            current_color[1] = int(value)
                            Config.TRACER_BASE_COLOR = tuple(current_color)
                        elif key == 'TRACER_BASE_COLOR_R':
                            current_color = list(Config.TRACER_BASE_COLOR)
                            current_color[2] = int(value)
                            Config.TRACER_BASE_COLOR = tuple(current_color)
                        elif key == 'BG_TRACER_BASE_COLOR_B':
                            current_color = list(Config.BG_TRACER_BASE_COLOR)
                            current_color[0] = int(value)
                            Config.BG_TRACER_BASE_COLOR = tuple(current_color)
                        elif key == 'BG_TRACER_BASE_COLOR_G':
                            current_color = list(Config.BG_TRACER_BASE_COLOR)
                            current_color[1] = int(value)
                            Config.BG_TRACER_BASE_COLOR = tuple(current_color)
                        elif key == 'BG_TRACER_BASE_COLOR_R':
                            current_color = list(Config.BG_TRACER_BASE_COLOR)
                            current_color[2] = int(value)
                            Config.BG_TRACER_BASE_COLOR = tuple(current_color)
                        elif key == 'AUDIO_FILES':
                            if ',' in value:
                                Config.AUDIO_FILES = [item.strip() for item in value.split(',')]
                            else:
                                Config.AUDIO_FILES = [value]
                        elif hasattr(Config, key):
                            current_value = getattr(Config, key)
                            
                            # Converti in base al tipo dell'attributo esistente
                            if isinstance(current_value, bool):
                                new_value = value.lower() in ('true', '1', 'yes', 'on')
                            elif isinstance(current_value, int):
                                new_value = int(value)
                            elif isinstance(current_value, float):
                                new_value = float(value)
                            elif isinstance(current_value, str):
                                new_value = value
                            elif isinstance(current_value, tuple):
                                # Per i colori BGR
                                if key.endswith('_COLOR_B') or key.endswith('_COLOR_G') or key.endswith('_COLOR_R'):
                                    current_color = list(getattr(Config, key.rsplit('_', 1)[0]))
                                    if key.endswith('_B'):
                                        current_color[0] = int(value)
                                    elif key.endswith('_G'):
                                        current_color[1] = int(value)
                                    elif key.endswith('_R'):
                                        current_color[2] = int(value)
                                    setattr(Config, key.rsplit('_', 1)[0], tuple(current_color))
                                    continue
                            elif isinstance(current_value, list):
                                # Per liste di file audio
                                if ',' in value:
                                    new_value = [item.strip() for item in value.split(',')]
                                else:
                                    new_value = [value]
                            else:
                                new_value = value
                            
                            # Imposta il valore normalmente per parametri standard
                            setattr(Config, key, new_value)
                        else:
                            print(f"⚠️  Parametro sconosciuto '{key}' alla riga {line_num}")
                    except Exception as e:
                        print(f"⚠️  Errore nel parsing della riga {line_num}: {line} ({e})")
        
        # Ricalcola i valori dipendenti
        if Config.TEST_MODE:
            Config.FPS = 1
            Config.DURATION_SECONDS = 4
        Config.TOTAL_FRAMES = Config.DURATION_SECONDS * Config.FPS
        
        print("✅ Configurazione caricata dal file config")
    
    except Exception as e:
        print(f"⚠️  Errore nel caricamento del file config: {e}")
        print("📄 Uso valori di default")

def main():
    """Funzione principale per generare l'animazione del logo."""
    import os  # Assicuriamoci che os sia disponibile
    import sys  # Assicuriamoci che sys sia disponibile
    
    # --- Parsing degli argomenti da linea di comando ---
    parser = argparse.ArgumentParser(description='Crystal Therapy Video Generator')
    parser.add_argument('--preview', action='store_true', 
                       help='Avvia modalità Live Preview')
    parser.add_argument('--test', action='store_true',
                       help='Modalità test rapida (5 secondi)')
    args = parser.parse_args()
    
    # --- Carica configurazione dal file config ---
    load_config_from_file()
    
    # Applica le opzioni dalla linea di comando (override del config file)
    if args.test:
        Config.TEST_MODE = True
        Config.FPS = 1
        Config.DURATION_SECONDS = 4
        Config.TOTAL_FRAMES = Config.DURATION_SECONDS * Config.FPS
    
    if args.preview:
        Config.PREVIEW_MODE = True
        print("🌊 Modalità LIVE PREVIEW attivata!")
    
    # --- Codici ANSI per colori e stili nel terminale ---
    C_CYAN = '\033[96m'
    C_GREEN = '\033[92m'
    C_YELLOW = '\033[93m'
    C_BLUE = '\033[94m'
    C_MAGENTA = '\033[95m'
    C_RED = '\033[91m'  # Aggiungo colore rosso
    C_BOLD = '\033[1m'
    C_END = '\033[0m'
    SPINNER_CHARS = ['🔮', '✨', '🌟', '💎']
    
    # Mostra le opzioni di blending disponibili
    print_blending_options()
    
    # Assicurati che la cartella test esista se siamo in TEST_MODE
    if Config.TEST_MODE:
        test_dir = "output/test"
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
            print(f"📁 Creata cartella: {test_dir}")

    # 🎨 APPLICA PRESET BLENDING AUTOMATICO
    apply_blending_preset(Config)

    # NUOVO: Calcola dimensioni del video dalle dimensioni SVG + padding
    svg_width, svg_height = get_svg_dimensions(Config.SVG_PATH)

    # 📱 GESTIONE FORMATO VIDEO
    if Config.VIDEO_FORMAT == "IG_STORY":
        if Config.TEST_MODE:
            # Versione ridotta per test: 540x960 (metà di 1080x1920)
            Config.WIDTH = 540
            Config.HEIGHT = 960
        else:
            # Formato Instagram Stories standard: 1080x1920
            Config.WIDTH = 1080
            Config.HEIGHT = 1920
        format_info = "Instagram Stories (9:16)"
    elif Config.VIDEO_FORMAT == "IG_POST":
        if Config.TEST_MODE:
            # Versione ridotta per test: 540x540 (metà di 1080x1080)
            Config.WIDTH = 540
            Config.HEIGHT = 540
        else:
            # Formato Instagram Post standard: 1080x1080
            Config.WIDTH = 1080
            Config.HEIGHT = 1080
        format_info = "Instagram Post (1:1)"
    else:  # INPUT_VIDEO_SIZE
        # Formato tradizionale basato su dimensioni SVG
        Config.WIDTH = svg_width + (Config.SVG_PADDING * 2)
        Config.HEIGHT = svg_height + (Config.SVG_PADDING * 2)
        format_info = "Input Video Size"
    
    print(f"{C_BOLD}{C_CYAN}🌊 Avvio rendering Crystal Therapy - SVG CENTRATO...{C_END}")
    print(f"📐 Dimensioni SVG: {svg_width}x{svg_height}")
    print(f"📐 Dimensioni video: {Config.WIDTH}x{Config.HEIGHT} (formato: {format_info})")
    if Config.VIDEO_FORMAT == "IG_STORY" and not Config.TEST_MODE:
        print(f"📱 INSTAGRAM STORIES: Formato verticale ottimizzato per mobile")
    elif Config.VIDEO_FORMAT == "IG_POST" and not Config.TEST_MODE:
        print(f"📱 INSTAGRAM POST: Formato quadrato ottimizzato per feed")
    if Config.SVG_PADDING and Config.VIDEO_FORMAT == "INPUT_VIDEO_SIZE":
        print(f"🎨 Padding SVG: {Config.SVG_PADDING}px")
    if Config.TEST_MODE:
        print(f"🎬 TEST MODE: 10fps, {Config.DURATION_SECONDS}s, risoluzione ridotta per velocità")
    else:
        print(f"🎬 PRODUZIONE: 30fps, {Config.DURATION_SECONDS}s, risoluzione completa")
    source_type = "SVG vettoriale" if Config.USE_SVG_SOURCE else "PDF rasterizzato"
    print(f"📄 Sorgente: {source_type} con smoothing ottimizzato")
    print(f"🎥 Video sfondo: ORIGINALE senza crop, rallentato {Config.BG_SLOWDOWN_FACTOR}x")
    print(f"✨ Traccianti + Blending + Glow COMPATIBILE")
    print(f"� Variazione dinamica + codec video testati")
    print(f"💎 RENDERING MOVIMENTO GARANTITO per compatibilità VLC/QuickTime!")
    
    # Carica contorni da SVG o PDF
    if Config.USE_SVG_SOURCE:
        if Config.VIDEO_FORMAT == "IG_STORY":
            # Per Instagram Stories, centra il logo nel formato verticale con spostamento a destra
            horizontal_margin = (Config.WIDTH - svg_width) // 2
            # Riduci un po' il margine sinistro per spostare il logo leggermente a destra
            right_shift = 10 if Config.TEST_MODE else 20
            effective_padding = max(Config.SVG_PADDING, horizontal_margin - right_shift)
            contours, hierarchy = extract_contours_from_svg(Config.SVG_PATH, Config.WIDTH, Config.HEIGHT, effective_padding, Config.SVG_LEFT_PADDING, Config.LOGO_ZOOM_FACTOR)
        elif Config.VIDEO_FORMAT == "IG_POST":
            # Per Instagram Post, centra il logo nel formato quadrato
            horizontal_margin = (Config.WIDTH - svg_width) // 2
            vertical_margin = (Config.HEIGHT - svg_height) // 2
            effective_padding = max(Config.SVG_PADDING, min(horizontal_margin, vertical_margin))
            contours, hierarchy = extract_contours_from_svg(Config.SVG_PATH, Config.WIDTH, Config.HEIGHT, effective_padding, Config.SVG_LEFT_PADDING, Config.LOGO_ZOOM_FACTOR)
        else:  # INPUT_VIDEO_SIZE
            contours, hierarchy = extract_contours_from_svg(Config.SVG_PATH, Config.WIDTH, Config.HEIGHT, Config.SVG_PADDING, Config.SVG_LEFT_PADDING, Config.LOGO_ZOOM_FACTOR)
    else:
        if Config.VIDEO_FORMAT == "IG_STORY":
            # Per Instagram Stories, centra il logo nel formato verticale con spostamento a destra
            horizontal_margin = (Config.WIDTH - svg_width) // 2
            # Riduci un po' il margine sinistro per spostare il logo leggermente a destra
            right_shift = 10 if Config.TEST_MODE else 20
            effective_padding = max(Config.SVG_PADDING, horizontal_margin - right_shift)
            contours, hierarchy = extract_contours_from_pdf(Config.PDF_PATH, Config.WIDTH, Config.HEIGHT, effective_padding, Config.LOGO_ZOOM_FACTOR)
        elif Config.VIDEO_FORMAT == "IG_POST":
            # Per Instagram Post, centra il logo nel formato quadrato
            horizontal_margin = (Config.WIDTH - svg_width) // 2
            vertical_margin = (Config.HEIGHT - svg_height) // 2
            effective_padding = max(Config.SVG_PADDING, min(horizontal_margin, vertical_margin))
            contours, hierarchy = extract_contours_from_pdf(Config.PDF_PATH, Config.WIDTH, Config.HEIGHT, effective_padding, Config.LOGO_ZOOM_FACTOR)
        else:
            contours, hierarchy = extract_contours_from_pdf(Config.PDF_PATH, Config.WIDTH, Config.HEIGHT, Config.SVG_PADDING, Config.LOGO_ZOOM_FACTOR)

    if not contours:
        source_name = "SVG" if Config.USE_SVG_SOURCE else "PDF"
        print(f"Errore critico: nessun contorno valido trovato nel {source_name}. Uscita.")
        return

    print("Estrazione contorni riuscita.")

    # --- MODALITÀ LIVE PREVIEW ---
    if Config.PREVIEW_MODE:
        print("🌊 Avviando modalità Live Preview...")
        
        # Avvia la preview
        result = run_preview_mode(
            Config, render_frame, contours, hierarchy, Config.WIDTH, Config.HEIGHT,
            get_background_frame, load_texture_wrapper, initialize_lenses, load_audio_wrapper
        )
        
        if result == 'RESTART_SCRIPT':
            print("🔄 RESTART COMPLETO RICHIESTO - Rilanciando script...")
            import sys
            import os
            # Rilancia lo script con gli stessi parametri
            os.execv(sys.executable, [sys.executable] + sys.argv)
        elif result == 'FULL_VIDEO':
            print("🎬 Utente ha richiesto generazione video completo!")
            print("🚀 Passaggio a modalità produzione...")
            # Disabilita preview mode e continua con il rendering normale
            Config.PREVIEW_MODE = False
        elif result == 'TEST_MODE':
            print("⚡ Utente ha richiesto generazione video TEST mode!")
            print("🚀 Passaggio a modalità test temporanea...")
            # Disabilita preview mode e applica temporaneamente test mode
            Config.PREVIEW_MODE = False
            # Salva i valori originali per ripristinarli dopo
            original_test_mode = Config.TEST_MODE
            original_duration = Config.DURATION_SECONDS
            original_fps = Config.FPS
            # Applica temporaneamente le impostazioni di test
            Config.TEST_MODE = True
            Config.DURATION_SECONDS = 5  # Durata test
            Config.FPS = 20  # FPS test
            print(f"   📝 TEST_MODE temporaneo: durata {Config.DURATION_SECONDS}s, fps {Config.FPS}")
            
            # Dopo il rendering, ripristina i valori originali
            def restore_original_settings():
                Config.TEST_MODE = original_test_mode
                Config.DURATION_SECONDS = original_duration 
                Config.FPS = original_fps
                print("   🔄 Impostazioni originali ripristinate")
            
            # Memorizza la funzione di ripristino per dopo il rendering
            Config._restore_settings = restore_original_settings
        else:
            print("👋 Uscita dalla Live Preview")
            return

    # --- Caricamento Texture (se abilitata) ---
    texture_image = None
    if Config.TEXTURE_ENABLED:
        # Prima cerca la texture automaticamente
        texture_path = find_texture_file()
        # Poi carica la texture trovata (o fallback se non trovata)
        texture_image = load_texture(texture_path, Config.WIDTH, Config.HEIGHT)
        if texture_image is not None:
            print("Texture infusa con l'essenza del Natisone - Creata dal team Alex Ortiga, TV Int, Iaia & Friend.")
    else:
        print("La texturizzazione del logo è disabilitata.")

    # --- Apertura Video di Sfondo ---
    bg_video = cv2.VideoCapture(Config.BACKGROUND_VIDEO_PATH)
    if not bg_video.isOpened():
        print(f"Errore: impossibile aprire il video di sfondo in {Config.BACKGROUND_VIDEO_PATH}")
        # Crea uno sfondo nero di fallback
        bg_video = None
        bg_start_frame = 0
        bg_total_frames = 0  # Aggiungo variabile per fallback
    else:
        # NUOVO: Ottieni informazioni del video di sfondo per il rallentamento
        bg_total_frames = int(bg_video.get(cv2.CAP_PROP_FRAME_COUNT))
        bg_fps = bg_video.get(cv2.CAP_PROP_FPS)
        
        # 🎲 RANDOM START: Calcola frame di inizio casuale (max 2/3 del video)
        bg_start_frame = 0
        if Config.BG_RANDOM_START and bg_total_frames > Config.TOTAL_FRAMES:
            # Calcola quanti frame servono considerando il rallentamento
            frames_needed = int(Config.TOTAL_FRAMES / Config.BG_SLOWDOWN_FACTOR) + 1
            # Assicurati di avere abbastanza frame rimanenti per il rendering
            max_start_frame = max(0, int(bg_total_frames * 2/3) - frames_needed)
            if max_start_frame > 0:
                bg_start_frame = np.random.randint(0, max_start_frame)
                start_time = bg_start_frame / bg_fps
                end_time = start_time + (frames_needed / bg_fps)
                print(f"🎬 Video sfondo: {bg_total_frames} frame @ {bg_fps}fps")
                print(f"🎲 Inizio casuale da frame {bg_start_frame} ({start_time:.1f}s -> {end_time:.1f}s)")
                print(f"📊 Frame necessari: {frames_needed} (con rallentamento {Config.BG_SLOWDOWN_FACTOR}x)")
            else:
                print(f"🎬 Video sfondo: {bg_total_frames} frame @ {bg_fps}fps")
                print(f"⚠️ Video troppo corto per random start")
        else:
            print(f"🎬 Video sfondo: {bg_total_frames} frame @ {bg_fps}fps")
            if not Config.BG_RANDOM_START:
                print(f"🔄 Inizio dal primo frame (random start disabilitato)")
        
        print(f"🐌 RALLENTAMENTO ATTIVATO: Video sfondo {Config.BG_SLOWDOWN_FACTOR}x più lento")
    
    # Setup video writer con codec ottimizzato per WhatsApp
    if Config.WHATSAPP_COMPATIBLE:
        # H.264 è il migliore per WhatsApp
        fourcc = cv2.VideoWriter_fourcc(*'H264')  # Priorità H264 per WhatsApp
        print("🔄 Usando H.264 per compatibilità WhatsApp...")
    else:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Fallback generico
        
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
    
    # --- Inizializzazione Effetti ---
    tracer_history = deque(maxlen=Config.TRACER_TRAIL_LENGTH)
    
    # --- NUOVO: Inizializzazione Traccianti Sfondo ---
    bg_tracer_history = deque(maxlen=getattr(Config, 'BG_TRACER_TRAIL_LENGTH', 35))

    # --- Inizializzazione per Effetto Lenti (NUOVO) ---
    lenses = []
    if Config.LENS_DEFORMATION_ENABLED:
        lenses = initialize_lenses(Config)
        print(f"🌊 Liberate {len(lenses)} creature liquide per Alex Ortiga... texturizzizando con TVInt")

    # --- NUOVO: Caricamento e Analisi Audio ---
    audio_data = None
    if Config.AUDIO_ENABLED:
        audio_data = load_audio_analysis(
            Config.AUDIO_FILES, 
            Config.DURATION_SECONDS, 
            Config.FPS,
            Config.AUDIO_RANDOM_SELECTION,
            Config.AUDIO_RANDOM_START
        )
        if audio_data:
            print(f"🎵 Audio caricato: reattività lenti attivata con {len(lenses)} elementi sincronizzati")
            print(f"📂 File selezionato: {audio_data['selected_file']}")
            if audio_data['start_offset'] > 0:
                print(f"⏯️ Inizio da: {audio_data['start_offset']:.1f}s")
        else:
            if not AUDIO_AVAILABLE:
                print("🔇 Audio non disponibile: installare librosa per abilitare reattività audio")
            else:
                print("⚠️ Nessun file audio trovato: rendering senza sincronizzazione")
    else:
        print("🔇 Audio disabilitato nella configurazione")

    print(f"Rendering dell'animazione in corso... ({Config.TOTAL_FRAMES} frame da elaborare)")
    start_time = time.time()
    
    try:
        for i in range(Config.TOTAL_FRAMES):
            # --- Gestione Frame di Sfondo con RALLENTAMENTO ---
            if bg_video:
                # NUOVO: Calcola il frame del video di sfondo rallentato con offset casuale
                bg_frame_index = bg_start_frame + int(i / Config.BG_SLOWDOWN_FACTOR)
                
                # Controllo di sicurezza: assicurati che il frame sia valido
                if bg_frame_index >= bg_total_frames:
                    # Se superiamo la fine, torna al punto di partenza casuale
                    bg_frame_index = bg_start_frame + (bg_frame_index - bg_start_frame) % (bg_total_frames - bg_start_frame)
                
                # Imposta la posizione nel video di sfondo
                bg_video.set(cv2.CAP_PROP_POS_FRAMES, bg_frame_index)
                ret, bg_frame = bg_video.read()
                
                # Doppio controllo di sicurezza
                if not ret:
                    print(f"⚠️ Errore lettura frame {bg_frame_index}, riavvolgendo...")
                    bg_video.set(cv2.CAP_PROP_POS_FRAMES, bg_start_frame)
                    ret, bg_frame = bg_video.read()
                    if not ret:
                        # Ultima risorsa: crea frame nero
                        bg_frame = np.zeros((Config.HEIGHT, Config.WIDTH, 3), dtype=np.uint8)
                # RIMOSSO: Non ridimensionare qui, lo fa process_background
                # bg_frame = cv2.resize(bg_frame, (Config.WIDTH, Config.HEIGHT))
            else:
                # Crea uno sfondo nero se non c'è video
                bg_frame = np.zeros((Config.HEIGHT, Config.WIDTH, 3), dtype=np.uint8)

            frame_result = render_frame(contours, hierarchy, Config.WIDTH, Config.HEIGHT, i, Config.TOTAL_FRAMES, Config, bg_frame, texture_image, tracer_history, bg_tracer_history, lenses, audio_data)
            
            if len(frame_result) == 3:
                frame, current_logo_edges, current_bg_edges = frame_result
            else:
                frame, current_logo_edges = frame_result
                current_bg_edges = None
            
            # Aggiorna la storia dei traccianti
            if Config.TRACER_ENABLED:
                tracer_history.append(current_logo_edges)
            
            # Aggiorna la storia dei traccianti dello sfondo
            if hasattr(Config, 'BG_TRACER_ENABLED') and Config.BG_TRACER_ENABLED and current_bg_edges is not None:
                bg_tracer_history.append(current_bg_edges)
            
            out.write(frame)
            
            # --- Log di Avanzamento Magico (aggiornamento fluido) ---
            elapsed = time.time() - start_time
            fps = (i + 1) / elapsed if elapsed > 0 else 0
            
            # Calcolo ETA con smoothing
            remaining_frames = Config.TOTAL_FRAMES - (i + 1)
            eta_seconds = remaining_frames / fps if fps > 0 else 0
            eta_minutes, eta_sec = divmod(int(eta_seconds), 60)
            eta_str = f"{eta_minutes:02d}:{eta_sec:02d}"

            # Barra di avanzamento fluida con più dettagli
            progress = (i + 1) / Config.TOTAL_FRAMES
            bar_length = 30  # Barra più lunga per maggiore dettaglio
            filled_length = int(bar_length * progress)
            
            # --- Barra colorata dinamica con gradiente ---
            progress_color_map = [C_MAGENTA, C_BLUE, C_CYAN, C_GREEN, C_YELLOW, C_RED]
            color_index = min(int(progress * len(progress_color_map)), len(progress_color_map) - 1)
            bar_color = progress_color_map[color_index]
            
            # Barra con carattere di riempimento più preciso
            partial_char = ['▏', '▎', '▍', '▌', '▋', '▊', '▉', '█']
            partial_fill = (bar_length * progress) - filled_length
            partial_index = int(partial_fill * len(partial_char))
            partial_symbol = partial_char[min(partial_index, len(partial_char) - 1)] if partial_fill > 0 and filled_length < bar_length else ''
            
            bar = f"{bar_color}{'█' * filled_length}{partial_symbol}{C_END}{'░' * (bar_length - filled_length - (1 if partial_symbol else 0))}"
            
            # Spinner organico più fluido
            spinner_organic = ['🌊', '🌀', '💫', '✨', '🔮', '💎', '⭐', '🌟']
            spinner = spinner_organic[i % len(spinner_organic)]
            
            # Frame rate color coding
            fps_color = C_GREEN if fps >= 15 else C_YELLOW if fps >= 8 else C_RED

            log_message = (
                f"\r{spinner} {C_BOLD}{C_GREEN}Natisone Trip{C_END} "
                f"{C_CYAN}[{bar}]{C_END} {C_BOLD}{progress:.1%}{C_END} "
                f"│ {fps_color}⚡{fps:.1f}fps{C_END} "
                f"│ {C_MAGENTA}⏱️{eta_str}{C_END} "
                f"│ {C_YELLOW}🎬{i+1}/{Config.TOTAL_FRAMES}{C_END}"
            )
            print(log_message, end="", flush=True)  # flush=True per aggiornamento immediato
        
        print(f"\n{C_BOLD}{C_GREEN}🌿 Cristallizzazione ULTRA completata con effetti IPNOTICI!{C_END}")
        print(f"💥 Deformazioni organiche ESAGERATE ma ultra-fluide!")
        print(f"� Traccianti DOPPI (logo rosa + sfondo viola) dinamici!")
        print(f"💎 Qualità SUPREMA (1000 DPI, smoothing perfetto)!")
        print(f"🔮 Movimento IPNOTICO e curioso - Alex Ortiga & TV Int ULTIMATE!")
        
    finally:
        # Assicurati sempre di chiudere correttamente i file video
        out.release()
        if bg_video: 
            bg_video.release()
        
        # --- AGGIUNTA AUDIO AL VIDEO ---
        if audio_data:
            print(f"\n{C_BOLD}{C_CYAN}🎵 Aggiungendo audio al video...{C_END}")
            final_output_filename = add_audio_to_video(output_filename, audio_data, Config.DURATION_SECONDS)
        else:
            final_output_filename = output_filename
            
        if Config.TEST_MODE:
            print(f"🧪 TEST - Animazione salvata in: {C_BOLD}{final_output_filename}{C_END}")
        else:
            print(f"🎬 PRODUZIONE - Animazione salvata in: {C_BOLD}{final_output_filename}{C_END}")

        # --- GESTIONE VERSIONAMENTO ---
        try:
            print(f"\n{C_BLUE}🚀 Avvio gestore di versioni...{C_END}")
            source_script_path = os.path.abspath(__file__)
            # Assicurati che il percorso di version_manager.py sia corretto
            version_manager_path = os.path.join(os.path.dirname(source_script_path), 'components', 'version_manager.py')
            
            if os.path.exists(version_manager_path):
                result = subprocess.run(
                    [sys.executable, version_manager_path, final_output_filename, source_script_path],
                    capture_output=True,
                    text=True,
                    check=False # Mettiamo a False per gestire l'errore manualmente
                )
                # Stampa sempre stdout e stderr per il debug
                print(result.stdout)
                if result.stderr:
                    # Gestisce il caso "nothing to commit" come un'informazione, non un errore
                    if "nothing to commit" in result.stderr.lower():
                         print(f"{C_GREEN}ℹ️ Nessuna nuova modifica da salvare nel versionamento.{C_END}")
                    else:
                        print(f"{C_YELLOW}Output di errore dal gestore versioni:{C_END}\n{result.stderr}")
            else:
                print(f"{C_YELLOW}ATTENZIONE: version_manager.py non trovato. Saltando il versionamento.{C_END}")

        except Exception as e:
            print(f"{C_YELLOW}Errore inatteso durante il versionamento: {e}{C_END}")
    
    # Ripristina le impostazioni originali se erano state modificate per TEST_MODE temporaneo
    if hasattr(Config, '_restore_settings') and Config._restore_settings:
        Config._restore_settings()
        delattr(Config, '_restore_settings')

if __name__ == "__main__":
    main()