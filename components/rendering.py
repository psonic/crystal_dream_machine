"""
🎬 CORE RENDERING ENGINE - Crystal Therapy
Sistema di rendering centrale per la generazione dei frame

Funzionalità:
- Rendering frame-by-frame con pipeline completa
- Gestione deformazioni organiche e lenti
- Sistema di traccianti per logo e sfondo
- Applicazione texture dinamica
- Composizione finale con blending avanzato
- Estrazione bordi e contorni per traccianti
"""

import cv2
import numpy as np
import math
from datetime import datetime
import random

from components.deformations import apply_organic_deformation, get_organic_deformation_params
from components.lenses import apply_lens_deformation
from components.blending import apply_advanced_blending, apply_texture_blending
from components.tracers import apply_logo_tracers, apply_background_tracers, calculate_tracer_dynamic_params, extract_logo_and_bg_tracers
from components.audio import get_audio_reactive_factors, get_organic_deformation_factors


def render_frame(contours, hierarchy, width, height, frame_index, total_frames, config, bg_frame, texture_image, tracer_history, bg_tracer_history, lenses, audio_data=None):
    """
    Rende un singolo frame dell'animazione, applicando la pipeline di effetti completa.
    
    Pipeline di rendering:
    1. Preparazione sfondo e traccianti
    2. Applicazione traccianti logo e sfondo
    3. Creazione maschera del logo
    4. Deformazione organica (con audio reattivo)
    5. Deformazione a lenti
    6. Estrazione traccianti del logo
    7. Applicazione texture dinamica
    8. Creazione layer logo e glow
    9. Composizione finale con blending avanzato
    
    Args:
        contours: Contorni del logo da processare
        hierarchy: Gerarchia dei contorni
        width, height: Dimensioni del frame
        frame_index: Indice del frame corrente
        total_frames: Numero totale di frame
        config: Oggetto configurazione
        bg_frame: Frame di sfondo
        texture_image: Immagine texture (opzionale)
        tracer_history: Storia traccianti logo
        bg_tracer_history: Storia traccianti sfondo
        lenses: Array delle lenti per deformazione
        audio_data: Dati audio per reattività (opzionale)
        
    Returns:
        tuple: (final_frame, combined_logo_edges, current_bg_edges)
    """
    # --- 0. Ottieni Parametri Dinamici ---
    dynamic_params = get_dynamic_parameters(frame_index, total_frames, config)
    
    # --- 0.5. Calcola Fattori Audio-Reattivi ---
    audio_factors = get_audio_reactive_factors(audio_data, frame_index, config)

    # --- 1. Preparazione Sfondo e Traccianti ---
    bg_result = process_background(bg_frame, config)
    if len(bg_result) == 3:
        final_frame, current_logo_edges, current_bg_edges = bg_result
    else:
        final_frame, current_logo_edges = bg_result
        current_bg_edges = None
    
    # --- 2. Applicazione Traccianti del Logo ---
    final_frame = apply_logo_tracers(final_frame, tracer_history, frame_index, config, dynamic_params)

    # --- 2.5. Applicazione Traccianti Sfondo ---
    final_frame = apply_background_tracers(final_frame, bg_tracer_history, frame_index, config, dynamic_params)

    # --- 3. Creazione Maschera del Logo ---
    from components.svg_pdf import create_unified_mask
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

    # --- 8. Composizione Finale con BLENDING AVANZATO SCRITTA-SFONDO ---
    
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
    
    Questa funzione analizza la maschera del logo per estrarre i bordi
    e creare traccianti che seguono più fedelmente la forma del logo.
    
    Args:
        logo_mask: Maschera binaria del logo
        config: Oggetto configurazione con parametri traccianti
        
    Returns:
        numpy.ndarray: Immagine con i bordi estratti per traccianti
    """
    # Estrae i bordi della maschera del logo
    logo_edges = cv2.Canny(logo_mask, 50, 150)
    
    # Dilata leggermente i bordi per renderli più visibili
    kernel = np.ones((2,2), np.uint8)
    logo_edges = cv2.dilate(logo_edges, kernel, iterations=1)
    
    return logo_edges


def get_dynamic_parameters(frame_index, total_frames, config):
    """
    Calcola parametri dinamici per animazioni basate sul frame corrente.
    
    Genera valori che cambiano nel tempo per creare animazioni fluide
    come pulsazioni, oscillazioni e variazioni di intensità.
    
    Args:
        frame_index: Indice del frame corrente
        total_frames: Numero totale di frame
        config: Oggetto configurazione
        
    Returns:
        dict: Dizionario con parametri dinamici calcolati
    """
    # Progresso normalizzato (0.0 - 1.0)
    progress = frame_index / total_frames
    
    # Tempo normalizzato per funzioni trigonometriche
    time = progress * 2 * math.pi
    
    # Pulsazione principale (respiro lento)
    main_pulse = 0.5 + 0.3 * math.sin(time * 2)
    
    # Pulsazione secondaria (battito più veloce)
    secondary_pulse = 0.5 + 0.2 * math.sin(time * 8)
    
    # Oscillazione per variazioni delicate
    oscillation = 0.5 + 0.1 * math.sin(time * 3)
    
    # Intensità glow che varia nel tempo
    glow_intensity = config.GLOW_INTENSITY * (0.7 + 0.3 * math.sin(time * 1.5)) if hasattr(config, 'GLOW_INTENSITY') else 0.5
    
    # Fattore di variazione generale
    variation_factor = 1.0
    if hasattr(config, 'DYNAMIC_VARIATION_ENABLED') and config.DYNAMIC_VARIATION_ENABLED:
        variation_factor = 0.8 + 0.4 * math.sin(time * config.VARIATION_SPEED_SLOW * 10)
    
    # Parametri per lenti e deformazioni
    lens_speed_factor = getattr(config, 'LENS_SPEED_FACTOR', 0.1)
    lens_strength_multiplier = 1.0 + 0.2 * math.sin(time * 4)
    
    # Parametri tracers dinamici
    tracer_opacity_multiplier = 0.8 + 0.4 * oscillation
    tracer_trail_variation = 0.9 + 0.2 * math.sin(time * 6)
    
    return {
        'progress': progress,
        'time': time,
        'main_pulse': main_pulse,
        'secondary_pulse': secondary_pulse,
        'oscillation': oscillation,
        'glow_intensity': glow_intensity,
        'variation_factor': variation_factor,
        'lens_speed_factor': lens_speed_factor,
        'lens_strength_multiplier': lens_strength_multiplier,
        'tracer_opacity_multiplier': tracer_opacity_multiplier,
        'tracer_trail_variation': tracer_trail_variation,
        'deformation_speed': getattr(config, 'DEFORMATION_SPEED', 0.02) * variation_factor,
        'deformation_scale': getattr(config, 'DEFORMATION_SCALE', 0.005) * variation_factor,
        'deformation_intensity': getattr(config, 'DEFORMATION_INTENSITY', 15.0) * main_pulse
    }


def process_background(bg_frame, config):
    """
    Processa il frame di sfondo applicando effetti e estraendo traccianti.
    
    Applica darken, contrast e altri effetti al frame di sfondo,
    ed estrae i bordi per i sistemi di traccianti.
    
    Args:
        bg_frame: Frame di sfondo da processare
        config: Oggetto configurazione
        
    Returns:
        tuple: (processed_frame, logo_edges, bg_edges) o (processed_frame, logo_edges)
    """
    if bg_frame is None:
        # Crea un frame nero se non c'è sfondo
        height, width = config.HEIGHT, config.WIDTH
        processed_frame = np.zeros((height, width, 3), dtype=np.uint8)
        logo_edges = np.zeros((height, width), dtype=np.uint8)
        return processed_frame, logo_edges
    
    # Copia il frame per evitare modifiche all'originale
    processed_frame = bg_frame.copy()
    
    # Applica darken factor
    if hasattr(config, 'BG_DARKEN_FACTOR') and config.BG_DARKEN_FACTOR != 1.0:
        processed_frame = cv2.multiply(processed_frame, config.BG_DARKEN_FACTOR)
    
    # Applica contrast factor
    if hasattr(config, 'BG_CONTRAST_FACTOR') and config.BG_CONTRAST_FACTOR != 1.0:
        processed_frame = cv2.multiply(processed_frame, config.BG_CONTRAST_FACTOR)
        processed_frame = np.clip(processed_frame, 0, 255).astype(np.uint8)
    
    # Estrai bordi per traccianti se abilitati
    current_logo_edges = np.zeros((processed_frame.shape[0], processed_frame.shape[1]), dtype=np.uint8)
    current_bg_edges = None
    
    if (hasattr(config, 'TRACER_ENABLED') and config.TRACER_ENABLED) or \
       (hasattr(config, 'BG_TRACER_ENABLED') and config.BG_TRACER_ENABLED):
        
        # Converti in scala di grigi per estrazione bordi
        gray_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
        
        # Estrai bordi usando Canny
        logo_threshold1 = getattr(config, 'TRACER_THRESHOLD1', 50)
        logo_threshold2 = getattr(config, 'TRACER_THRESHOLD2', 200)
        current_logo_edges = cv2.Canny(gray_frame, logo_threshold1, logo_threshold2)
        
        # Bordi per sfondo con soglie diverse
        if hasattr(config, 'BG_TRACER_ENABLED') and config.BG_TRACER_ENABLED:
            bg_threshold1 = getattr(config, 'BG_TRACER_THRESHOLD1', 20)
            bg_threshold2 = getattr(config, 'BG_TRACER_THRESHOLD2', 100)
            current_bg_edges = cv2.Canny(gray_frame, bg_threshold1, bg_threshold2)
            return processed_frame, current_logo_edges, current_bg_edges
    
    return processed_frame, current_logo_edges


def get_background_frame(bg_video, frame_index, bg_start_frame, config):
    """
    Funzione helper per ottenere un frame di sfondo con offset casuale.
    
    Args:
        bg_video: OpenCV VideoCapture object
        frame_index: Current frame index
        bg_start_frame: Starting frame offset for background video
        config: Configuration object
        
    Returns:
        Background frame or black frame if unavailable
    """
    if bg_video and bg_video.isOpened():
        # Calcola il frame considerando il rallentamento e l'offset casuale
        bg_frame_index = int(frame_index / config.BG_SLOWDOWN_FACTOR) + bg_start_frame
        bg_video.set(cv2.CAP_PROP_POS_FRAMES, bg_frame_index)
        ret, bg_frame = bg_video.read()
        if ret:
            return bg_frame
    
    # Fallback: crea un frame nero
    return np.zeros((config.HEIGHT, config.WIDTH, 3), dtype=np.uint8)


def get_timestamp_filename():
    """
    Genera un nome file con timestamp per i video di output.
    
    Returns:
        str: Nome file base con timestamp nel formato crystalpy_YYYYMMDD_HHMMSS
    """
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    return f"crystalpy_{timestamp}"


def get_video_writer_params(config, output_filename):
    """
    Determina i parametri per il VideoWriter in base al file di output.
    
    Args:
        config: Oggetto configurazione
        output_filename: Nome del file di output
        
    Returns:
        tuple: (fourcc, fps, size) parametri per cv2.VideoWriter
    """
    # Codec
    if output_filename.endswith('.mp4'):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    elif output_filename.endswith('.avi'):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
    else:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Default
    
    # FPS e dimensioni
    fps = config.FPS
    size = (config.WIDTH, config.HEIGHT)
    
    return fourcc, fps, size
