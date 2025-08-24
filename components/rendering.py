"""
Rendering Helper Functions Component for Crystal Python

This module contains helper functions for the rendering pipeline:
- Dynamic parameter calculation
- Background frame processing
- Background frame retrieval
- Timestamp generation utilities
"""

import cv2
import numpy as np
from datetime import datetime
import random

from components.deformations import get_organic_deformation_params
from components.tracers import calculate_tracer_dynamic_params, extract_logo_and_bg_tracers


def get_dynamic_parameters(frame_index, total_frames, config):
    """
    Calcola parametri che cambiano automaticamente nel tempo per creare variazioni.
    
    Args:
        frame_index: Current frame index
        total_frames: Total number of frames
        config: Configuration object
        
    Returns:
        dict: Dictionary containing dynamic parameters for effects
    """
    t = frame_index / total_frames  # Progresso animazione (0.0 a 1.0)
    params = {}

    # Pulsazione del glow
    glow_pulse = np.sin(t * np.pi)
    params['glow_intensity'] = config.GLOW_INTENSITY + (glow_pulse * 0.2)

    # Variazioni automatiche dei parametri principali
    if config.DYNAMIC_VARIATION_ENABLED:
        base_seed = frame_index * 0.001
        
        # Usa il nuovo componente deformazioni per parametri dinamici
        # (compatibilità mantenuta con i nomi della Config)
        enable_variation = config.DYNAMIC_VARIATION_ENABLED
        deformation_params = get_organic_deformation_params(config, enable_variation)
        
        params['deformation_speed'] = deformation_params['speed']
        params['deformation_scale'] = deformation_params['scale']
        params['deformation_intensity'] = deformation_params['intensity']
        
        # Variazioni medie per lenti
        lens_var_x = np.sin(base_seed * config.VARIATION_SPEED_MEDIUM + 3.0) * config.VARIATION_AMPLITUDE
        lens_var_y = np.cos(base_seed * config.VARIATION_SPEED_MEDIUM + 5.5) * config.VARIATION_AMPLITUDE
        
        params['lens_speed_factor'] = config.LENS_SPEED_FACTOR * (1.0 + lens_var_x)
        params['lens_strength_multiplier'] = 1.0 + lens_var_y
        
        # Aggiungi parametri tracers tramite funzione dedicata
        tracer_params = calculate_tracer_dynamic_params(base_seed, config)
        params.update(tracer_params)
    else:
        # Usa valori fissi se le variazioni sono disabilitate
        deformation_params = get_organic_deformation_params(config, False)
        params['deformation_speed'] = deformation_params['speed']
        params['deformation_scale'] = deformation_params['scale']
        params['deformation_intensity'] = deformation_params['intensity']
        params['lens_speed_factor'] = config.LENS_SPEED_FACTOR
        params['lens_strength_multiplier'] = 1.0
        
        # Aggiungi parametri tracers con valori fissi
        tracer_params = {
            'tracer_opacity_multiplier': 1.0,
            'bg_tracer_opacity_multiplier': 1.0
        }
        params.update(tracer_params)
    
    return params


def process_background(bg_frame, config):
    """
    Processa il frame di sfondo: lo adatta alle dimensioni video senza crop,
    lo scurisce e ne estrae i contorni per i traccianti.
    
    Args:
        bg_frame: Raw background frame from video
        config: Configuration object
        
    Returns:
        tuple: (processed_frame, logo_edges, bg_edges)
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

    # 3. & 4. Estrae i traccianti del logo e dello sfondo usando la funzione dedicata
    gray_bg = cv2.cvtColor(final_bg, cv2.COLOR_BGR2GRAY)  # Usa il frame processato
    # Applica un leggero blur per ridurre il rumore prima di Canny
    gray_bg = cv2.GaussianBlur(gray_bg, (3, 3), 0)
    
    # Crea frame temporaneo per l'estrazione tracers
    temp_frame = cv2.cvtColor(gray_bg, cv2.COLOR_GRAY2BGR)
    logo_edges, bg_edges = extract_logo_and_bg_tracers(temp_frame, config)
    
    return final_bg, logo_edges, bg_edges


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
    
    # Fallback: frame nero
    return np.zeros((config.HEIGHT, config.WIDTH, 3), dtype=np.uint8)


def get_timestamp_filename():
    """
    Genera nome file con timestamp e carattere decorativo.
    
    Returns:
        str: Filename with timestamp and decorative character
    """
    decorative_chars = ['α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'ι', 'κ', 'λ', 'μ', 'ν', 'ξ', 'ο', 'π', 'ρ', 'σ', 'τ', 'υ', 'φ', 'χ', 'ψ', 'ω', 'Α', 'Β', 'Γ', 'Δ', 'Ε', 'Ζ', 'Η', 'Θ', 'Ι', 'Κ', 'Λ', 'Μ', 'Ν', 'Ξ', 'Ο', 'Π', 'Ρ', 'Σ', 'Τ', 'Υ', 'Φ', 'Χ', 'Ψ', 'Ω', '✨', '🌟', '💫', '⭐', '🌙', '☀️', '🔥', '💎', '🌊', '🎨', '🎭', '🎪', '🎯', '🎲', '🔮', '💜', '💙', '💚', '💛', '🧡', '❤️', '🖤', '🤍', '💖', '💝', '☯', '🕉', 'ॐ']
    decorative_char = random.choice(decorative_chars)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"crystalpy_{timestamp}_{decorative_char}"


def get_video_writer_params(config, output_filename):
    """
    Get video writer parameters and codec configuration.
    
    Args:
        config: Configuration object
        output_filename: Output video filename
        
    Returns:
        tuple: (fourcc, fps, size) for VideoWriter
    """
    # Try different codecs based on WhatsApp compatibility
    if config.WHATSAPP_COMPATIBLE:
        print("🔄 Usando codec ottimizzati per WhatsApp...")
        codecs_to_try = [
            cv2.VideoWriter_fourcc(*'H264'),  # H264 - best for WhatsApp
            cv2.VideoWriter_fourcc(*'mp4v'),  # MP4V - fallback 
            cv2.VideoWriter_fourcc(*'XVID'),  # XVID - last resort
        ]
    else:
        codecs_to_try = [
            cv2.VideoWriter_fourcc(*'mp4v'),  # MP4V - most compatible
            cv2.VideoWriter_fourcc(*'XVID'),  # XVID - fallback
            cv2.VideoWriter_fourcc(*'H264'),  # H264 - high quality
        ]
    
    size = (config.WIDTH, config.HEIGHT)
    fps = config.FPS
    
    for i, fourcc in enumerate(codecs_to_try):
        test_writer = cv2.VideoWriter(output_filename, fourcc, fps, size)
        if test_writer.isOpened():
            test_writer.release()
            print(f"✅ Codec funzionante trovato: {['H264', 'mp4v', 'XVID'][i] if config.WHATSAPP_COMPATIBLE else ['mp4v', 'XVID', 'H264'][i]}")
            return fourcc, fps, size
        else:
            codec_name = ['H264', 'mp4v', 'XVID'][i] if config.WHATSAPP_COMPATIBLE else ['mp4v', 'XVID', 'H264'][i]
            print(f"❌ Codec {codec_name} non funziona, provo il prossimo...")
    
    # If all fail, return the first one (will be handled by caller)
    print("⚠️ Nessun codec funziona, ritorno il primo per gestione errore...")
    return codecs_to_try[0], fps, size
