"""
Tracer System Component for Crystal Python

This module handles all tracer-related functionality including:
- Logo edge detection and tracing
- Background tracer effects
- Dynamic color shifting for tracers
- Opacity management and trail effects
- Tracer history management
"""

import cv2
import numpy as np
from collections import deque


def extract_logo_tracers(logo_mask, config):
    """
    Estrae i contorni dal logo stesso per creare traccianti più aderenti.
    
    Args:
        logo_mask: Binary mask of the logo
        config: Configuration object containing tracer settings
        
    Returns:
        Logo edges for tracer effects
    """
    # Estrae i bordi della maschera del logo
    logo_edges = cv2.Canny(logo_mask, 50, 150)
    
    # Dilata leggermente i bordi per renderli più visibili
    kernel = np.ones((2,2), np.uint8)
    logo_edges = cv2.dilate(logo_edges, kernel, iterations=1)
    
    return logo_edges


def extract_logo_and_bg_tracers(bg_frame, config):
    """
    Extract both logo and background tracers from background frame.
    
    Args:
        bg_frame: Background frame for tracer extraction
        config: Configuration object
        
    Returns:
        tuple: (logo_edges, bg_edges) or (logo_edges, None) if bg_tracer disabled
    """
    # Convert to grayscale for edge detection
    gray_bg = cv2.cvtColor(bg_frame, cv2.COLOR_BGR2GRAY)
    
    # Extract logo edges
    logo_edges = cv2.Canny(gray_bg, config.TRACER_THRESHOLD1, config.TRACER_THRESHOLD2)
    
    # Extract background edges if enabled
    bg_edges = None
    if hasattr(config, 'BG_TRACER_ENABLED') and config.BG_TRACER_ENABLED:
        # Use different thresholds for background
        bg_edges = cv2.Canny(gray_bg, config.BG_TRACER_THRESHOLD1, config.BG_TRACER_THRESHOLD2)
    
    return logo_edges, bg_edges


def apply_logo_tracers(final_frame, tracer_history, frame_index, config, dynamic_params):
    """
    Apply logo tracer effects to the frame.
    
    Args:
        final_frame: Current frame to apply tracers to
        tracer_history: History of logo tracer data
        frame_index: Current frame index for dynamic effects
        config: Configuration object
        dynamic_params: Dynamic parameters for opacity modulation
        
    Returns:
        Modified frame with logo tracers applied
    """
    if not config.TRACER_ENABLED or len(tracer_history) == 0:
        return final_frame
        
    tracer_layer = np.zeros_like(final_frame, dtype=np.float32)
    
    # Apply dynamic multiplier to opacity
    dynamic_opacity = config.TRACER_MAX_OPACITY * dynamic_params.get('tracer_opacity_multiplier', 1.0)
    opacities = np.linspace(0, dynamic_opacity, len(tracer_history))
    
    for i, past_edges in enumerate(reversed(tracer_history)):
        # Dynamic color for tracers - solo lungo la scia, non nel tempo
        hue_shift = (i * 0.5) % 180  # Solo basato sulla posizione nella scia
        base_color_hsv = cv2.cvtColor(np.uint8([[config.TRACER_BASE_COLOR]]), cv2.COLOR_BGR2HSV)[0][0]
        new_hue = (base_color_hsv[0] + hue_shift) % 180
        dynamic_color_hsv = np.uint8([[[new_hue, base_color_hsv[1], base_color_hsv[2]]]])
        dynamic_color_bgr = cv2.cvtColor(dynamic_color_hsv, cv2.COLOR_HSV2BGR)[0][0]
        
        # Color the edges and apply dynamic opacity
        colored_tracer = cv2.cvtColor(past_edges, cv2.COLOR_GRAY2BGR).astype(np.float32)
        colored_tracer[past_edges > 0] = np.array(dynamic_color_bgr, dtype=np.float32)
        tracer_with_opacity = cv2.multiply(colored_tracer, opacities[i])
        tracer_layer = cv2.add(tracer_layer, tracer_with_opacity)
    
    # Apply tracer layer to final frame
    final_frame = cv2.add(final_frame.astype(np.float32), tracer_layer)
    final_frame = np.clip(final_frame, 0, 255).astype(np.uint8)
    
    return final_frame


def apply_background_tracers(final_frame, bg_tracer_history, frame_index, config, dynamic_params):
    """
    Apply background tracer effects to the frame.
    
    Args:
        final_frame: Current frame to apply tracers to
        bg_tracer_history: History of background tracer data
        frame_index: Current frame index for dynamic effects
        config: Configuration object
        dynamic_params: Dynamic parameters for opacity modulation
        
    Returns:
        Modified frame with background tracers applied
    """
    if not (hasattr(config, 'BG_TRACER_ENABLED') and config.BG_TRACER_ENABLED and len(bg_tracer_history) > 0):
        return final_frame
        
    bg_tracer_layer = np.zeros_like(final_frame, dtype=np.float32)
    
    # Apply dynamic multiplier to background opacity
    dynamic_bg_opacity = config.BG_TRACER_MAX_OPACITY * dynamic_params.get('bg_tracer_opacity_multiplier', 1.0)
    bg_opacities = np.linspace(0, dynamic_bg_opacity, len(bg_tracer_history))
    
    for i, past_bg_edges in enumerate(reversed(bg_tracer_history)):
        # Dynamic color for background tracers - solo lungo la scia, non nel tempo
        hue_shift_bg = (i * 0.3) % 180  # Solo basato sulla posizione nella scia
        base_color_hsv_bg = cv2.cvtColor(np.uint8([[config.BG_TRACER_BASE_COLOR]]), cv2.COLOR_BGR2HSV)[0][0]
        new_hue_bg = (base_color_hsv_bg[0] + hue_shift_bg) % 180
        dynamic_color_hsv_bg = np.uint8([[[new_hue_bg, base_color_hsv_bg[1], base_color_hsv_bg[2]]]])
        dynamic_color_bgr_bg = cv2.cvtColor(dynamic_color_hsv_bg, cv2.COLOR_HSV2BGR)[0][0]
        
        # Color the background edges and apply dynamic opacity
        colored_bg_tracer = cv2.cvtColor(past_bg_edges, cv2.COLOR_GRAY2BGR).astype(np.float32)
        colored_bg_tracer[past_bg_edges > 0] = np.array(dynamic_color_bgr_bg, dtype=np.float32)
        bg_tracer_with_opacity = cv2.multiply(colored_bg_tracer, bg_opacities[i])
        bg_tracer_layer = cv2.add(bg_tracer_layer, bg_tracer_with_opacity)
    
    # Apply background tracer layer to final frame
    final_frame = cv2.add(final_frame.astype(np.float32), bg_tracer_layer)
    final_frame = np.clip(final_frame, 0, 255).astype(np.uint8)
    
    return final_frame


def initialize_tracer_histories(config):
    """
    Initialize tracer history data structures.
    
    Args:
        config: Configuration object containing tracer settings
        
    Returns:
        tuple: (tracer_history, bg_tracer_history) - deque objects for storing tracer data
    """
    # Initialize logo tracer history
    tracer_history = deque(maxlen=config.TRACER_TRAIL_LENGTH)
    
    # Initialize background tracer history
    bg_tracer_history = deque(maxlen=getattr(config, 'BG_TRACER_TRAIL_LENGTH', 35))
    
    return tracer_history, bg_tracer_history


def update_tracer_histories(tracer_history, bg_tracer_history, current_logo_edges, current_bg_edges, config):
    """
    Update tracer histories with current frame data.
    
    Args:
        tracer_history: Logo tracer history deque
        bg_tracer_history: Background tracer history deque
        current_logo_edges: Current frame logo edges
        current_bg_edges: Current frame background edges (can be None)
        config: Configuration object
    """
    # Update logo tracer history
    if config.TRACER_ENABLED:
        tracer_history.append(current_logo_edges)
    
    # Update background tracer history
    if hasattr(config, 'BG_TRACER_ENABLED') and config.BG_TRACER_ENABLED and current_bg_edges is not None:
        bg_tracer_history.append(current_bg_edges)


def calculate_tracer_dynamic_params(base_seed, config):
    """
    Calculate dynamic parameters for tracer opacity modulation.
    
    Args:
        base_seed: Base seed for parameter calculation
        config: Configuration object
        
    Returns:
        dict: Dictionary containing tracer dynamic parameters
    """
    # Calculate tracer variations
    tracer_var_x = np.sin(base_seed * config.VARIATION_SPEED_FAST + 2.0) * config.VARIATION_AMPLITUDE
    tracer_var_y = np.cos(base_seed * config.VARIATION_SPEED_FAST + 6.0) * config.VARIATION_AMPLITUDE
    
    params = {}
    params['tracer_opacity_multiplier'] = 1.0 + tracer_var_x
    params['bg_tracer_opacity_multiplier'] = 1.0 + tracer_var_y
    
    return params
