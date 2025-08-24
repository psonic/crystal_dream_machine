"""
🎨 CRYSTAL THERAPY - SISTEMA SHADER AVANZATO
Implementa tecniche di rendering avanzate per deformazioni fluide senza pixelamento.
"""

import cv2
import numpy as np
from typing import Tuple, Optional

def super_sample_deformation(mask: np.ndarray, map_x: np.ndarray, map_y: np.ndarray, 
                           scale_factor: int = 2) -> np.ndarray:
    """
    🔬 Super-sampling per deformazioni ultra-smooth.
    Render a risoluzione più alta, poi downscale per eliminare aliasing.
    
    Args:
        mask: Maschera originale
        map_x, map_y: Mappe di deformazione
        scale_factor: Fattore di super-sampling (2 = 4x pixel, 3 = 9x pixel)
    
    Returns:
        Maschera deformata con super-sampling
    """
    h, w = mask.shape
    
    # Upscale la maschera
    mask_upscaled = cv2.resize(mask, (w * scale_factor, h * scale_factor), 
                              interpolation=cv2.INTER_LANCZOS4)
    
    # Upscale le mappe di deformazione
    map_x_upscaled = cv2.resize(map_x, (w * scale_factor, h * scale_factor), 
                               interpolation=cv2.INTER_CUBIC) * scale_factor
    map_y_upscaled = cv2.resize(map_y, (w * scale_factor, h * scale_factor), 
                               interpolation=cv2.INTER_CUBIC) * scale_factor
    
    # Applica deformazione alla versione upscaled
    deformed_upscaled = cv2.remap(mask_upscaled, map_x_upscaled, map_y_upscaled,
                                 interpolation=cv2.INTER_LANCZOS4,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    # Downscale con filtro anti-aliasing
    deformed_final = cv2.resize(deformed_upscaled, (w, h), 
                               interpolation=cv2.INTER_AREA)
    
    return deformed_final


def edge_aware_blur(mask: np.ndarray, intensity: float = 0.5) -> np.ndarray:
    """
    🎯 Blur che preserva i bordi - smoothing senza perdere dettagli.
    
    Args:
        mask: Maschera da processare
        intensity: Intensità blur (0.0-1.0)
    
    Returns:
        Maschera con blur edge-aware
    """
    if intensity <= 0:
        return mask
    
    # Bilateral filter preserva i bordi
    kernel_size = 5
    sigma_color = 80 * intensity
    sigma_space = 80 * intensity
    
    # Converte in uint8 se necessario
    if mask.dtype != np.uint8:
        mask_uint8 = (mask * 255).astype(np.uint8)
    else:
        mask_uint8 = mask
    
    # Applica bilateral filter
    blurred = cv2.bilateralFilter(mask_uint8, kernel_size, sigma_color, sigma_space)
    
    # Ritorna nel formato originale
    if mask.dtype != np.uint8:
        return blurred.astype(np.float32) / 255.0
    else:
        return blurred


def anti_fattening_deformation(mask: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    """
    🎯 ANTI-FATTENING: Deformazione precisa che mantiene lo spessore originale.
    Usa interpolazione bilineare pura con conservazione della massa.
    
    Args:
        mask: Maschera originale
        map_x, map_y: Mappe di deformazione
    
    Returns:
        Maschera deformata senza ingrossamento
    """
    h, w = mask.shape
    
    # Assicura che la maschera sia float per interpolazione precisa
    if mask.dtype != np.float32:
        mask_float = mask.astype(np.float32) / 255.0
    else:
        mask_float = mask
    
    # INTERPOLAZIONE BILINEARE PURA (nessun filtro che possa ingrossare)
    deformed = cv2.remap(mask_float, map_x, map_y,
                        interpolation=cv2.INTER_LINEAR,  # Linear è il più "thin-preserving"
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    # POST-PROCESSING ANTI-FATTENING: Normalizza per conservare la massa
    original_mass = np.sum(mask_float)
    deformed_mass = np.sum(deformed)
    
    if deformed_mass > 0 and original_mass > 0:
        # Scala per conservare la massa totale (anti-fattening)
        mass_ratio = original_mass / deformed_mass
        if mass_ratio < 1.0:  # Solo se c'è stato ingrossamento
            deformed = deformed * mass_ratio
    
    # SHARPENING SELETTIVO: Recupera nitidezza persa dalla LINEAR
    # Solo sui bordi per non ingrossare le aree piene
    edges = cv2.Canny((deformed * 255).astype(np.uint8), 30, 100)
    edge_mask = (edges > 0).astype(np.float32) * 0.3  # Solo 30% di sharpening
    
    # Kernel di sharpening molto leggero
    sharpen_kernel = np.array([
        [0, -0.1, 0],
        [-0.1, 1.4, -0.1],
        [0, -0.1, 0]
    ])
    
    sharpened = cv2.filter2D(deformed, -1, sharpen_kernel)
    
    # Applica sharpening solo sui bordi
    result = deformed * (1 - edge_mask) + sharpened * edge_mask
    
    # Ritorna nel formato originale
    if mask.dtype != np.float32:
        return (np.clip(result, 0, 1) * 255.0).astype(mask.dtype)
    else:
        return np.clip(result, 0, 1)


def adaptive_interpolation(mask: np.ndarray, map_x: np.ndarray, map_y: np.ndarray,
                          edge_threshold: float = 0.1) -> np.ndarray:
    """
    🧠 Interpolazione adattiva: usa metodi diversi in base al contenuto.
    - Bordi netti: LANCZOS4 per preservare dettagli
    - Aree lisce: CUBIC per smoothness
    
    Args:
        mask: Maschera originale
        map_x, map_y: Mappe di deformazione
        edge_threshold: Soglia per rilevamento bordi
    
    Returns:
        Maschera deformata con interpolazione adattiva
    """
    h, w = mask.shape
    
    # Rileva i bordi nella maschera originale
    edges = cv2.Canny((mask * 255).astype(np.uint8), 50, 150)
    edge_mask = (edges > 0).astype(np.float32)
    
    # Dilata la maschera dei bordi per catturare aree vicine
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edge_mask_dilated = cv2.dilate(edge_mask, kernel, iterations=1)
    
    # Deformazione con LANCZOS4 per i bordi
    deformed_lanczos = cv2.remap(mask, map_x, map_y,
                                interpolation=cv2.INTER_LANCZOS4,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    # Deformazione con CUBIC per le aree lisce
    deformed_cubic = cv2.remap(mask, map_x, map_y,
                              interpolation=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    # Combina i due risultati basandosi sulla mappa dei bordi
    alpha = edge_mask_dilated
    result = deformed_lanczos * alpha + deformed_cubic * (1 - alpha)
    
    return result.astype(mask.dtype)


def temporal_stabilization(current_frame: np.ndarray, previous_frame: Optional[np.ndarray],
                          stabilization_factor: float = 0.1) -> np.ndarray:
    """
    ⏰ Stabilizzazione temporale per ridurre flickering tra frame.
    
    Args:
        current_frame: Frame corrente
        previous_frame: Frame precedente (None per il primo frame)
        stabilization_factor: Fattore di stabilizzazione (0.0-0.5)
    
    Returns:
        Frame stabilizzato
    """
    if previous_frame is None or stabilization_factor <= 0:
        return current_frame
    
    # Blend con frame precedente per ridurre flickering
    stabilized = cv2.addWeighted(current_frame, 1.0 - stabilization_factor,
                                previous_frame, stabilization_factor, 0)
    
    return stabilized


def apply_shader_deformation(mask: np.ndarray, map_x: np.ndarray, map_y: np.ndarray,
                           quality_level: str = "high",
                           previous_frame: Optional[np.ndarray] = None) -> np.ndarray:
    """
    🎨 Applica deformazione con shader-like quality.
    
    Args:
        mask: Maschera da deformare
        map_x, map_y: Mappe di deformazione
        quality_level: "low", "medium", "high", "ultra", "anti_fattening"
        previous_frame: Frame precedente per stabilizzazione temporale
    
    Returns:
        Maschera deformata con qualità shader
    """
    if quality_level == "low":
        # Metodo standard veloce
        return cv2.remap(mask, map_x, map_y,
                        interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    elif quality_level == "anti_fattening":
        # SOLUZIONE ANTI-FATTENING specializzata
        return anti_fattening_deformation(mask, map_x, map_y)
    
    elif quality_level == "medium":
        # Interpolazione adattiva
        return adaptive_interpolation(mask, map_x, map_y)
    
    elif quality_level == "high":
        # Super-sampling 2x + edge-aware blur
        result = super_sample_deformation(mask, map_x, map_y, scale_factor=2)
        result = edge_aware_blur(result, intensity=0.3)
        
        # Stabilizzazione temporale
        if previous_frame is not None:
            result = temporal_stabilization(result, previous_frame, 0.1)
        
        return result
    
    elif quality_level == "ultra":
        # Super-sampling 3x + interpolazione adattiva + stabilizzazione
        result = super_sample_deformation(mask, map_x, map_y, scale_factor=3)
        result = edge_aware_blur(result, intensity=0.2)
        
        # Stabilizzazione temporale più aggressiva
        if previous_frame is not None:
            result = temporal_stabilization(result, previous_frame, 0.15)
        
        return result
    
    else:
        raise ValueError(f"Quality level '{quality_level}' non supportato")


def create_flow_field_visualization(map_x: np.ndarray, map_y: np.ndarray,
                                  output_size: Tuple[int, int] = (512, 512)) -> np.ndarray:
    """
    🌊 Crea visualizzazione del campo di deformazione per debug.
    
    Args:
        map_x, map_y: Mappe di deformazione
        output_size: Dimensioni output per visualizzazione
    
    Returns:
        Immagine RGB che visualizza il flow field
    """
    h, w = map_x.shape
    
    # Ridimensiona per visualizzazione
    flow_x = cv2.resize(map_x, output_size, interpolation=cv2.INTER_LINEAR)
    flow_y = cv2.resize(map_y, output_size, interpolation=cv2.INTER_LINEAR)
    
    # Crea griglia di coordinate originali
    y_coords, x_coords = np.mgrid[0:output_size[1], 0:output_size[0]]
    
    # Calcola direzione e intensità del flow
    dx = flow_x - x_coords
    dy = flow_y - y_coords
    
    # Converti in coordinate polari
    magnitude = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dy, dx)
    
    # Normalizza per visualizzazione HSV
    # Hue = direzione, Value = intensità
    hue = ((angle + np.pi) / (2 * np.pi) * 179).astype(np.uint8)
    saturation = np.full_like(hue, 255, dtype=np.uint8)
    value = np.clip(magnitude * 10, 0, 255).astype(np.uint8)
    
    # Crea immagine HSV e converti in RGB
    hsv = np.stack([hue, saturation, value], axis=-1)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    return rgb
