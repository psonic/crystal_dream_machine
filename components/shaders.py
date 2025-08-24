"""
🎨 CRYSTAL THERAPY - SISTEMA SHADER AVANZATO
Implementa tecniche di rendering avanzate per deformazioni fluide senza pixelamento.
"""

import cv2
import numpy as np
from typing import Tuple, Optional

def sub_pixel_deformation(mask: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    """
    🎯 Deformazione sub-pixel precisa senza ingrossamento.
    Usa interpolazione bilineare ad alta precisione invece di super-sampling.
    
    Args:
        mask: Maschera originale
        map_x, map_y: Mappe di deformazione
    
    Returns:
        Maschera deformata con precisione sub-pixel
    """
    h, w = mask.shape
    
    # Assicura che la maschera sia float per interpolazione precisa
    if mask.dtype != np.float32:
        mask_float = mask.astype(np.float32) / 255.0
    else:
        mask_float = mask
    
    # Applica deformazione con interpolazione LANCZOS4 (migliore per preservare dettagli)
    # ma con bordi riflessi per evitare artefatti ai margini
    deformed = cv2.remap(mask_float, map_x, map_y,
                        interpolation=cv2.INTER_LANCZOS4,
                        borderMode=cv2.BORDER_REFLECT_101)
    
    # Ritorna nel formato originale
    if mask.dtype != np.float32:
        return (deformed * 255.0).astype(mask.dtype)
    else:
        return deformed


def smart_interpolation(mask: np.ndarray, map_x: np.ndarray, map_y: np.ndarray,
                       edge_threshold: float = 30) -> np.ndarray:
    """
    🧠 Interpolazione intelligente che preserva i dettagli senza ingrossare.
    
    Args:
        mask: Maschera originale
        map_x, map_y: Mappe di deformazione
        edge_threshold: Soglia per rilevamento bordi (più basso = più sensibile)
    
    Returns:
        Maschera deformata con interpolazione intelligente
    """
    h, w = mask.shape
    
    # Converte in uint8 per edge detection se necessario
    if mask.dtype != np.uint8:
        mask_uint8 = (mask * 255).astype(np.uint8) if mask.max() <= 1.0 else mask.astype(np.uint8)
    else:
        mask_uint8 = mask
    
    # Rileva i bordi con Canny più sensibile
    edges = cv2.Canny(mask_uint8, edge_threshold, edge_threshold * 2)
    edge_mask = (edges > 0).astype(np.float32)
    
    # Dilata leggermente la maschera dei bordi
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edge_mask_dilated = cv2.dilate(edge_mask, kernel, iterations=1)
    
    # Per i bordi: usa LANCZOS4 per preservare nitidezza
    deformed_edges = cv2.remap(mask, map_x, map_y,
                              interpolation=cv2.INTER_LANCZOS4,
                              borderMode=cv2.BORDER_REFLECT_101)
    
    # Per le aree lisce: usa interpolazione bilineare più smooth
    deformed_smooth = cv2.remap(mask, map_x, map_y,
                               interpolation=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT_101)
    
    # Combina i due risultati
    alpha = edge_mask_dilated
    result = deformed_edges * alpha + deformed_smooth * (1 - alpha)
    
    return result.astype(mask.dtype)


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
                           quality: str = "medium", temporal_smoothing: bool = False) -> np.ndarray:
    """
    🎨 Applica deformazione con qualità shader configurabile.
    
    Args:
        mask: Maschera da deformare
        map_x, map_y: Mappe di deformazione
        quality: Livello qualità ('low', 'medium', 'high', 'ultra')
        temporal_smoothing: Abilita smoothing temporale (future)
    
    Returns:
        Maschera deformata con qualità shader
    """
    print(f"🎨 Usando shader qualità {quality}")
    
    if quality == "low":
        # Interpolazione semplice e veloce
        result = cv2.remap(mask, map_x, map_y,
                          interpolation=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        
    elif quality == "medium":
        # Sub-pixel precision senza ingrossamento
        result = sub_pixel_deformation(mask, map_x, map_y)
        
    elif quality == "high":
        # Interpolazione intelligente con edge detection
        result = smart_interpolation(mask, map_x, map_y, edge_threshold=25)
        
    elif quality == "ultra":
        # Interpolazione intelligente ultra-precisa + edge enhancement
        result = smart_interpolation(mask, map_x, map_y, edge_threshold=15)
        # Leggero sharpening solo sui bordi per compensare il blur della deformazione
        kernel = np.array([[-0.1, -0.1, -0.1],
                          [-0.1,  1.8, -0.1],
                          [-0.1, -0.1, -0.1]])
        enhanced = cv2.filter2D(result.astype(np.float32), -1, kernel)
        result = np.clip(enhanced, 0, 255).astype(mask.dtype)
        
    else:
        # Fallback su medium
        result = sub_pixel_deformation(mask, map_x, map_y)
    
    return result


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
