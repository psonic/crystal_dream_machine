"""
🌊 Componente Deformazioni per CrystalPython3
Gestisce le deformazioni organiche e reattive all'audio usando noise di Perlin.
"""

import numpy as np
import cv2

# Import condizionale per noise
NOISE_AVAILABLE = False
pnoise2 = None

try:
    from noise import pnoise2
    NOISE_AVAILABLE = True
    print("🌊 Perlin noise disponibile - Deformazioni organiche attivate!")
except ImportError:
    NOISE_AVAILABLE = False
    print("⚠️ Noise non disponibile - Deformazioni organiche disabilitate")
    print("   Per abilitare le deformazioni: pip install noise")


def apply_organic_deformation(mask, frame_index, params, dynamic_params=None):
    """Applica una deformazione organica che stira e allunga la scritta in modo drammatico."""
    if not NOISE_AVAILABLE:
        print("⚠️ Deformazione organica saltata: modulo noise non disponibile")
        return mask
    
    h, w = mask.shape
    
    # Usa parametri dinamici se forniti, altrimenti quelli statici
    if dynamic_params:
        speed = dynamic_params.get('deformation_speed', params['speed'])
        scale = dynamic_params.get('deformation_scale', params['scale'])
        intensity = dynamic_params.get('deformation_intensity', params['intensity'])
    else:
        speed = params['speed']
        scale = params['scale']
        intensity = params['intensity']
    
    time_component = frame_index * speed
    
    # NUOVO APPROCCIO: Stretching organico invece di piccole ondulazioni
    x_indices, y_indices = np.meshgrid(np.arange(w), np.arange(h))
    
    # Onde principali per stretching orizzontale (più ampie e drammatiche)
    wave_frequency_x = scale * 2.0  # Onde più ampie
    wave_amplitude_x = intensity * 0.8  # Stretching più evidente
    
    # Onde principali per stretching verticale 
    wave_frequency_y = scale * 1.5
    wave_amplitude_y = intensity * 0.6
    
    # Stretching orizzontale organico (effetto "fisarmonica")
    horizontal_stretch = np.zeros_like(x_indices, dtype=np.float32)
    for y in range(0, h, 8):  # Campionamento per performance
        for x in range(0, w, 8):
            # Onde principali per stretching
            stretch_factor = pnoise2(
                x * wave_frequency_x + time_component,
                y * wave_frequency_x * 0.3,
                octaves=3, persistence=0.6, lacunarity=2.5
            )
            # Converti in fattore di stretching (0.5 = compressione, 2.0 = allungamento)
            stretch_factor = 0.7 + stretch_factor * 0.6  # Range: 0.1 - 1.3
            horizontal_stretch[y:y+8, x:x+8] = stretch_factor
    
    # Stretching verticale organico (effetto "respirazione")
    vertical_stretch = np.zeros_like(y_indices, dtype=np.float32)
    for y in range(0, h, 8):
        for x in range(0, w, 8):
            stretch_factor = pnoise2(
                x * wave_frequency_y * 0.5,
                y * wave_frequency_y + time_component * 0.7,
                octaves=2, persistence=0.7, lacunarity=3.0
            )
            stretch_factor = 0.8 + stretch_factor * 0.4  # Range: 0.4 - 1.2
            vertical_stretch[y:y+8, x:x+8] = stretch_factor
    
    # Interpola per ottenere valori fluidi
    horizontal_stretch = cv2.resize(horizontal_stretch, (w, h), interpolation=cv2.INTER_CUBIC)
    vertical_stretch = cv2.resize(vertical_stretch, (w, h), interpolation=cv2.INTER_CUBIC)
    
    # Applica lo stretching organico
    center_x, center_y = w // 2, h // 2
    
    # Calcola nuove coordinate con stretching
    map_x = center_x + (x_indices - center_x) * horizontal_stretch
    map_y = center_y + (y_indices - center_y) * vertical_stretch
    
    # Aggiungi anche piccole ondulazioni per organicità extra
    fine_noise_x = np.zeros((h, w), dtype=np.float32)
    fine_noise_y = np.zeros((h, w), dtype=np.float32)
    
    for y in range(0, h, 4):
        for x in range(0, w, 4):
            fine_noise_x[y:y+4, x:x+4] = pnoise2(
                x * scale * 8 + time_component * 2,
                y * scale * 8,
                octaves=3, persistence=0.4
            ) * intensity * 0.2
            
            fine_noise_y[y:y+4, x:x+4] = pnoise2(
                x * scale * 8,
                y * scale * 8 + time_component * 2,
                octaves=3, persistence=0.4
            ) * intensity * 0.2
    
    # Combina stretching e ondulazioni fini
    map_x = map_x + fine_noise_x
    map_y = map_y + fine_noise_y
    
    # Assicurati che le coordinate siano nei limiti
    map_x = np.clip(map_x, 0, w-1).astype(np.float32)
    map_y = np.clip(map_y, 0, h-1).astype(np.float32)
    
    # Applica la deformazione
    deformed_mask = cv2.remap(mask, map_x, map_y, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    return deformed_mask


def get_organic_deformation_params(config, enable_random_variation=True):
    """
    🌊 Genera i parametri per la deformazione organica.
    
    Args:
        config: Configurazione con parametri base
        enable_random_variation: Se True, aggiunge variazione casuale ai parametri
    
    Returns:
        dict: Parametri per la deformazione organica
    """
    params = {}
    
    if enable_random_variation and hasattr(config, 'RANDOM_DEFORMATION_PARAMS') and config.RANDOM_DEFORMATION_PARAMS:
        # Genera parametri con variazione casuale
        deform_var_x = np.random.uniform(-0.3, 0.3)
        deform_var_y = np.random.uniform(-0.3, 0.3) 
        deform_var_z = np.random.uniform(-0.3, 0.3)
        
        params['deformation_speed'] = config.DEFORMATION_SPEED * (1.0 + deform_var_x)
        params['deformation_scale'] = config.DEFORMATION_SCALE * (1.0 + deform_var_y)
        params['deformation_intensity'] = config.DEFORMATION_INTENSITY * (1.0 + deform_var_z)
    else:
        # Usa parametri statici dalla configurazione
        params['deformation_speed'] = config.DEFORMATION_SPEED
        params['deformation_scale'] = config.DEFORMATION_SCALE
        params['deformation_intensity'] = config.DEFORMATION_INTENSITY
    
    # Converte i nomi per compatibilità con apply_organic_deformation
    return {
        'speed': params['deformation_speed'],
        'scale': params['deformation_scale'],
        'intensity': params['deformation_intensity']
    }


def validate_deformation_config(config):
    """
    🔧 Valida e imposta valori di default per la configurazione delle deformazioni.
    
    Args:
        config: Oggetto configurazione da validare
    
    Returns:
        bool: True se la configurazione è valida
    """
    required_attrs = [
        'DEFORMATION_SPEED',
        'DEFORMATION_SCALE', 
        'DEFORMATION_INTENSITY'
    ]
    
    missing_attrs = []
    for attr in required_attrs:
        if not hasattr(config, attr):
            missing_attrs.append(attr)
    
    if missing_attrs:
        print(f"⚠️ Attributi di configurazione deformazione mancanti: {missing_attrs}")
        return False
    
    # Valida i valori
    if config.DEFORMATION_SPEED <= 0:
        print("⚠️ DEFORMATION_SPEED deve essere maggiore di 0")
        return False
        
    if config.DEFORMATION_SCALE <= 0:
        print("⚠️ DEFORMATION_SCALE deve essere maggiore di 0") 
        return False
        
    if config.DEFORMATION_INTENSITY < 0:
        print("⚠️ DEFORMATION_INTENSITY deve essere maggiore o uguale a 0")
        return False
    
    print("✅ Configurazione deformazioni validata")
    return True


def apply_deformation_wrapper(mask, frame_index, config, dynamic_params=None):
    """
    🌊 Wrapper per l'applicazione delle deformazioni che gestisce la configurazione.
    
    Args:
        mask: Maschera da deformare
        frame_index: Indice del frame corrente
        config: Configurazione
        dynamic_params: Parametri dinamici (opzionali)
    
    Returns:
        numpy.ndarray: Maschera deformata
    """
    if not NOISE_AVAILABLE:
        return mask
    
    if not validate_deformation_config(config):
        print("⚠️ Configurazione deformazioni non valida, saltando deformazione")
        return mask
    
    # Genera i parametri base
    params = get_organic_deformation_params(config)
    
    # Applica la deformazione
    return apply_organic_deformation(mask, frame_index, params, dynamic_params)
