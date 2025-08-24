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
    """Applica una deformazione organica super fluida usando calcolo a griglia con parametri dinamici."""
    if not NOISE_AVAILABLE:
        # Se il modulo noise non è disponibile, restituisce la mask non modificata
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
    
    # Creo una griglia ridotta per calcolare il noise più velocemente
    # poi interpolo per ottenere un movimento fluido
    grid_size = 6  # Griglia più fitta per curve più morbide, ma ancora ottimizzata
    h_grid = h // grid_size + 1
    w_grid = w // grid_size + 1
    
    # Griglie per il noise
    noise_x = np.zeros((h_grid, w_grid), dtype=np.float32)
    noise_y = np.zeros((h_grid, w_grid), dtype=np.float32)
    
    # Calcolo il noise solo sui punti della griglia
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
    
    # Interpolo il noise per ottenere valori fluidi per tutti i pixel
    noise_x_full = cv2.resize(noise_x, (w, h), interpolation=cv2.INTER_CUBIC)
    noise_y_full = cv2.resize(noise_y, (w, h), interpolation=cv2.INTER_CUBIC)
    
    # Applico l'intensità dinamica
    displacement_x = noise_x_full * intensity
    displacement_y = noise_y_full * intensity
    
    # Creo le mappe di rimappatura
    x_indices, y_indices = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x_indices + displacement_x).astype(np.float32)
    map_y = (y_indices + displacement_y).astype(np.float32)
    
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
