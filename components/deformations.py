"""
🌊 Componente Deformazioni per CrystalPython3
Gestisce le deformazioni organiche e reattive all'audio usando noise di Perlin.
"""

import numpy as np
import cv2

# Importa il sistema shader avanzato
try:
    from .shaders import apply_shader_deformation, edge_aware_blur, adaptive_interpolation
    SHADER_AVAILABLE = True
    print("🎨 Sistema Shader disponibile - Qualità avanzata attivata!")
except ImportError:
    SHADER_AVAILABLE = False
    print("⚠️ Sistema Shader non disponibile - Usando metodi standard")

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


def apply_organic_deformation_old_style(mask, frame_index, params, dynamic_params=None):
    """Applica la vecchia deformazione organica con piccole ondulazioni sottili."""
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
    
    # VECCHIO APPROCCIO: Piccole ondulazioni usando griglia ottimizzata
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


def apply_organic_deformation_new_style(mask, frame_index, params, dynamic_params=None):
    """Applica deformazione organica stretch in stile semplice e funzionante."""
    if not NOISE_AVAILABLE:
        print("⚠️ Deformazione organica saltata: modulo noise non disponibile")
        return mask
    
    h, w = mask.shape

    config = params.get('config')

    # Usa parametri dinamici se forniti, altrimenti quelli statici
    if dynamic_params and False:
        print("dynamic fuck")
        speed = dynamic_params.get('deformation_speed', params['speed'])
        scale = dynamic_params.get('deformation_scale', params['scale'])
        intensity = dynamic_params.get('deformation_intensity', params['intensity'])
    else:
        speed = params['speed']
        scale = params['scale']
        intensity = params['intensity']

    # Nuovi parametri per controllo stretch avanzato
    horizontal_factor = params.get('horizontal_factor', 1.0)
    vertical_factor = params.get('vertical_factor', 1.0)
    fine_detail = params.get('fine_detail', 0.0)
    
    time_component = frame_index * speed
    
    # Coordinate per deformazione
    x_indices, y_indices = np.meshgrid(np.arange(w), np.arange(h))
    
    # Genera deformazione usando pnoise2 (molto più semplice e stabile)
    wave_x = np.zeros((h, w), dtype=np.float32)
    wave_y = np.zeros((h, w), dtype=np.float32)
    
    # Generiamo le onde con campionamento ogni X pixel per performance

    pixel_granularity = params.get('pixel_granularity', 1)

    for y in range(0, h, pixel_granularity):
        for x in range(0, w, pixel_granularity):
            # Onda X per stretching orizzontale (con fattore di controllo)
            wave_val_x = pnoise2(
                x * scale + time_component,
                y * scale * 0.5,
                octaves=3, persistence=0.5
            ) * intensity * 15 * horizontal_factor  # Applicato il fattore orizzontale
            
            # Onda Y per stretching verticale (con fattore di controllo)
            wave_val_y = pnoise2(
                x * scale * 0.5,
                y * scale + time_component * 0.007,
                octaves=3, persistence=0.5
            ) * intensity * 15 * vertical_factor  # Applicato il fattore verticale
            
            # Dettagli fini aggiuntivi se abilitati
            if fine_detail > 0:
                fine_noise_x = pnoise2(
                    x * scale * 5 + time_component * 0.02,
                    y * scale * 5,
                    octaves=2, persistence=0.3
                ) * intensity * fine_detail
                
                fine_noise_y = pnoise2(
                    x * scale * 5,
                    y * scale * 5 + time_component * 0.02,
                    octaves=2, persistence=0.3
                ) * intensity * fine_detail
                
                wave_val_x += fine_noise_x
                wave_val_y += fine_noise_y
            
            # Riempi il blocco 10x10
            wave_x[y:y+10, x:x+10] = wave_val_x
            wave_y[y:y+10, x:x+10] = wave_val_y
    
    # Interpola per rendere fluido
    wave_x = cv2.resize(wave_x, (w, h), interpolation=cv2.INTER_CUBIC)
    wave_y = cv2.resize(wave_y, (w, h), interpolation=cv2.INTER_CUBIC)
    
    # Applica deformazione additiva semplice
    map_x = (x_indices + wave_x).astype(np.float32)
    map_y = (y_indices + wave_y).astype(np.float32)
    
    # Clamp
    map_x = np.clip(map_x, 0, w-1)
    map_y = np.clip(map_y, 0, h-1)
    
    # Deformazione con interpolazione CUBIC (perfetto bilanciamento qualità/velocità)
    deformed_mask = cv2.remap(mask, map_x, map_y,
                             interpolation=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REFLECT)
    
    return deformed_mask
    
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
    
    # MIGLIORAMENTO ANTI-ALIASING: Sistema Shader Avanzato
    config = params.get('config')
    
    # Controlla se l'anti-aliasing è abilitato nella configurazione
    if config and hasattr(config, 'STRETCH_ANTIALIASING_ENABLED') and config.STRETCH_ANTIALIASING_ENABLED:
        
        # Determina livello qualità shader
        quality_level = "medium"  # default
        if hasattr(config, 'STRETCH_SHADER_QUALITY'):
            quality_level = config.STRETCH_SHADER_QUALITY
        
        # Usa sistema shader se disponibile
        if SHADER_AVAILABLE and quality_level in ["high", "ultra"]:
            print(f"🎨 Usando shader qualità {quality_level}")
            deformed_mask = apply_shader_deformation(mask, map_x, map_y, 
                                                   quality_level=quality_level)
        else:
            # Multi-pass standard migliorato
            # Pass 1: LANCZOS4 per dettagli fini (migliore per text/edge)
            deformed_mask_pass1 = cv2.remap(mask, map_x, map_y, 
                                           interpolation=cv2.INTER_LANCZOS4, 
                                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            
            # Pass 2: CUBIC per smoothness
            deformed_mask_pass2 = cv2.remap(mask, map_x, map_y, 
                                           interpolation=cv2.INTER_CUBIC, 
                                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            
            # Blend dei due passaggi per il meglio di entrambi
            blend_ratio = getattr(config, 'STRETCH_MULTIPASS_BLENDING', 0.7)
            deformed_mask = cv2.addWeighted(deformed_mask_pass1, blend_ratio, 
                                           deformed_mask_pass2, 1.0 - blend_ratio, 0)
            
            # MIGLIORAMENTO FINALE: Leggero blur selettivo per eliminare artefatti
            blur_threshold = getattr(config, 'STRETCH_BLUR_THRESHOLD', 10)
            blur_strength = getattr(config, 'STRETCH_BLUR_STRENGTH', 0.5)
            
            total_deformation_intensity = np.mean(np.abs(map_x - x_indices)) + np.mean(np.abs(map_y - y_indices))
            if total_deformation_intensity > blur_threshold:
                # Usa edge-aware blur se disponibile, altrimenti Gaussian standard
                if SHADER_AVAILABLE:
                    deformed_mask = edge_aware_blur(deformed_mask, intensity=blur_strength)
                else:
                    kernel_size = 3
                    deformed_mask = cv2.GaussianBlur(deformed_mask, (kernel_size, kernel_size), blur_strength)
    else:
        # Metodo standard senza anti-aliasing
        deformed_mask = cv2.remap(mask, map_x, map_y, 
                                 interpolation=cv2.INTER_CUBIC, 
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    return deformed_mask


def apply_organic_deformation(mask, frame_index, config, dynamic_params=None):
    """
    Funzione principale che applica le deformazioni basandosi sulla configurazione.
    Ora usa funzioni separate per i parametri di ogni tipo di deformazione.
    """
    # Determina quale deformazione applicare
    organic_enabled = hasattr(config, 'ORGANIC_DEFORMATION_ENABLED') and config.ORGANIC_DEFORMATION_ENABLED
    stretch_enabled = hasattr(config, 'STRETCH_DEFORMATION_ENABLED') and config.STRETCH_DEFORMATION_ENABLED
    
    result_mask = mask
    
    if organic_enabled and stretch_enabled:
        # APPLICAZIONE SEQUENZIALE: prima organic, poi stretch
        # 1. Prima applica deformazione organica (ondulazioni)
        organic_params = get_organic_deformation_params(config)
        result_mask = apply_organic_deformation_old_style(result_mask, frame_index, organic_params, dynamic_params)
        
        # 2. Poi applica stretch sulla maschera già deformata
        stretch_params = get_stretch_deformation_params(config)        
        result_mask = apply_organic_deformation_new_style(result_mask, frame_index, stretch_params, dynamic_params)
        
    elif organic_enabled:
        # Solo deformazione organica classica
        organic_params = get_organic_deformation_params(config)
        result_mask = apply_organic_deformation_old_style(result_mask, frame_index, organic_params, dynamic_params)
        
    elif stretch_enabled:        
        # Solo deformazione stretch
        stretch_params = get_stretch_deformation_params(config)        
        result_mask = apply_organic_deformation_new_style(result_mask, frame_index, stretch_params, dynamic_params)
    else:
        # Nessuno attivo, restituisci maschera non modificata
        pass
    
    return result_mask


def get_organic_deformation_params(config, enable_random_variation=False):
    """
    🌊 Genera i parametri per la deformazione organica classica (ondulazioni).
    
    Args:
        config: Configurazione con parametri base
        enable_random_variation: Se True, aggiunge variazione casuale ai parametri
    
    Returns:
        dict: Parametri per la deformazione organica
    """
    # Parametri base per deformazione organica
    base_speed = config.ORGANIC_SPEED if hasattr(config, 'ORGANIC_SPEED') else 0.015
    base_scale = config.ORGANIC_SCALE if hasattr(config, 'ORGANIC_SCALE') else 0.0008
    base_intensity = config.ORGANIC_INTENSITY if hasattr(config, 'ORGANIC_INTENSITY') else 25.0
    
    if enable_random_variation and hasattr(config, 'RANDOM_DEFORMATION_PARAMS') and config.RANDOM_DEFORMATION_PARAMS:
        # Genera parametri con variazione casuale
        deform_var_x = np.random.uniform(-0.3, 0.3)
        deform_var_y = np.random.uniform(-0.3, 0.3) 
        deform_var_z = np.random.uniform(-0.3, 0.3)
        
        speed = base_speed
        scale = base_scale * (1.0 + deform_var_y)
        intensity = base_intensity * (1.0 + deform_var_z)
    else:
        # Usa parametri statici dalla configurazione
        speed = base_speed
        scale = base_scale
        intensity = base_intensity
    
    return {
        'speed': speed,
        'scale': scale,
        'intensity': intensity
    }


def get_stretch_deformation_params(config, enable_random_variation=False):
    """
    🎪 Genera i parametri per la deformazione stretch (stretching organico).
    
    Args:
        config: Configurazione con parametri base
        enable_random_variation: Se True, aggiunge variazione casuale ai parametri
    
    Returns:
        dict: Parametri per la deformazione stretch
    """
    # Parametri base per deformazione stretch
    base_speed = config.STRETCH_SPEED if hasattr(config, 'STRETCH_SPEED') else 0.1
    base_scale = config.STRETCH_SCALE if hasattr(config, 'STRETCH_SCALE') else 0.002
    base_intensity = config.STRETCH_INTENSITY if hasattr(config, 'STRETCH_INTENSITY') else 50.0
    
    # Nuovi parametri per controllo avanzato
    horizontal_factor = config.STRETCH_HORIZONTAL_FACTOR if hasattr(config, 'STRETCH_HORIZONTAL_FACTOR') else 1.0
    vertical_factor = config.STRETCH_VERTICAL_FACTOR if hasattr(config, 'STRETCH_VERTICAL_FACTOR') else 1.0
    fine_detail = config.STRETCH_FINE_DETAIL if hasattr(config, 'STRETCH_FINE_DETAIL') else 0.0
    pixel_granularity = config.STRETCH_PIXEL_GRANULARITY if hasattr(config, 'STRETCH_PIXEL_GRANULARITY') else 10

    if enable_random_variation and hasattr(config, 'RANDOM_DEFORMATION_PARAMS') and config.RANDOM_DEFORMATION_PARAMS:
        # Genera parametri con variazione casuale        
        deform_var_y = np.random.uniform(-0.3, 0.3) 
        deform_var_z = np.random.uniform(-0.3, 0.3)

        speed = base_speed
        scale = base_scale * (1.0 + deform_var_y)
        intensity = base_intensity * (1.0 + deform_var_z)
    else:
        # Usa parametri statici dalla configurazione
        speed = base_speed
        scale = base_scale
        intensity = base_intensity

    return {
        'speed': speed,
        'scale': scale,
        'intensity': intensity,
        'horizontal_factor': horizontal_factor,
        'vertical_factor': vertical_factor,
        'fine_detail': fine_detail,
        'pixel_granularity': pixel_granularity
    }


def validate_deformation_config(config):
    """
    🔧 Valida e imposta valori di default per la configurazione delle deformazioni.
    Ora supporta sia parametri organici che stretch.
    
    Args:
        config: Oggetto configurazione da validare
    
    Returns:
        bool: True se la configurazione è valida
    """
    # Controlla almeno uno dei due sistemi di deformazione
    has_organic = (hasattr(config, 'ORGANIC_DEFORMATION_ENABLED') and 
                   hasattr(config, 'ORGANIC_SPEED') and 
                   hasattr(config, 'ORGANIC_SCALE') and 
                   hasattr(config, 'ORGANIC_INTENSITY'))
    
    has_stretch = (hasattr(config, 'STRETCH_DEFORMATION_ENABLED') and 
                   hasattr(config, 'STRETCH_SPEED') and 
                   hasattr(config, 'STRETCH_SCALE') and 
                   hasattr(config, 'STRETCH_INTENSITY'))
    
    if not has_organic and not has_stretch:
        print("⚠️ Nessun sistema di deformazione configurato correttamente")
        return False
    
    # Valida i valori organici se presenti
    if has_organic and config.ORGANIC_DEFORMATION_ENABLED:
        if config.ORGANIC_SPEED <= 0:
            print(f"⚠️ ORGANIC_SPEED deve essere > 0, trovato: {config.ORGANIC_SPEED}")
            return False
        if config.ORGANIC_SCALE <= 0:
            print(f"⚠️ ORGANIC_SCALE deve essere > 0, trovato: {config.ORGANIC_SCALE}")
            return False
        if config.ORGANIC_INTENSITY <= 0:
            print(f"⚠️ ORGANIC_INTENSITY deve essere > 0, trovato: {config.ORGANIC_INTENSITY}")
            return False
    
    # Valida i valori stretch se presenti
    if has_stretch and config.STRETCH_DEFORMATION_ENABLED:
        if config.STRETCH_SPEED <= 0:
            print(f"⚠️ STRETCH_SPEED deve essere > 0, trovato: {config.STRETCH_SPEED}")
            return False
        if config.STRETCH_SCALE <= 0:
            print(f"⚠️ STRETCH_SCALE deve essere > 0, trovato: {config.STRETCH_SCALE}")
            return False
        if config.STRETCH_INTENSITY <= 0:
            print(f"⚠️ STRETCH_INTENSITY deve essere > 0, trovato: {config.STRETCH_INTENSITY}")
            return False
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
    
    # Applica la deformazione passando direttamente la config
    return apply_organic_deformation(mask, frame_index, config, dynamic_params)
