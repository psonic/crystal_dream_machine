"""
🎨 Componente Blending per CrystalPython3
Gestisce il sistema di blending avanzato, preset, effetti visivi e texture.
"""

import numpy as np
import cv2
import os


def apply_blending_preset(config):
    """
    🎨 Applica automaticamente i preset di blending alla configurazione.
    Se BLENDING_PRESET != 'manual', sovrascrive i parametri di blending.
    """
    if config.BLENDING_PRESET == 'manual':
        print("🔧 Usando configurazione blending manuale")
        return
    
    # Definizione dei preset (importati dal file blending_presets.py)
    presets = {
        'cinematic': {
            'BLENDING_MODE': 'overlay',
            'BLENDING_STRENGTH': 0.7,
            'EDGE_DETECTION_ENABLED': False,
            'EDGE_BLUR_RADIUS': 3,
            'ADAPTIVE_BLENDING': True,
            'COLOR_HARMONIZATION': True,
            'LUMINANCE_MATCHING': False,
            'BLEND_TRANSPARENCY': 0.2,
            'COLOR_BLENDING_STRENGTH': 0.9
        },
        'artistic': {
            'BLENDING_MODE': 'difference',
            'BLENDING_STRENGTH': 0.9,
            'EDGE_DETECTION_ENABLED': True,
            'EDGE_BLUR_RADIUS': 5,
            'ADAPTIVE_BLENDING': False,
            'COLOR_HARMONIZATION': False,
            'LUMINANCE_MATCHING': False,
            'BLEND_TRANSPARENCY': 0.1,
            'COLOR_BLENDING_STRENGTH': 0.2
        },
        'soft': {
            'BLENDING_MODE': 'soft_light',
            'BLENDING_STRENGTH': 0.6,
            'EDGE_DETECTION_ENABLED': True,
            'EDGE_BLUR_RADIUS': 3,
            'ADAPTIVE_BLENDING': True,
            'COLOR_HARMONIZATION': True,
            'LUMINANCE_MATCHING': True,
            'BLEND_TRANSPARENCY': 0.4,
            'COLOR_BLENDING_STRENGTH': 0.5
        },
        'dramatic': {
            'BLENDING_MODE': 'multiply',
            'BLENDING_STRENGTH': 0.85,
            'EDGE_DETECTION_ENABLED': True,
            'EDGE_BLUR_RADIUS': 3,
            'ADAPTIVE_BLENDING': True,
            'COLOR_HARMONIZATION': False,
            'LUMINANCE_MATCHING': True,
            'BLEND_TRANSPARENCY': 0.15,
            'COLOR_BLENDING_STRENGTH': 0.3
        },
        'bright': {
            'BLENDING_MODE': 'screen',
            'BLENDING_STRENGTH': 0.7,
            'EDGE_DETECTION_ENABLED': False,
            'EDGE_BLUR_RADIUS': 5,
            'ADAPTIVE_BLENDING': True,
            'COLOR_HARMONIZATION': True,
            'LUMINANCE_MATCHING': False,
            'BLEND_TRANSPARENCY': 0.3,
            'COLOR_BLENDING_STRENGTH': 0.4
        },
        'intense': {
            'BLENDING_MODE': 'hard_light',
            'BLENDING_STRENGTH': 0.9,
            'EDGE_DETECTION_ENABLED': True,
            'EDGE_BLUR_RADIUS': 3,
            'ADAPTIVE_BLENDING': False,
            'COLOR_HARMONIZATION': False,
            'LUMINANCE_MATCHING': False,
            'BLEND_TRANSPARENCY': 0.1,
            'COLOR_BLENDING_STRENGTH': 0.25
        },
        'psychedelic': {
            'BLENDING_MODE': 'exclusion',
            'BLENDING_STRENGTH': 0.95,
            'EDGE_DETECTION_ENABLED': True,
            'EDGE_BLUR_RADIUS': 3,
            'ADAPTIVE_BLENDING': False,
            'COLOR_HARMONIZATION': False,
            'LUMINANCE_MATCHING': False,
            'BLEND_TRANSPARENCY': 0.05,
            'COLOR_BLENDING_STRENGTH': 0.1
        },
        'glow': {
            'BLENDING_MODE': 'color_dodge',
            'BLENDING_STRENGTH': 0.75,
            'EDGE_DETECTION_ENABLED': False,
            'EDGE_BLUR_RADIUS': 3,
            'ADAPTIVE_BLENDING': True,
            'COLOR_HARMONIZATION': True,
            'LUMINANCE_MATCHING': False,
            'BLEND_TRANSPARENCY': 0.2,
            'COLOR_BLENDING_STRENGTH': 0.35
        },
        'dark': {
            'BLENDING_MODE': 'color_burn',
            'BLENDING_STRENGTH': 0.8,
            'EDGE_DETECTION_ENABLED': True,
            'EDGE_BLUR_RADIUS': 3,
            'ADAPTIVE_BLENDING': True,
            'COLOR_HARMONIZATION': False,
            'LUMINANCE_MATCHING': True,
            'BLEND_TRANSPARENCY': 0.25,
            'COLOR_BLENDING_STRENGTH': 0.4
        },
        'geometric': {
            'BLENDING_MODE': 'normal',
            'BLENDING_STRENGTH': 1.0,
            'EDGE_DETECTION_ENABLED': True,
            'EDGE_BLUR_RADIUS': 1,
            'ADAPTIVE_BLENDING': False,
            'COLOR_HARMONIZATION': False,
            'LUMINANCE_MATCHING': False,
            'BLEND_TRANSPARENCY': 0.0,
            'COLOR_BLENDING_STRENGTH': 0.0
        }
    }
    
    preset_name = config.BLENDING_PRESET.lower()
    if preset_name in presets:
        preset = presets[preset_name]
        print(f"🎨 Applicando preset blending: {preset_name.upper()}")
        
        # Applica tutti i parametri del preset alla configurazione
        for param_name, param_value in preset.items():
            setattr(config, param_name, param_value)
            
        # Mostra i parametri applicati
        print(f"   ✓ Modalità: {preset['BLENDING_MODE']}")
        print(f"   ✓ Intensità: {preset['BLENDING_STRENGTH']}")
        print(f"   ✓ Bordi sfumati: {preset['EDGE_BLUR_RADIUS']}px")
        if preset['ADAPTIVE_BLENDING']:
            print(f"   ✓ Blending adattivo attivo")
        if preset['COLOR_HARMONIZATION']:
            print(f"   ✓ Armonizzazione colori attiva")
    else:
        print(f"⚠️ Preset '{preset_name}' non trovato! Preset disponibili:")
        print("   🎬 cinematic, 🌟 artistic, 🌙 soft, ⚡ dramatic, ✨ bright")
        print("   🔥 intense, 🌈 psychedelic, 💡 glow, 🖤 dark, 📐 geometric")
        print("   🔧 Usando configurazione manuale...")


def apply_texture_blending(base_image, texture_image, alpha, blending_mode='overlay', mask=None):
    """
    Applica texture su un'immagine con diversi modalità di blending.
    
    Args:
        base_image: Immagine base (BGR)
        texture_image: Texture da applicare (BGR)
        alpha: Opacità texture (0.0-1.0)
        blending_mode: Modalità blending ('normal', 'overlay', 'multiply', 'screen',
                      'soft_light', 'hard_light', 'color_dodge', 'color_burn', 
                      'darken', 'lighten', 'difference', 'exclusion')
        mask: Maschera opzionale per limitare l'applicazione
    """
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
            mask_norm = cv2.merge([mask_norm, mask_norm, mask_norm])
        result = base_float * (1 - mask_norm) + result * mask_norm
    
    # Riconverti a uint8
    return np.clip(result * 255, 0, 255).astype(np.uint8)


def apply_advanced_blending(background_frame, logo_layer, logo_mask, config):
    """
    Applica un blending avanzato configurabile tra la scritta e lo sfondo.
    Supporta diversi modi di blending e opzioni avanzate.
    """
    # Converti tutto in float32 per calcoli precisi
    bg_frame_f = background_frame.astype(np.float32) / 255.0
    logo_layer_f = logo_layer.astype(np.float32) / 255.0
    
    # 1. NUOVO: Crea maschera avanzata con rilevamento bordi
    if config.EDGE_DETECTION_ENABLED:
        # Rileva i bordi del logo per blending selettivo
        logo_edges = cv2.Canny((logo_mask).astype(np.uint8), 50, 150)
        # Espandi i bordi
        kernel = np.ones((config.EDGE_BLUR_RADIUS//3, config.EDGE_BLUR_RADIUS//3), np.uint8)
        logo_edges = cv2.dilate(logo_edges, kernel, iterations=2)
        # Crea maschera sfumata per i bordi
        edge_mask = cv2.GaussianBlur(logo_edges.astype(np.float32), 
                                   (config.EDGE_BLUR_RADIUS, config.EDGE_BLUR_RADIUS), 0) / 255.0
    else:
        edge_mask = np.ones_like(logo_mask.astype(np.float32))
    
    # 2. Crea maschera base del logo
    if config.EDGE_SOFTNESS % 2 == 0:
        edge_softness = config.EDGE_SOFTNESS + 1
    else:
        edge_softness = config.EDGE_SOFTNESS
    
    soft_mask = cv2.GaussianBlur(logo_mask.astype(np.float32), 
                                (edge_softness, edge_softness), 0) / 255.0
    soft_mask_3ch = cv2.merge([soft_mask, soft_mask, soft_mask])
    
    # 3. NUOVO: Adattamento colori e luminanza
    blended_logo = logo_layer_f.copy()
    
    if config.ADAPTIVE_BLENDING:
        # Estrai colori dello sfondo nell'area del logo
        logo_area_mask = logo_mask > 0
        if np.any(logo_area_mask):
            # Calcola colore medio dello sfondo nell'area del logo
            bg_colors_in_logo = bg_frame_f[logo_area_mask]
            avg_bg_color = np.mean(bg_colors_in_logo, axis=0)
            
            if config.COLOR_HARMONIZATION:
                # Armonizza i colori del logo con lo sfondo
                logo_colors = blended_logo[logo_area_mask]
                # Mix tra colore originale del logo e colore medio dello sfondo
                harmonized_colors = logo_colors * (1 - config.COLOR_BLENDING_STRENGTH) + \
                                  avg_bg_color * config.COLOR_BLENDING_STRENGTH
                blended_logo[logo_area_mask] = harmonized_colors
            
            if config.LUMINANCE_MATCHING:
                # Adatta la luminosità del logo alla luminosità locale dello sfondo
                logo_luminance = np.dot(blended_logo[..., :3], [0.299, 0.587, 0.114])
                bg_luminance = np.dot(bg_frame_f[..., :3], [0.299, 0.587, 0.114])
                
                # Calcola fattore di correzione luminanza
                avg_bg_luminance = np.mean(bg_luminance[logo_area_mask])
                avg_logo_luminance = np.mean(logo_luminance[logo_area_mask])
                
                if avg_logo_luminance > 0:
                    luminance_factor = avg_bg_luminance / avg_logo_luminance
                    # Applica correzione con moderazione
                    correction_strength = 0.5
                    blended_logo[logo_area_mask] *= (1 - correction_strength + correction_strength * luminance_factor)
    
    # 4. NUOVO: Applica modalità di blending configurabile
    def apply_blend_mode(base, blend, mode, strength):
        """Applica diverse modalità di blending"""
        base = np.clip(base, 0, 1)
        blend = np.clip(blend, 0, 1)
        
        if mode == 'normal':
            result = blend
        elif mode == 'multiply':
            result = base * blend
        elif mode == 'screen':
            result = 1 - (1 - base) * (1 - blend)
        elif mode == 'overlay':
            result = np.where(base < 0.5, 
                            2 * base * blend, 
                            1 - 2 * (1 - base) * (1 - blend))
        elif mode == 'soft_light':
            result = np.where(blend < 0.5,
                            base - (1 - 2 * blend) * base * (1 - base),
                            base + (2 * blend - 1) * (np.sqrt(base) - base))
        elif mode == 'hard_light':
            result = np.where(blend < 0.5,
                            2 * base * blend,
                            1 - 2 * (1 - base) * (1 - blend))
        elif mode == 'color_dodge':
            result = np.where(blend >= 1, 1, np.minimum(1, base / (1 - blend + 1e-6)))
        elif mode == 'color_burn':
            result = np.where(blend <= 0, 0, 1 - np.minimum(1, (1 - base) / (blend + 1e-6)))
        elif mode == 'difference':
            result = np.abs(base - blend)
        elif mode == 'exclusion':
            result = base + blend - 2 * base * blend
        else:
            result = blend  # Fallback to normal
        
        # Applica la forza del blending
        return base * (1 - strength) + result * strength
    
    # 5. Applica il blending nelle aree appropriate
    logo_area_mask_3ch = soft_mask_3ch > 0.1
    
    # Blending principale
    bg_in_logo_area = bg_frame_f * soft_mask_3ch
    blended_result = apply_blend_mode(bg_in_logo_area, blended_logo * soft_mask_3ch, 
                                    config.BLENDING_MODE, config.BLENDING_STRENGTH)
    
    # 6. Composizione finale
    # Applica trasparenza se configurata
    if config.BLEND_TRANSPARENCY > 0:
        alpha = 1.0 - config.BLEND_TRANSPARENCY
        blended_result = blended_result * alpha + bg_frame_f * soft_mask_3ch * config.BLEND_TRANSPARENCY
    
    # Combina con lo sfondo
    final_result = bg_frame_f * (1 - soft_mask_3ch) + blended_result
    
    # Gestione bordi con edge mask se abilitata
    if config.EDGE_DETECTION_ENABLED:
        edge_mask_3ch = cv2.merge([edge_mask, edge_mask, edge_mask])
        # Blending più intenso sui bordi
        edge_blended = apply_blend_mode(bg_frame_f, logo_layer_f, 
                                      config.BLENDING_MODE, config.BLENDING_STRENGTH * 1.5)
        final_result = final_result * (1 - edge_mask_3ch) + edge_blended * edge_mask_3ch * soft_mask_3ch
    
    # Riconverti a uint8 e ritorna
    return np.clip(final_result * 255, 0, 255).astype(np.uint8)


def print_blending_options():
    """
    Stampa tutte le opzioni di blending disponibili con descrizioni
    """
    print("\n=== MODALITÀ DI BLENDING DISPONIBILI ===")
    blending_modes = {
        'normal': 'Blending normale - mostra il logo sopra lo sfondo',
        'multiply': 'Moltiplica i colori - effetto scuro e saturo',
        'screen': 'Schiarisce i colori - effetto luminoso',
        'overlay': 'Combina multiply e screen - mantiene contrasto',
        'soft_light': 'Luce soffusa - effetto sottile e naturale',
        'hard_light': 'Luce dura - effetto intenso',
        'color_dodge': 'Schiarisce basandosi sui colori del logo',
        'color_burn': 'Scurisce basandosi sui colori del logo',
        'difference': 'Differenza tra i colori - effetto artistico',
        'exclusion': 'Esclusione - simile a difference ma più soft'
    }
    
    for mode, description in blending_modes.items():
        print(f"  • {mode:12} - {description}")
    
    print("\n=== PARAMETRI CONFIGURABILI ===")
    params = {
        'BLENDING_MODE': 'Scegli una delle modalità sopra',
        'BLENDING_STRENGTH': 'Intensità del blending (0.0-1.0)',
        'EDGE_DETECTION_ENABLED': 'Rileva i bordi per blending selettivo',
        'EDGE_BLUR_RADIUS': 'Raggio sfumatura bordi (numero dispari)',
        'ADAPTIVE_BLENDING': 'Adatta il logo ai colori dello sfondo',
        'COLOR_HARMONIZATION': 'Armonizza i colori logo-sfondo',
        'LUMINANCE_MATCHING': 'Adatta la luminosità del logo',
        'COLOR_BLENDING_STRENGTH': 'Forza mescolamento colori (0.0-1.0)',
        'BLEND_TRANSPARENCY': 'Trasparenza globale logo (0.0-1.0)',
        'LOGO_BLEND_FACTOR': 'Bilanciamento logo originale/blended'
    }
    
    for param, description in params.items():
        print(f"  • {param:25} - {description}")
    
    print("\n=== SUGGERIMENTI PER ESPERIMENTI ===")
    suggestions = [
        "Per effetto cinematografico: overlay + ADAPTIVE_BLENDING=True",
        "Per logo integrato: soft_light + COLOR_HARMONIZATION=True", 
        "Per effetto artistico: difference + EDGE_DETECTION_ENABLED=True",
        "Per logo sottile: screen + BLEND_TRANSPARENCY=0.3",
        "Per effetto drammatico: multiply + LUMINANCE_MATCHING=True"
    ]
    
    for i, suggestion in enumerate(suggestions, 1):
        print(f"  {i}. {suggestion}")
    
    print()


def load_texture(texture_path, width, height):
    """Carica e ridimensiona immagine di texture."""
    if not os.path.exists(texture_path):
        print(f"ATTENZIONE: File texture non trovato in '{texture_path}'. Il logo non verrà texturizzato.")
        return None
    try:
        print("Analisi texture fornita da TV Int dalle acque del Natisone... completata.")
        print("Texture infusa con l'essenza digitale del team creativo.")
        texture = cv2.imread(texture_path, cv2.IMREAD_COLOR)
        if texture is None:
            raise Exception("cv2.imread ha restituito None.")
        # Ridimensiona la texture per adattarla al frame
        return cv2.resize(texture, (width, height), interpolation=cv2.INTER_LINEAR)
    except Exception as e:
        print(f"Errore durante il caricamento della texture: {e}")
        return None


def find_texture_file(config):
    """
    Cerca automaticamente un file texture con priorità: texture.tif > texture.png > texture.jpg
    Se non trova nessuno, usa il fallback configurato.
    """
    base_path = 'input/texture'
    extensions = ['tif', 'png', 'jpg', 'jpeg']
    
    # Cerca con priorità
    for ext in extensions:
        texture_path = f"{base_path}.{ext}"
        if os.path.exists(texture_path):
            print(f"🎨 Texture trovata: {texture_path}")
            return texture_path
    
    # Se non trova nessuna texture.*, usa il fallback
    if os.path.exists(config.TEXTURE_FALLBACK_PATH):
        print(f"🎨 Uso texture fallback: {config.TEXTURE_FALLBACK_PATH}")
        return config.TEXTURE_FALLBACK_PATH
    
    print(f"⚠️ Nessuna texture trovata, il logo non sarà texturizzato")
    return None


def validate_blending_config(config):
    """
    🔧 Valida e imposta valori di default per la configurazione di blending.
    
    Args:
        config: Oggetto configurazione da validare
    
    Returns:
        bool: True se la configurazione è valida
    """
    required_attrs = [
        'BLENDING_MODE',
        'BLENDING_STRENGTH',
        'EDGE_SOFTNESS'
    ]
    
    missing_attrs = []
    for attr in required_attrs:
        if not hasattr(config, attr):
            missing_attrs.append(attr)
    
    if missing_attrs:
        print(f"⚠️ Attributi di configurazione blending mancanti: {missing_attrs}")
        return False
    
    # Valida i valori
    valid_modes = ['normal', 'multiply', 'screen', 'overlay', 'soft_light', 'hard_light', 
                   'color_dodge', 'color_burn', 'difference', 'exclusion']
    
    if config.BLENDING_MODE not in valid_modes:
        print(f"⚠️ BLENDING_MODE '{config.BLENDING_MODE}' non valida. Valori supportati: {valid_modes}")
        return False
        
    if not (0.0 <= config.BLENDING_STRENGTH <= 1.0):
        print("⚠️ BLENDING_STRENGTH deve essere tra 0.0 e 1.0")
        return False
        
    if config.EDGE_SOFTNESS < 1:
        print("⚠️ EDGE_SOFTNESS deve essere almeno 1")
        return False
    
    print("✅ Configurazione blending validata")
    return True


def load_texture_wrapper(texture_path, width, height):
    """Wrapper per la funzione load_texture"""
    return load_texture(texture_path, width, height)


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
