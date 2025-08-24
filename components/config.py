"""
🔧 CONFIGURATION MANAGEMENT - Crystal Therapy
Sistema di gestione configurazione centralizzato

Funzionalità:
- Caricamento parametri dal file config
- Gestione valori di default
- Validazione e conversione tipi
- Sistema di override parametri
"""

import os
from datetime import datetime


def setup_config_defaults(Config):
    """Imposta i valori di default per tutti i parametri di configurazione"""
    # --- PARAMETRI BASE ---
    Config.TEST_MODE = False
    Config.PREVIEW_MODE = False
    Config.FAST_PREVIEW = True
    
    # --- FORMATO E DIMENSIONI ---
    Config.VIDEO_FORMAT = "IG_POST"
    Config.WIDTH = 1080
    Config.HEIGHT = 1350
    
    # --- COMPATIBILITÀ ---
    Config.WHATSAPP_COMPATIBLE = True
    Config.CREATE_WHATSAPP_VERSION = True
    
    # --- SORGENTI ---
    Config.USE_SVG_SOURCE = False
    Config.SVG_PATH = "input/logo.svg"
    Config.PDF_PATH = "input/logo.pdf"
    Config.SVG_LEFT_PADDING = 50
    
    # --- TEXTURE ---
    Config.TEXTURE_AUTO_SEARCH = True
    Config.TEXTURE_FALLBACK_PATH = "input/texture.jpg"
    Config.TEXTURE_ENABLED = True
    Config.TEXTURE_TARGET = "logo"
    Config.TEXTURE_ALPHA = 0.6
    Config.TEXTURE_BACKGROUND_ALPHA = 0.5
    Config.TEXTURE_BLENDING_MODE = "multiply"
    
    # --- VIDEO PARAMETRI ---
    Config.SVG_PADDING = 5
    Config.FPS = 20
    Config.DURATION_SECONDS = 3
    Config.TOTAL_FRAMES = Config.DURATION_SECONDS * Config.FPS
    
    # --- COLORI E STILE ---
    Config.LOGO_COLOR_B = 255
    Config.LOGO_COLOR_G = 255
    Config.LOGO_COLOR_R = 255
    Config.LOGO_COLOR = (Config.LOGO_COLOR_B, Config.LOGO_COLOR_G, Config.LOGO_COLOR_R)
    Config.LOGO_ALPHA = 0.8
    Config.LOGO_ZOOM_FACTOR = 1.0
    
    # --- SFONDO VIDEO ---
    Config.BACKGROUND_VIDEO_PATH = "input/sfondo.MOV"
    Config.BG_USE_ORIGINAL_SIZE = True
    Config.BG_ZOOM_FACTOR = 1.1
    Config.BG_SLOWDOWN_FACTOR = 2.0
    Config.BG_DARKEN_FACTOR = 0.001
    Config.BG_CONTRAST_FACTOR = 1
    Config.BG_RANDOM_START = True
    
    # --- CROP VIDEO ---
    Config.BG_CROP_Y_START = 0.0
    Config.BG_CROP_X_START = 0.0
    Config.BG_CROP_WIDTH_RATIO = 1.0
    Config.BG_CROP_HEIGHT_RATIO = 0.5
    
    # --- AUDIO ---
    Config.AUDIO_ENABLED = True
    Config.AUDIO_FILES = "input/audio1.aif,input/audio2.aif"
    Config.AUDIO_RANDOM_SELECTION = True
    Config.AUDIO_RANDOM_START = True
    Config.AUDIO_REACTIVE_LENSES = True
    Config.AUDIO_BASS_SENSITIVITY = 0.5
    Config.AUDIO_MID_SENSITIVITY = 0.3
    Config.AUDIO_HIGH_SENSITIVITY = 0.25
    Config.AUDIO_SMOOTHING = 0.5
    Config.AUDIO_BOOST_FACTOR = 4.0
    Config.AUDIO_SPEED_INFLUENCE = 1.0
    Config.AUDIO_STRENGTH_INFLUENCE = 2
    Config.AUDIO_PULSATION_INFLUENCE = 1.3
    
    # --- GLOW ---
    Config.GLOW_ENABLED = False
    Config.GLOW_KERNEL_SIZE = 30
    Config.GLOW_INTENSITY = 0.5
    
    # --- DEFORMAZIONE ORGANICA ---
    Config.DEFORMATION_ENABLED = False
    Config.DEFORMATION_SPEED = 0.02
    Config.DEFORMATION_SCALE = 0.005
    Config.DEFORMATION_INTENSITY = 15.0
    Config.DEFORMATION_AUDIO_REACTIVE = True
    Config.DEFORMATION_BASS_INTENSITY = 0.22
    Config.DEFORMATION_BASS_SPEED = 0.03
    Config.DEFORMATION_MID_SCALE = 0.002
    Config.DEFORMATION_SMOOTHING = 0.95
    Config.DEFORMATION_AUDIO_MULTIPLIER = 1.7
    
    # --- LENTI ---
    Config.LENS_DEFORMATION_ENABLED = False
    Config.NUM_LENSES = 25
    Config.LENS_MIN_STRENGTH = -1.2
    Config.LENS_MAX_STRENGTH = 1.5
    Config.LENS_MIN_RADIUS = 5
    Config.LENS_MAX_RADIUS = 55
    Config.LENS_SPEED_FACTOR = 0.1
    Config.LENS_PATH_SPEED_MULTIPLIER = 0.1
    Config.LENS_BASE_SPEED_MULTIPLIER = 0.1
    Config.LENS_ROTATION_SPEED_MULTIPLIER = 0.01
    Config.LENS_INERTIA = 0.95
    Config.LENS_ROTATION_SPEED_MIN = -0.02
    Config.LENS_ROTATION_SPEED_MAX = 0.02
    Config.LENS_HORIZONTAL_BIAS = 2
    Config.LENS_PULSATION_ENABLED = False
    Config.LENS_PULSATION_SPEED = 0.0005
    Config.LENS_PULSATION_AMPLITUDE = 0.2
    Config.LENS_FORCE_PULSATION_ENABLED = True
    Config.LENS_FORCE_PULSATION_AMPLITUDE = 0.2
    Config.WORM_SHAPE_ENABLED = False
    Config.WORM_LENGTH = 1.8
    Config.WORM_COMPLEXITY = 5
    
    # --- SMOOTHING ---
    Config.SMOOTHING_ENABLED = True
    Config.SMOOTHING_FACTOR = 0.0001
    
    # --- TRACCIANTI ---
    Config.TRACER_ENABLED = False
    Config.TRACER_TRAIL_LENGTH = 45
    Config.TRACER_MAX_OPACITY = 0.26
    Config.TRACER_BASE_COLOR_B = 255
    Config.TRACER_BASE_COLOR_G = 200
    Config.TRACER_BASE_COLOR_R = 220
    Config.TRACER_THRESHOLD1 = 50
    Config.TRACER_THRESHOLD2 = 200
    
    # --- TRACCIANTI SFONDO ---
    Config.BG_TRACER_ENABLED = False
    Config.BG_TRACER_TRAIL_LENGTH = 45
    Config.BG_TRACER_MAX_OPACITY = 0.2
    Config.BG_TRACER_BASE_COLOR_B = 255
    Config.BG_TRACER_BASE_COLOR_G = 200
    Config.BG_TRACER_BASE_COLOR_R = 220
    Config.BG_TRACER_THRESHOLD1 = 20
    Config.BG_TRACER_THRESHOLD2 = 100
    
    # --- BLENDING ---
    Config.ADVANCED_BLENDING = False
    Config.BLENDING_PRESET = "manual"
    Config.BLENDING_MODE = "overlay"
    Config.BLENDING_STRENGTH = 1.0
    Config.EDGE_DETECTION_ENABLED = False
    Config.EDGE_BLUR_RADIUS = 1
    Config.ADAPTIVE_BLENDING = True
    Config.COLOR_HARMONIZATION = True
    Config.LUMINANCE_MATCHING = True
    Config.LOGO_BLEND_FACTOR = 0.5
    Config.EDGE_SOFTNESS = 0
    Config.BLEND_TRANSPARENCY = 0.7
    Config.COLOR_BLENDING_STRENGTH = 0.8
    
    # --- DEBUG ---
    Config.DEBUG_MASK = False
    
    # --- VARIAZIONE DINAMICA ---
    Config.DYNAMIC_VARIATION_ENABLED = False
    Config.VARIATION_AMPLITUDE = 0.8
    Config.VARIATION_SPEED_SLOW = 0.01
    Config.VARIATION_SPEED_MEDIUM = 0.025
    Config.VARIATION_SPEED_FAST = 0.005


def load_config_from_file(Config):
    """Carica i parametri dal file config se esiste"""
    # Prima imposta i valori di default
    setup_config_defaults(Config)
    
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
                        if key in ['AUDIO_FILES']:
                            # Lista separata da virgole
                            setattr(Config, key, value)
                        elif key in ['VIDEO_FORMAT', 'SVG_PATH', 'PDF_PATH', 'TEXTURE_FALLBACK_PATH', 
                                   'TEXTURE_TARGET', 'TEXTURE_BLENDING_MODE', 'BACKGROUND_VIDEO_PATH',
                                   'BLENDING_PRESET', 'BLENDING_MODE']:
                            # Stringhe
                            setattr(Config, key, value)
                        elif value.lower() in ['true', 'false']:
                            # Boolean
                            setattr(Config, key, value.lower() == 'true')
                        elif '.' in value:
                            # Float
                            setattr(Config, key, float(value))
                        else:
                            # Integer
                            setattr(Config, key, int(value))
                        
                        # Debug per parametri sconosciuti
                        if not hasattr(Config, key):
                            print(f"⚠️  Parametro sconosciuto '{key}' alla riga {line_num}")
                        
                    except ValueError as e:
                        print(f"⚠️  Errore nel parsing della riga {line_num}: {e}")
                        print(f"     Contenuto: {line}")
                        continue
    
    except FileNotFoundError:
        print("⚠️  File config non trovato, uso valori di default")
    except Exception as e:
        print(f"⚠️  Errore nel caricamento del config: {e}")
    
    # Aggiorna parametri dipendenti
    _update_dependent_params(Config)
    
    print("✅ Configurazione caricata dal file config")


def _update_dependent_params(Config):
    """Aggiorna parametri che dipendono da altri"""
    # Aggiorna TOTAL_FRAMES basato su DURATION_SECONDS e FPS
    Config.TOTAL_FRAMES = Config.DURATION_SECONDS * Config.FPS
    
    # Aggiorna LOGO_COLOR tuple
    Config.LOGO_COLOR = (Config.LOGO_COLOR_B, Config.LOGO_COLOR_G, Config.LOGO_COLOR_R)
    
    # Calcola dimensioni basate su VIDEO_FORMAT
    if Config.VIDEO_FORMAT == "IG_STORY":
        Config.WIDTH = 1080
        Config.HEIGHT = 1920  # 9:16
    elif Config.VIDEO_FORMAT == "IG_POST":
        Config.WIDTH = 1080
        Config.HEIGHT = 1350  # 4:5
    elif Config.VIDEO_FORMAT == "INPUT_VIDEO_SIZE":
        # Mantiene le dimensioni originali (da SVG o altro)
        pass
    
    # Adatta dimensioni per TEST_MODE
    if Config.TEST_MODE:
        Config.WIDTH = int(Config.WIDTH * 0.4)  # Riduce del 60%
        Config.HEIGHT = int(Config.HEIGHT * 0.4)
        Config.FPS = min(Config.FPS, 10)  # Max 10 FPS in test
        Config.DURATION_SECONDS = min(Config.DURATION_SECONDS, 4)  # Max 4 secondi in test
        Config.TOTAL_FRAMES = Config.DURATION_SECONDS * Config.FPS


def validate_config(Config):
    """Valida la configurazione e mostra avvertimenti se necessario"""
    warnings = []
    
    # Valida FPS
    if Config.FPS > 60:
        warnings.append(f"FPS molto alto ({Config.FPS}) potrebbe causare problemi di performance")
    elif Config.FPS < 5:
        warnings.append(f"FPS molto basso ({Config.FPS}) potrebbe causare video troppo scattosi")
    
    # Valida durata
    if Config.DURATION_SECONDS > 60:
        warnings.append(f"Durata molto lunga ({Config.DURATION_SECONDS}s) aumenterà drasticamente il tempo di rendering")
    
    # Valida dimensioni
    if Config.WIDTH * Config.HEIGHT > 3840 * 2160:  # 4K
        warnings.append(f"Risoluzione molto alta ({Config.WIDTH}x{Config.HEIGHT}) richiederà molta memoria")
    
    # Mostra warnings
    for warning in warnings:
        print(f"⚠️  {warning}")
    
    return len(warnings) == 0


def print_config_summary(Config):
    """Stampa un riassunto della configurazione attuale"""
    print(f"\n📋 CONFIGURAZIONE ATTUALE:")
    print(f"   🎬 Modalità: {'TEST' if Config.TEST_MODE else 'PRODUZIONE'}")
    print(f"   📐 Dimensioni: {Config.WIDTH}x{Config.HEIGHT} ({Config.VIDEO_FORMAT})")
    print(f"   🎞️ Video: {Config.FPS}fps, {Config.DURATION_SECONDS}s ({Config.TOTAL_FRAMES} frame)")
    print(f"   📄 Sorgente: {'SVG' if Config.USE_SVG_SOURCE else 'PDF'}")
    print(f"   🎨 Texture: {'Attiva' if Config.TEXTURE_ENABLED else 'Disattiva'}")
    print(f"   🌊 Deformazioni: {'Attive' if Config.DEFORMATION_ENABLED else 'Disattive'}")
    print(f"   🔍 Lenti: {'Attive' if Config.LENS_DEFORMATION_ENABLED else 'Disattive'}")
    print(f"   ✨ Traccianti: {'Attivi' if Config.TRACER_ENABLED else 'Disattivi'}")
    print(f"   🎵 Audio: {'Attivo' if Config.AUDIO_ENABLED else 'Disattivo'}")
