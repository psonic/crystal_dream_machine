import cv2
import numpy as np
import datetime
from scipy.interpolate import splprep, splev
from noise import pnoise2
import multiprocessing
from functools import partial
import time
import os
from collections import deque
import xml.etree.ElementTree as ET
from svgpathtools import svg2paths2
import re
import subprocess
import sys

# Import condizionale per PDF
try:
    import fitz  # PyMuPDF per PDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("⚠️ PyMuPDF non disponibile, solo modalità SVG")

# CAIROSVG verrà importato solo se necessario
CAIROSVG_AVAILABLE = None

# Disabilita il warning PIL per le immagini ad alta risoluzione
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # Rimuove il limite di sicurezza PIL

# Import per gestione audio (opzionale)
try:
    import librosa
    import librosa.display
    AUDIO_AVAILABLE = True
    print("🎵 Librosa disponibile - Supporto audio attivato!")
except ImportError:
    AUDIO_AVAILABLE = False
    print("⚠️ Librosa non disponibile. Per supporto audio: pip install librosa")

# --- CONFIGURAZIONE GLOBALE ---

class Config:
    # --- Modalità e Qualità ---
    TEST_MODE = True  # Test rapido per verifiche (True = 5 sec, False = durata completa)        

    # --- Formato Video ---
    INSTAGRAM_STORIES_MODE = True    # True = formato verticale 9:16 (1080x1920) per Instagram Stories
                                    # False = formato originale basato su dimensioni SVG

    # --- Compatibilità WhatsApp ---
    WHATSAPP_COMPATIBLE = True   # Ottimizza per WhatsApp/social media
    CREATE_WHATSAPP_VERSION = True  # Crea versione aggiuntiva con ffmpeg
    
    # --- Sorgente Logo e Texture ---
    USE_SVG_SOURCE = True        # True = usa SVG, False = usa PDF
    SVG_PATH = 'input/logo.svg'  # Percorso file SVG
    PDF_PATH = 'input/logo.pdf'  # Percorso file PDF alternativo
    SVG_LEFT_PADDING = 50        # Padding sinistro aggiuntivo per SVG (range: 0-200, 50=standard)
    TEXTURE_AUTO_SEARCH = True   # Cerca automaticamente file texture.*
    TEXTURE_FALLBACK_PATH = 'input/texture.jpg'  # Texture di fallback
    
    # --- Sistema Texture Avanzato ---
    TEXTURE_ENABLED = True       # Attiva sistema texture
    TEXTURE_TARGET = 'background'      # Dove applicare: 'logo', 'background', 'both'
    TEXTURE_ALPHA = 0.1          # Opacità texture logo (range: 0.0-1.0, 0.3=leggera, 0.7=forte)
    TEXTURE_BACKGROUND_ALPHA = 0.1  # Opacità texture sfondo (range: 0.1-0.8, 0.2=sottile, 0.5=visibile)
    # Modalità texture disponibili: 'normal', 'overlay', 'multiply', 'screen', 'soft_light', 'hard_light', 
    # 'color_dodge', 'color_burn', 'darken', 'lighten', 'difference', 'exclusion'
    TEXTURE_BLENDING_MODE = 'lighten'  # Modalità blending texture

    # --- Parametri Video ---
    SVG_PADDING = 20  # Spazio intorno al logo (range: 50-300, ridotto in test mode per velocità)
    FPS = 1 if TEST_MODE else 20  # Frame per secondo (range: 10-60, 24=cinema, 30=standard, 60=fluido)
    DURATION_SECONDS = 4 if TEST_MODE else 10  # Durata video in secondi
    TOTAL_FRAMES = DURATION_SECONDS * FPS     # Frame totali calcolati   

    # --- Colore e Stile ---
    LOGO_COLOR = (255, 255, 255)    # Colore logo BGR (range: 0-255 per canale, (0,0,0)=nero, (255,255,255)=bianco)
    LOGO_ALPHA = 0.7             # Opacità logo (range: 0.0-1.0, 0.5=semitrasparente, 1.0=opaco)
    LOGO_ZOOM_FACTOR = 1.0       # Zoom del logo (range: 0.5-3.0, 1=normale, 1.5=ingrandito, 2=doppio, 0.8=ridotto)
    
    # --- Video di Sfondo ---
    BACKGROUND_VIDEO_PATH = 'input/sfondo.MOV'  # Percorso video di sfondo
    BG_USE_ORIGINAL_SIZE = True  # Usa dimensioni originali video senza crop
    BG_ZOOM_FACTOR = 1.4         # Zoom dello sfondo (range: 0.8-2.5, 1=normale, 1.5=zoomato, 2=molto zoomato)
    BG_SLOWDOWN_FACTOR = 1.0     # Rallentamento sfondo (range: 0.5-3.0, 1=normale, 2=metà velocità, 0.8=più veloce)
    BG_DARKEN_FACTOR = 0.7      # Scurimento sfondo (range: 0.1-1.0, 0.3=scuro, 0.7=normale)
    BG_CONTRAST_FACTOR = 1.0     # Contrasto sfondo (range: 0.5-2.5, 1=normale, 1.5=più contrasto)
    BG_RANDOM_START = True       # Inizia da punto casuale del video (max 2/3 della durata)
    
    # --- Sistema Audio Reattivo ---
    AUDIO_ENABLED = True         # Attiva reattività audio per lenti
    AUDIO_FILES = ['input/audio1.aif', 'input/audio2.aif']  # Lista file audio per selezione casuale
    AUDIO_RANDOM_SELECTION = True  # Seleziona casualmente un file dalla lista
    AUDIO_RANDOM_START = True    # Inizia da punto casuale (max 2/3 del file)
    AUDIO_REACTIVE_LENSES = True # Le lenti reagiscono all'audio
    AUDIO_BASS_SENSITIVITY = 0.5 # Sensibilità alle frequenze basse (range: 0.1-1.0, 0.2=delicato, 0.5=forte)
    AUDIO_MID_SENSITIVITY = 0.3  # Sensibilità alle frequenze medie (range: 0.1-0.8, 0.15=sottile, 0.4=intensa)
    AUDIO_HIGH_SENSITIVITY = 0.25 # Sensibilità alle frequenze acute (range: 0.05-0.5, 0.1=leggero, 0.3=vivace)
    AUDIO_SMOOTHING = 0.5        # Smoothing reattività audio (range: 0.3-0.95, 0.5=reattivo, 0.9=fluido)
    AUDIO_BOOST_FACTOR = 4.0     # Amplificazione reattività (range: 1.0-10.0, 2=normale, 5=estrema)
    
    # --- Parametri Audio Lenti ---
    AUDIO_SPEED_INFLUENCE = 1.0   # Quanto l'audio influenza velocità lenti (range: 0.5-3.0, 1=normale, 2=doppia)
    AUDIO_STRENGTH_INFLUENCE = 2 # Quanto l'audio influenza forza lenti (range: 0.8-2.5, 1=normale, 2=intensa)
    AUDIO_PULSATION_INFLUENCE = 1.3 # Quanto l'audio influenza pulsazione (range: 0.5-2.0, 1=normale, 1.8=estrema)
    
    # --- Effetto Glow ---
    GLOW_ENABLED = True          # Attiva effetto bagliore intorno al logo
    GLOW_KERNEL_SIZE = 50  # Dimensione bagliore (range: 5-200, 25=sottile, 50=normale, 100=molto ampio)
    GLOW_INTENSITY = 0.4         # Intensità bagliore (range: 0.0-1.0, 0.1=tenue, 0.2=normale, 0.5=forte)

    # --- Deformazione Organica ---
    # Questo effetto fa "respirare" il logo creando ondulazioni fluide che lo deformano nel tempo
    DEFORMATION_ENABLED = True  # Attiva movimento ondulatorio del logo
    DEFORMATION_SPEED = 0.01   # Velocità cambio onde (range: 0.01-0.5, 0.05=lento, 0.1=normale, 0.3=veloce)
    DEFORMATION_SCALE = 0.002   # Frequenza onde (range: 0.0005-0.01, 0.001=fini, 0.002=medie, 0.005=larghe)
    DEFORMATION_INTENSITY = 10.0  # Forza deformazione (range: 0.5-20, 2=leggera, 5=normale, 15=estrema)
    
    # --- Reattività Audio Deformazione Organica ---
    DEFORMATION_AUDIO_REACTIVE = True  # Collega deformazione organica all'audio
    DEFORMATION_BASS_INTENSITY = 0.22  # Quanto i bassi influenzano l'intensità (range: 0.1-0.5, 0.15=leggero, 0.3=forte)
    DEFORMATION_BASS_SPEED = 0.03     # Quanto i bassi influenzano la velocità (range: 0.005-0.03, 0.01=lento, 0.02=veloce)
    DEFORMATION_MID_SCALE = 0.002     # Quanto i medi influenzano la scala/frequenza (range: 0.0005-0.003, 0.001=sottile, 0.002=ampio)
    DEFORMATION_SMOOTHING = 0.85       # Smoothing per effetto rimbalzo (range: 0.6-0.95, 0.7=veloce, 0.9=lento)
    DEFORMATION_AUDIO_MULTIPLIER = 1.4 # Moltiplicatore globale audio deformazione (range: 0.5-2.0, 1=normale, 1.5=intenso)

    # --- Deformazione a Lenti ---
    LENS_DEFORMATION_ENABLED = True  # Attiva effetto lenti che distorcono il logo
    NUM_LENSES = 70             # Numero di lenti (range: 5-100, 20=poche, 40=normale, 80=molte)
    LENS_MIN_STRENGTH = -1.2     # Forza minima ridotta per deformazione più delicata
    LENS_MAX_STRENGTH = 1.5      # Forza massima ridotta per deformazione più delicata
    LENS_MIN_RADIUS = 2         # Raggio minimo area influenza (range: 5-50, 10=piccola, 30=grande)
    LENS_MAX_RADIUS = 35         # Raggio massimo area influenza (range: 20-150, 50=media, 100=ampia)
    LENS_SPEED_FACTOR = 0.1    # Velocità movimento (range: 0.005-0.1, 0.01=lenta, 0.05=veloce)
    
    # --- Parametri Movimento Lenti ---
    LENS_PATH_SPEED_MULTIPLIER = 0.1    # Velocità percorso (range: 1-20, 5=lenta, 10=normale, 15=veloce)
    LENS_BASE_SPEED_MULTIPLIER = 0.1    # Moltiplicatore velocità base (range: 0.5-3, 1=normale, 2=doppia)
    LENS_ROTATION_SPEED_MULTIPLIER = 0.01  # Velocità rotazione verme (range: 1-15, 5=lenta, 10=veloce)
    LENS_INERTIA = 0.95                  # Fluidità movimento (range: 0.1-0.95, 0.3=scattoso, 0.9=fluido)
    LENS_ROTATION_SPEED_MIN = -0.02     # Velocità rotazione minima (range: -0.02 a 0)
    LENS_ROTATION_SPEED_MAX = 0.02     # Velocità rotazione massima (range: 0 a 0.02)

    # --- Movimento e Pulsazione Lenti ---
    LENS_HORIZONTAL_BIAS = 2             # Preferenza movimento orizzontale (range: 1-5, 1=uniforme, 3=bias, 5=solo orizzontale)
    LENS_PULSATION_ENABLED = True        # Attiva pulsazione dimensioni lenti
    LENS_PULSATION_SPEED = 0.0005         # Velocità pulsazione (range: 0.001-0.02, 0.003=lenta, 0.01=veloce)
    LENS_PULSATION_AMPLITUDE = 0.2       # Ampiezza pulsazione dimensioni (range: 0.1-0.8, 0.2=leggera, 0.5=forte)
    LENS_FORCE_PULSATION_ENABLED = True  # Attiva pulsazione anche della forza
    LENS_FORCE_PULSATION_AMPLITUDE = 0.2 # Ampiezza pulsazione forza (range: 0.1-0.5, 0.2=normale, 0.4=estrema)
    
    WORM_SHAPE_ENABLED = True  # Forma allungata delle lenti (tipo verme)
    WORM_LENGTH = 1.8          # Lunghezza forma verme (range: 1.5-4, 2=normale, 3=lungo)
    WORM_COMPLEXITY = 5        # Complessità movimento verme (range: 1-8, 2=semplice, 6=complesso)

    # --- Smussamento Contorni ---
    SMOOTHING_ENABLED = True      # Attiva bordi lisci del logo
    SMOOTHING_FACTOR = 0.0001     # Intensità smussamento (range: 0.00001-0.01, 0.0001=leggero, 0.001=forte)

    # --- Traccianti Logo ---
    TRACER_ENABLED = True            # Attiva scie colorate sui bordi del logo
    TRACER_TRAIL_LENGTH = 25  # Lunghezza scie (ridotta in test mode per velocità)
    TRACER_MAX_OPACITY = 0.05        # Opacità massima scie (range: 0.01-0.2, 0.02=sottili, 0.05=visibili, 0.1=forti)
    TRACER_BASE_COLOR = (255, 200, 220)  # Colore base scie (BGR: 0-255 per ogni canale)
    TRACER_THRESHOLD1 = 50           # Soglia bassa rilevamento bordi (range: 20-100, 30=sensibile, 70=selettivo)
    TRACER_THRESHOLD2 = 200          # Soglia alta rilevamento bordi (range: 100-500, 200=normale, 400=rigido)
    
    # --- Traccianti Sfondo ---
    BG_TRACER_ENABLED = True         # Attiva scie sui contorni dello sfondo
    BG_TRACER_TRAIL_LENGTH = 25  # Lunghezza scie sfondo (ridotta in test mode)
    BG_TRACER_MAX_OPACITY = 0.03     # Opacità scie sfondo (range: 0.005-0.1, 0.02=sottili, 0.06=evidenti)
    BG_TRACER_BASE_COLOR = (200, 170, 200)  # Colore scie sfondo (BGR: tonalità viola/magenta)
    BG_TRACER_THRESHOLD1 = 50        # Soglia bassa contorni sfondo (range: 10-80, 20=tutto, 50=selettivo)
    BG_TRACER_THRESHOLD2 = 100       # Soglia alta contorni sfondo (range: 50-200, 80=normale, 150=rigido)
    
    # --- Blending Avanzato ---
    ADVANCED_BLENDING = True  # Attiva fusione avanzata logo-sfondo
    
    # 🎨 SISTEMA PRESET AUTOMATICO
    # Preset disponibili: 'manual', 'cinematic', 'artistic', 'soft', 'dramatic', 'bright', 'intense', 'psychedelic', 'glow', 'dark', 'geometric'
    BLENDING_PRESET = 'glow'  # Usa 'manual' per configurazione manuale sotto
    
    # Parametri blending configurabili (usati solo se BLENDING_PRESET = 'manual')
    # Modalità disponibili: 'normal', 'multiply', 'screen', 'overlay', 'soft_light', 'hard_light', 'color_dodge', 'color_burn', 'darken', 'lighten', 'difference', 'exclusion'
    BLENDING_MODE = 'color_burn'     # Modalità fusione logo-sfondo
    BLENDING_STRENGTH = 0.7          # Intensità fusione (range: 0.0-1.0, 0.3=leggera, 0.7=forte, 1.0=solo effetto)
    EDGE_DETECTION_ENABLED = True    # Rileva bordi per fusione selettiva
    EDGE_BLUR_RADIUS = 21            # Raggio sfumatura bordi (range: 5-50, 15=netti, 25=morbidi, 40=molto sfumati)
    ADAPTIVE_BLENDING = False        # Adatta fusione ai colori dello sfondo
    COLOR_HARMONIZATION = False      # Armonizza colori logo-sfondo
    LUMINANCE_MATCHING = False       # Adatta luminosità logo allo sfondo
    
    # Parametri fusione classici
    LOGO_BLEND_FACTOR = 0.6          # Bilanciamento logo/fuso (range: 0.0-1.0, 0.3=più logo, 0.8=più effetto)
    EDGE_SOFTNESS = 80               # Morbidezza bordi (range: 20-150, 50=netti, 100=morbidi)
    BLEND_TRANSPARENCY = 0.3         # Trasparenza globale (range: 0.0-0.8, 0.2=opaco, 0.5=trasparente)
    COLOR_BLENDING_STRENGTH = 0.3    # Intensità mescolamento colori (range: 0.0-1.0, 0.2=leggero, 0.6=forte)
    
    # --- Debug e Qualità ---
    DEBUG_MASK = False  # Mostra maschera di debug (per sviluppatori)
    
    # --- Variazione Dinamica ---
    DYNAMIC_VARIATION_ENABLED = True  # Attiva variazioni automatiche nel tempo
    VARIATION_AMPLITUDE = 0.8         # Ampiezza variazioni (range: 0.2-2.0, 0.5=leggere, 1.0=normali, 1.5=forti)
    VARIATION_SPEED_SLOW = 0.01       # Velocità variazioni lente (range: 0.005-0.05, 0.01=molto lente, 0.03=moderate)
    VARIATION_SPEED_MEDIUM = 0.025    # Velocità variazioni medie (range: 0.01-0.1, 0.02=normali, 0.05=veloci)
    VARIATION_SPEED_FAST = 0.005      # Velocità variazioni veloci (range: 0.002-0.02, 0.005=rapide, 0.015=frenetiche)

# --- FUNZIONI DI SUPPORTO ---

def get_dynamic_parameters(frame_index, total_frames):
    """
    Calcola parametri che cambiano automaticamente nel tempo per creare variazioni.
    """
    t = frame_index / total_frames  # Progresso animazione (0.0 a 1.0)
    params = {}

    # Pulsazione del glow
    glow_pulse = np.sin(t * np.pi)
    params['glow_intensity'] = Config.GLOW_INTENSITY + (glow_pulse * 0.2)

    # Variazioni automatiche dei parametri principali
    if Config.DYNAMIC_VARIATION_ENABLED:
        base_seed = frame_index * 0.001
        
        # Variazioni lente per deformazioni organiche
        deform_var_x = np.sin(base_seed * Config.VARIATION_SPEED_SLOW + 1.0) * Config.VARIATION_AMPLITUDE
        deform_var_y = np.cos(base_seed * Config.VARIATION_SPEED_SLOW + 2.5) * Config.VARIATION_AMPLITUDE
        deform_var_z = np.sin(base_seed * Config.VARIATION_SPEED_SLOW + 4.0) * Config.VARIATION_AMPLITUDE
        
        params['deformation_speed'] = Config.DEFORMATION_SPEED * (1.0 + deform_var_x)
        params['deformation_scale'] = Config.DEFORMATION_SCALE * (1.0 + deform_var_y)
        params['deformation_intensity'] = Config.DEFORMATION_INTENSITY * (1.0 + deform_var_z)
        
        # Variazioni medie per lenti
        lens_var_x = np.sin(base_seed * Config.VARIATION_SPEED_MEDIUM + 3.0) * Config.VARIATION_AMPLITUDE
        lens_var_y = np.cos(base_seed * Config.VARIATION_SPEED_MEDIUM + 5.5) * Config.VARIATION_AMPLITUDE
        
        params['lens_speed_factor'] = Config.LENS_SPEED_FACTOR * (1.0 + lens_var_x)
        params['lens_strength_multiplier'] = 1.0 + lens_var_y
        
        # Variazioni veloci per traccianti
        tracer_var_x = np.sin(base_seed * Config.VARIATION_SPEED_FAST + 2.0) * Config.VARIATION_AMPLITUDE
        tracer_var_y = np.cos(base_seed * Config.VARIATION_SPEED_FAST + 6.0) * Config.VARIATION_AMPLITUDE
        
        params['tracer_opacity_multiplier'] = 1.0 + tracer_var_x
        params['bg_tracer_opacity_multiplier'] = 1.0 + tracer_var_y
    else:
        # Usa valori fissi se le variazioni sono disabilitate
        params['deformation_speed'] = Config.DEFORMATION_SPEED
        params['deformation_scale'] = Config.DEFORMATION_SCALE
        params['deformation_intensity'] = Config.DEFORMATION_INTENSITY
        params['lens_speed_factor'] = Config.LENS_SPEED_FACTOR
        params['lens_strength_multiplier'] = 1.0
        params['tracer_opacity_multiplier'] = 1.0
        params['bg_tracer_opacity_multiplier'] = 1.0
    
    return params

def get_timestamp_filename():
    """Genera nome file con timestamp e carattere decorativo."""
    now = datetime.datetime.now()
    magic_chars = ['α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'ι', 'κ', 'λ', 'μ', 'ν', 'ξ', 'ο', 'π', 'ρ', 'σ', 'τ', 'υ', 'φ', 'χ', 'ψ', 'ω', 'ॐ', '☯', '✨', 'Δ', 'Σ', 'Ω']
    magic_char = np.random.choice(magic_chars)
    
    # Aggiungi "test" al nome se in modalità test
    test_suffix = "_TEST" if Config.TEST_MODE else ""
    
    return f"output/crystalpy_{now.strftime('%Y%m%d_%H%M%S')}{test_suffix}_{magic_char}.mp4"

# --- Sistema di Smoothing per Effetto Rimbalzo Audio ---
class AudioSmoothingState:
    """Memorizza lo stato per il smoothing dell'audio reattivo con effetto rimbalzo."""
    def __init__(self):
        self.prev_intensity = None
        self.prev_speed = None
        self.prev_scale = None

# Istanza globale per il smoothing
_audio_smoothing_state = AudioSmoothingState()

def add_audio_to_video(video_path, audio_data, duration):
    """
    🎵 Aggiunge l'audio selezionato al video usando ffmpeg.
    
    Args:
        video_path: Percorso del video senza audio
        audio_data: Dati audio che contengono il file selezionato e offset
        duration: Durata del video in secondi
    
    Returns:
        str: Percorso del video finale con audio
    """
    if not audio_data:
        print("🔇 Nessun audio da aggiungere")
        return video_path
    
    # Genera nome del file finale
    base_name = video_path.replace('.mp4', '')
    final_video_path = f"{base_name}_with_audio.mp4"
    
    try:
        # Costruisci comando ffmpeg con parametri corretti
        cmd = [
            'ffmpeg', '-y',  # -y per sovrascrivere senza chiedere
            '-i', video_path,  # Video input
            '-ss', str(audio_data['start_offset']),  # Offset per l'audio
            '-i', audio_data['selected_file'],  # Audio input con offset
            '-t', str(duration),  # Durata del video
            '-c:v', 'copy',  # Copia video senza ricodifica
            '-c:a', 'aac',   # Codifica audio in AAC per compatibilità
            '-map', '0:v:0', # Usa video dal primo input
            '-map', '1:a:0', # Usa audio dal secondo input
            '-shortest',     # Interrompi quando il più corto finisce
            final_video_path
        ]
        
        print(f"🎵 Aggiungendo audio al video...")
        print(f"📂 Audio: {audio_data['selected_file']}")
        print(f"⏯️ Offset: {audio_data['start_offset']:.1f}s")
        print(f"🔧 Comando: {' '.join(cmd)}")  # Debug del comando
        
        # Esegui ffmpeg
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Video con audio creato: {final_video_path}")
            # Verifica che il file sia stato creato correttamente
            if os.path.exists(final_video_path) and os.path.getsize(final_video_path) > 1000:
                # Rimuovi il video temporaneo senza audio
                try:
                    os.remove(video_path)
                    print(f"🗑️ Rimosso video temporaneo: {video_path}")
                except:
                    pass
                return final_video_path
            else:
                print(f"⚠️ File audio generato ma sembra corrotto (dimensione: {os.path.getsize(final_video_path) if os.path.exists(final_video_path) else 0} bytes)")
                return video_path
        else:
            print(f"⚠️ Errore ffmpeg (codice {result.returncode}):")
            print(f"📤 stdout: {result.stdout}")
            print(f"📤 stderr: {result.stderr}")
            print(f"🔇 Mantengo video senza audio: {video_path}")
            return video_path
            
    except Exception as e:
        print(f"⚠️ Errore nell'aggiunta audio: {e}")
        print(f"🔇 Mantengo video senza audio: {video_path}")
        return video_path

def load_audio_analysis(audio_files, duration, fps=30, random_selection=True, random_start=True):
    """
    🎵 Carica e analizza il file audio per l'estrazione delle frequenze.
    Supporta selezione casuale di file e inizio casuale.
    
    Args:
        audio_files: Lista di percorsi dei file audio o singolo percorso
        duration: Durata del video in secondi
        fps: Frame rate del video
        random_selection: Se True, seleziona casualmente un file dalla lista
        random_start: Se True, inizia da un punto casuale (max 2/3 del file)
    
    Returns:
        dict: Contiene i dati audio processati per frame
    """
    
    # Gestisci sia lista che singolo file
    if isinstance(audio_files, str):
        audio_files = [audio_files]
    
    # Filtra solo i file esistenti
    existing_files = [f for f in audio_files if os.path.exists(f)]
    
    if not existing_files:
        print(f"⚠️ Nessun file audio trovato tra: {audio_files}")
        return None
    
    # Selezione del file audio
    if random_selection and len(existing_files) > 1:
        selected_audio = np.random.choice(existing_files)
        print(f"🎲 Selezionato casualmente: {selected_audio}")
    else:
        selected_audio = existing_files[0]
        print(f"🎵 Usando audio: {selected_audio}")
    
    try:
        # Prima carica per ottenere la durata totale del file audio
        y_full, sr = librosa.load(selected_audio)
        full_duration = len(y_full) / sr
        
        # Calcola offset casuale se richiesto
        start_offset = 0
        if random_start and full_duration > duration:
            # Non iniziare oltre i 2/3 del file per evitare silenzio finale
            max_start = min(full_duration - duration, full_duration * 0.67)
            if max_start > 0:
                start_offset = np.random.uniform(0, max_start)
                print(f"🎯 Inizio casuale a {start_offset:.1f}s (file lungo {full_duration:.1f}s)")
        
        # Carica la porzione desiderata
        y, sr = librosa.load(selected_audio, offset=start_offset, duration=duration)
        
        # Calcola lo spettrogramma
        stft = librosa.stft(y, hop_length=int(sr / fps))
        magnitude = np.abs(stft)
        
        # Separazione delle bande di frequenza
        freqs = librosa.fft_frequencies(sr=sr)
        
        # Definizione delle bande (in Hz)
        bass_mask = freqs <= 250
        mid_mask = (freqs > 250) & (freqs <= 4000)
        high_mask = freqs > 4000
        
        # Estrazione dell'energia per ogni banda per frame
        frames = magnitude.shape[1]
        audio_data = {
            'bass': np.mean(magnitude[bass_mask], axis=0),
            'mid': np.mean(magnitude[mid_mask], axis=0),
            'high': np.mean(magnitude[high_mask], axis=0),
            'total': np.mean(magnitude, axis=0),
            'frames': frames,
            'duration': duration,
            'selected_file': selected_audio,
            'start_offset': start_offset
        }
        
        # Normalizzazione dei valori
        for key in ['bass', 'mid', 'high', 'total']:
            if len(audio_data[key]) > 0:
                audio_data[key] = audio_data[key] / np.max(audio_data[key])
        
        print(f"🎵 Audio caricato: {frames} frames, {duration:.1f}s")
        if start_offset > 0:
            print(f"⏯️ Offset: {start_offset:.1f}s -> {start_offset + duration:.1f}s")
        
        return audio_data
        
    except Exception as e:
        print(f"⚠️ Errore nel caricamento audio {selected_audio}: {e}")
        print("🔇 Rendering senza audio reactivity")
        return None

def get_audio_reactive_factors(audio_data, frame_idx, config):
    """
    🎚️ Calcola i fattori di reattività audio per il frame corrente.
    
    Args:
        audio_data: Dati audio preprocessati
        frame_idx: Indice del frame corrente
        config: Configurazione con parametri audio
    
    Returns:
        dict: Fattori per modulare i parametri delle lenti
    """
    if not audio_data or not config.AUDIO_ENABLED:
        return {
            'speed_factor': 1.0,
            'strength_factor': 1.0,
            'pulsation_factor': 1.0
        }
    
    # Assicurati che l'indice del frame sia valido
    audio_frame_idx = min(frame_idx, len(audio_data['bass']) - 1)
    
    if audio_frame_idx < 0:
        audio_frame_idx = 0
    
    # Estrai i valori per il frame corrente
    bass = audio_data['bass'][audio_frame_idx]
    mid = audio_data['mid'][audio_frame_idx]
    high = audio_data['high'][audio_frame_idx]
    total = audio_data['total'][audio_frame_idx]
    
    # Calcola i fattori di modulazione
    factors = {
        'speed_factor': 1.0 + (bass * config.AUDIO_BASS_SENSITIVITY),
        'strength_factor': 1.0 + (mid * config.AUDIO_MID_SENSITIVITY),
        'pulsation_factor': 1.0 + (high * config.AUDIO_HIGH_SENSITIVITY)
    }
    
    # Applica limiti per evitare valori estremi (range ridotto per movimento delicato)
    for key in factors:
        factors[key] = np.clip(factors[key], 0.5, 1.5)
    
    return factors

def get_organic_deformation_factors(audio_data, frame_idx, config):
    """
    🎵 Calcola i parametri dinamici per la deformazione organica basati sull'audio con effetto rimbalzo.
    
    Args:
        audio_data: Dati audio preprocessati
        frame_idx: Indice del frame corrente
        config: Configurazione con parametri audio
    
    Returns:
        dict: Parametri dinamici per la deformazione organica (o None se audio disabilitato)
    """
    global _audio_smoothing_state
    
    if not audio_data or not config.AUDIO_ENABLED or not config.DEFORMATION_AUDIO_REACTIVE:
        return None
    
    # Assicurati che l'indice del frame sia valido
    audio_frame_idx = min(frame_idx, len(audio_data['bass']) - 1)
    
    if audio_frame_idx < 0:
        audio_frame_idx = 0
    
    # Estrai i valori per il frame corrente
    bass = audio_data['bass'][audio_frame_idx]
    mid = audio_data['mid'][audio_frame_idx]
    high = audio_data['high'][audio_frame_idx]
    
    # Calcola i parametri dinamici raw (in modo delicato)
    raw_intensity = config.DEFORMATION_INTENSITY + (bass * config.DEFORMATION_BASS_INTENSITY)
    raw_speed = config.DEFORMATION_SPEED + (bass * config.DEFORMATION_BASS_SPEED)
    raw_scale = config.DEFORMATION_SCALE + (mid * config.DEFORMATION_MID_SCALE)
    
    # Applica smoothing con effetto rimbalzo per movimento più fluido
    smoothing = config.DEFORMATION_SMOOTHING
    
    # Inizializza valori precedenti se necessario
    if _audio_smoothing_state.prev_intensity is None:
        _audio_smoothing_state.prev_intensity = raw_intensity
        _audio_smoothing_state.prev_speed = raw_speed
        _audio_smoothing_state.prev_scale = raw_scale
    
    # Applica smoothing con interpolazione lineare per effetto rimbalzo
    smoothed_intensity = _audio_smoothing_state.prev_intensity * smoothing + raw_intensity * (1.0 - smoothing)
    smoothed_speed = _audio_smoothing_state.prev_speed * smoothing + raw_speed * (1.0 - smoothing)
    smoothed_scale = _audio_smoothing_state.prev_scale * smoothing + raw_scale * (1.0 - smoothing)
    
    # Memorizza per il prossimo frame
    _audio_smoothing_state.prev_intensity = smoothed_intensity
    _audio_smoothing_state.prev_speed = smoothed_speed
    _audio_smoothing_state.prev_scale = smoothed_scale
    
    dynamic_params = {
        'deformation_intensity': smoothed_intensity,
        'deformation_speed': smoothed_speed,
        'deformation_scale': smoothed_scale
    }
    
    # Applica limiti per mantenere valori ragionevoli (con range leggermente più ampio)
    dynamic_params['deformation_intensity'] = np.clip(dynamic_params['deformation_intensity'], 
                                                    config.DEFORMATION_INTENSITY * 0.6, 
                                                    config.DEFORMATION_INTENSITY * 1.4)
    dynamic_params['deformation_speed'] = np.clip(dynamic_params['deformation_speed'], 
                                                config.DEFORMATION_SPEED * 0.7, 
                                                config.DEFORMATION_SPEED * 1.5)
    dynamic_params['deformation_scale'] = np.clip(dynamic_params['deformation_scale'], 
                                                config.DEFORMATION_SCALE * 0.8, 
                                                config.DEFORMATION_SCALE * 1.3)
    
    return dynamic_params

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
            mask_norm = cv2.cvtColor(mask_norm, cv2.COLOR_GRAY2BGR)
        result = base_float * (1 - mask_norm) + result * mask_norm
    
    # Converti back a uint8
    return np.clip(result * 255, 0, 255).astype(np.uint8)

def get_svg_dimensions(svg_path):
    """Estrae dimensioni da file SVG."""
    try:
        import xml.etree.ElementTree as ET
        
        tree = ET.parse(svg_path)
        root = tree.getroot()
        
        # Prova a leggere width/height dagli attributi
        width = root.get('width')
        height = root.get('height')
        
        if width and height:
            # Rimuovi unità come 'px' se presenti
            width = float(width.replace('px', '').replace('pt', ''))
            height = float(height.replace('px', '').replace('pt', ''))
            return int(width), int(height)
        
        # Se non ci sono width/height, usa viewBox
        viewbox = root.get('viewBox')
        if viewbox:
            _, _, width, height = map(float, viewbox.split())
            return int(width), int(height)
        
        # Fallback a dimensioni predefinite
        return 1920, 1080
        
    except Exception as e:
        print(f"⚠️ Errore lettura dimensioni SVG: {e}")
        return 1920, 1080  # Fallback

def extract_contours_from_svg(svg_path, width, height, padding, left_padding=0, logo_zoom_factor=1.0):
    """
    Estrae SOLO I CONTORNI/BORDI da un file SVG, senza riempimento.
    Utilizza rasterizzazione + edge detection per ottenere linee precise.
    
    Args:
        left_padding: Padding aggiuntivo dal lato sinistro per SVG
        logo_zoom_factor: Fattore di zoom del logo (1.0=normale, 2.0=doppio, 0.5=metà)
    """
    try:
        print("🎨 Caricamento SVG Crystal Therapy dalle acque del Natisone...")
        
        # Prima prova il metodo con PIL (più compatibile)
        try:
            from PIL import Image as PILImage, ImageDraw
            import xml.etree.ElementTree as ET
            
            # Leggi l'SVG e ottieni le dimensioni
            tree = ET.parse(svg_path)
            root = tree.getroot()
            svg_width = float(root.get('width', 100))
            svg_height = float(root.get('height', 100))
            
            # Scala per rendering ad alta risoluzione
            scale_factor = 4
            render_width = int(svg_width * scale_factor)
            render_height = int(svg_height * scale_factor)
            
            # Crea un'immagine bianca
            pil_img = PILImage.new('RGB', (render_width, render_height), 'white')
            draw = ImageDraw.Draw(pil_img)
            
            # Disegna solo i bordi del testo (non il riempimento)
            # Questo approccio è limitato ma più compatibile
            print("⚠️ Usando metodo semplificato - potrebbero includere riempimenti")
            return extract_contours_from_svg_fallback(svg_path, width, height, padding, left_padding, logo_zoom_factor)
            
        except:
            # Fallback al metodo cairosvg se PIL non funziona
            import cairosvg
            import io
            from PIL import Image as PILImage
        
        # Rasterizza SVG ad alta risoluzione per preservare i dettagli
        scale_factor = 4  # Alta risoluzione per migliore edge detection
        render_width = width * scale_factor
        render_height = height * scale_factor
        
        # Converti SVG in PNG ad alta risoluzione
        png_data = cairosvg.svg2png(
            url=svg_path,
            output_width=render_width,
            output_height=render_height
        )
        
        # Carica l'immagine
        pil_image = PILImage.open(io.BytesIO(png_data))
        img_array = np.array(pil_image)
        
        # Converti RGBA in RGB se necessario
        if img_array.shape[2] == 4:
            # Rimuovi il canale alpha, assume sfondo bianco
            img_rgb = img_array[:,:,:3]
            alpha = img_array[:,:,3] / 255.0
            img_rgb = img_rgb * alpha[:,:,np.newaxis] + 255 * (1 - alpha[:,:,np.newaxis])
            img_array = img_rgb.astype(np.uint8)
        
        # Converti in BGR per OpenCV
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Converti in scala di grigi
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # EDGE DETECTION per ottenere SOLO i contorni/bordi
        # Applica filtro Gaussiano per ridurre il rumore
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Usa Canny edge detection per ottenere solo i bordi
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
        
        # Dilata leggermente i bordi per assicurarsi che siano connessi
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Trova i contorni dei bordi
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            raise Exception("Nessun contorno trovato nell'SVG.")
        
        print(f"📝 Trovati {len(contours)} contorni di bordi...")
        
        # Filtra e processa i contorni
        processed_contours = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            # Filtra contorni troppo piccoli (rumore)
            if area > 100:  # Area minima per essere considerato valido
                # Scala il contorno alla risoluzione target
                scaled_contour = contour.astype(np.float32) / scale_factor
                processed_contours.append(scaled_contour.astype(np.int32))
                print(f"  ✓ Contorno {i+1}: {len(contour)} punti, area: {area/scale_factor/scale_factor:.1f}")
        
        if not processed_contours:
            raise Exception("Nessun contorno valido trovato dopo il filtraggio.")
        
        print(f"📐 Estratti {len(processed_contours)} contorni di BORDI (no riempimento)")
        print("Estrazione contorni da SVG completata.")
        return processed_contours, None
        
    except Exception as e:
        print(f"Errore durante l'estrazione dall'SVG: {e}")
        print("Tentativo fallback al metodo originale...")
        return extract_contours_from_svg_fallback(svg_path, width, height, padding, left_padding, logo_zoom_factor)

def extract_contours_from_svg_fallback(svg_path, width, height, padding, left_padding=0, logo_zoom_factor=1.0):
    """
    Metodo per l'estrazione SVG con OPZIONE per SOLO CONTORNI senza riempimento.
    
    Args:
        left_padding: Padding aggiuntivo dal lato sinistro per SVG
        logo_zoom_factor: Fattore di zoom del logo (1.0=normale, 2.0=doppio, 0.5=metà)
    """
    try:
        print("🔄 Usando estrazione migliorata per SOLI CONTORNI...")
        
        # Carica il file SVG
        paths, attributes, svg_attributes = svg2paths2(svg_path)
        
        if not paths:
            raise Exception("Nessun path trovato nel file SVG.")
        
        # Converti i path SVG in punti
        all_contours = []
        
        print(f"📝 Processando {len(paths)} path SVG per CONTORNI ESTERNI...")
        
        for i, path in enumerate(paths):
            # Discretizza il path in punti
            path_length = path.length()
            if path_length == 0:
                continue
                
            # Adatta il numero di punti alla complessità del path
            num_points = max(100, min(1000, int(path_length * 3)))  # Più punti per maggiore precisione
                
            points = []
            for j in range(num_points):
                t = j / (num_points - 1)
                try:
                    point = path.point(t)
                    # Verifica che il punto sia valido
                    if not (np.isnan(point.real) or np.isnan(point.imag)):
                        points.append([point.real, point.imag])
                except:
                    continue
            
            # Aggiungi contorno solo se ha abbastanza punti validi
            if len(points) > 10:
                contour = np.array(points, dtype=np.float32)
                
                # Verifica che il contour non sia degenere
                area = cv2.contourArea(contour)
                if area > 10:
                    all_contours.append(contour)
                    print(f"  ✓ Path {i+1}: {len(points)} punti, area: {area:.1f}")
        
        if not all_contours:
            raise Exception("Nessun contorno valido estratto dall'SVG.")
        
        print(f"📐 Trovati {len(all_contours)} path originali")
        
        # NUOVA LOGICA: Estrai solo i contorni ESTERNI (bordi) senza riempimento
        processed_contours = []
        
        # Calcola bounding box di tutti i contorni
        all_points = np.vstack(all_contours)
        x_min, y_min = np.min(all_points, axis=0)
        x_max, y_max = np.max(all_points, axis=0)
        
        svg_width = x_max - x_min
        svg_height = y_max - y_min
        
        if svg_width == 0 or svg_height == 0:
            raise Exception("SVG ha dimensioni zero.")
        
        # Crea un'immagine per il rendering ad alta risoluzione
        scale_factor = 4
        render_width = int(svg_width * scale_factor) + 100
        render_height = int(svg_height * scale_factor) + 100
        
        # Crea maschera binaria
        mask = np.zeros((render_height, render_width), dtype=np.uint8)
        
        # Disegna tutti i path come forme piene
        for contour in all_contours:
            # Trasla e scala per il rendering
            scaled_contour = (contour - np.array([x_min, y_min])) * scale_factor + 50
            scaled_contour = scaled_contour.astype(np.int32)
            cv2.fillPoly(mask, [scaled_contour], 255)
        
        # ESTRAI SOLO I BORDI usando operazioni morfologiche AGGRESSIVE
        # Usa erosion più forte per ottenere solo i bordi sottili
        kernel = np.ones((5,5), np.uint8)  # Kernel più grande per erosione più forte
        eroded = cv2.erode(mask, kernel, iterations=2)  # Più iterazioni
        
        # Sottrai l'interno dall'originale per ottenere solo i bordi
        edges = mask - eroded
        
        # Applica scheletonizzazione per ottenere linee sottili
        from skimage.morphology import skeletonize
        skeleton = skeletonize(edges > 0)
        edges = (skeleton * 255).astype(np.uint8)
        
        # Trova i contorni dei bordi
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        print(f"🔍 Estratti {len(contours)} contorni di BORDI usando erosione morfologica")
        
        # Filtra e scala i contorni alla risoluzione target
        target_w = width - (2 * padding)
        target_h = height - (2 * padding)
        base_scale = min(target_w / svg_width, target_h / svg_height)
        scale = base_scale * logo_zoom_factor  # Applica zoom del logo
        
        # Calcola offset per centrare con padding sinistro aggiuntivo
        scaled_w = svg_width * scale
        scaled_h = svg_height * scale
        offset_x = (width - scaled_w) / 2 + left_padding  # Aggiungi padding sinistro
        offset_y = (height - scaled_h) / 2
        
        if left_padding > 0:
            print(f"📐 SVG con padding sinistro: {left_padding}px (offset_x: {offset_x:.1f})")
        if logo_zoom_factor != 1.0:
            print(f"🔍 Logo zoom attivo: {logo_zoom_factor}x (scala finale: {scale:.2f})")
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            # Filtra contorni troppo piccoli
            if area > 500:  # Area minima per essere considerato un bordo significativo
                # Scala alla risoluzione target
                scaled_contour = contour.astype(np.float32) / scale_factor  # Riporta alle coordinate originali
                scaled_contour = (scaled_contour - 50) * scale  # Scala alla risoluzione target
                scaled_contour = scaled_contour + np.array([offset_x, offset_y])  # Centra
                
                processed_contours.append(scaled_contour.astype(np.int32))
                print(f"  ✓ Bordo {i+1}: {len(contour)} punti, area finale: {cv2.contourArea(scaled_contour.astype(np.int32)):.1f}")
        
        if not processed_contours:
            print("⚠️ Nessun bordo trovato, uso i path originali...")
            # Fallback ai path originali se la tecnica dei bordi non funziona
            processed_contours = []
            for contour in all_contours:
                scaled_contour = contour.copy()
                scaled_contour[:, 0] = (contour[:, 0] - x_min) * scale + offset_x
                scaled_contour[:, 1] = (contour[:, 1] - y_min) * scale + offset_y
                processed_contours.append(scaled_contour.astype(np.int32))
        
        print(f"📐 Risultato finale: {len(processed_contours)} contorni processati")
        print("Estrazione contorni da SVG completata.")
        return processed_contours, None
        
    except Exception as e:
        print(f"Errore durante l'estrazione dall'SVG: {e}")
        print("Assicurati che 'svgpathtools' sia installato: pip install svgpathtools")
        return None, None

def extract_contours_from_pdf(pdf_path, width, height, padding, logo_zoom_factor=1.0):
    """
    Estrae i contorni da un file PDF usando il metodo corretto di simple_logo_video.py.
    Questo approccio gestisce correttamente i buchi nelle lettere e i contorni esterni.
    
    Args:
        logo_zoom_factor: Fattore di zoom del logo (1.0=normale, 2.0=doppio, 0.5=metà)
    """
    if not PDF_AVAILABLE:
        raise Exception("PyMuPDF non disponibile. Installa con: pip install PyMuPDF")
    
    try:
        print("🎨 Caricamento PDF Crystal Therapy dalle acque del Natisone...")
        
        # STEP 1: Rasterizza il PDF usando il metodo di simple_logo_video.py
        doc = fitz.open(pdf_path)
        page = doc[0]  # Prima pagina
        
        # Usa scale factor 4 per alta qualità (come simple_logo_video.py usa scale=2)
        scale_factor = 4
        matrix = fitz.Matrix(scale_factor, scale_factor)
        pix = page.get_pixmap(matrix=matrix)
        
        # Converti in array numpy (metodo simple_logo_video.py)
        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        
        doc.close()
        
        # STEP 2: Estrai contorni usando il metodo corretto di simple_logo_video.py
        # Converti in BGR se necessario
        if len(img_array.shape) == 3 and img_array.shape[2] == 4:
            # RGBA to BGR
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        elif len(img_array.shape) == 3 and img_array.shape[2] == 3:
            # RGB to BGR
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            # Grayscale to BGR
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        
        # Converti in scala di grigi
        gray_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # CRUCIALE: Usa THRESH_BINARY_INV come in simple_logo_video.py
        # Questo inverte i colori: nero su bianco diventa bianco su nero
        _, binary_img = cv2.threshold(gray_img, 50, 255, cv2.THRESH_BINARY_INV)
        
        # CRUCIALE: Usa RETR_CCOMP per gestire i buchi nelle lettere
        contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            raise Exception("Nessun contorno trovato nel PDF.")
        
        print(f"📝 Estratti {len(contours)} contorni dal PDF con gestione buchi")
        
        # STEP 3: Centra e ridimensiona i contorni (adattato da simple_logo_video.py)
        if not contours:
            raise Exception("Nessun contorno valido trovato nel PDF.")
        
        # Calcola bounding box di tutti i contorni
        all_points = np.vstack([c for c in contours])
        x, y, w, h = cv2.boundingRect(all_points)
        
        # Trova centro dei contorni e del canvas target
        contour_center_x = x + w / 2
        contour_center_y = y + h / 2
        canvas_center_x = width / 2
        canvas_center_y = height / 2
        
        # Calcola area utilizzabile (con padding)
        padding_fraction = (padding * 2) / min(width, height)  # Converti padding in frazione
        canvas_drawable_width = width * (1 - padding_fraction)
        canvas_drawable_height = height * (1 - padding_fraction)
        
        # Calcola scala per adattare al canvas con zoom del logo
        base_scale = min(canvas_drawable_width / w if w > 0 else 1, 
                        canvas_drawable_height / h if h > 0 else 1)
        scale = base_scale * logo_zoom_factor  # Applica zoom del logo
        
        if logo_zoom_factor != 1.0:
            print(f"🔍 Logo PDF zoom attivo: {logo_zoom_factor}x (scala finale: {scale:.2f})")
        
        # Trasforma tutti i contorni
        scaled_contours = []
        for contour in contours:
            # Converti in float per calcoli precisi
            c_float = contour.astype(np.float32)
            # Trasla al centro e scala (con zoom applicato)
            c_float[:, 0, 0] = (c_float[:, 0, 0] - contour_center_x) * scale + canvas_center_x
            c_float[:, 0, 1] = (c_float[:, 0, 1] - contour_center_y) * scale + canvas_center_y
            # Riconverti in int32
            scaled_contours.append(c_float.astype(np.int32))
        
        print(f"📐 Logo PDF centrato e ridimensionato ({len(scaled_contours)} contorni)")
        print("Estrazione contorni da PDF completata con metodo simple_logo_video.py.")
        
        return scaled_contours, hierarchy
        
    except Exception as e:
        print(f"❌ Errore nell'estrazione contorni da PDF: {e}")
        raise

def smooth_contour(contour, smoothing_factor):
    """Applica lo smussamento spline a un singolo contorno."""
    if len(contour) < 4: return contour
    try:
        contour = contour.squeeze().astype(float)
        epsilon = smoothing_factor * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        tck, u = splprep([approx[:, 0], approx[:, 1]], s=0, per=True)
        u_new = np.linspace(u.min(), u.max(), int(len(contour) * 1.5))
        x_new, y_new = splev(u_new, tck, der=0)
        
        return np.c_[x_new, y_new].astype(np.int32)
    except Exception:
        return contour.astype(np.int32)

def create_unified_mask(contours, hierarchy, width, height, smoothing_enabled, smoothing_factor):
    """Crea una maschera unificata con algoritmo avanzato per eliminare spaccature SVG."""
    mask = np.zeros((height, width), dtype=np.uint8)
    
    if not contours:
        return mask

    smoothed_contours = []
    for contour in contours:
        if smoothing_enabled:
            smoothed_contours.append(smooth_contour(contour, smoothing_factor))
        else:
            smoothed_contours.append(contour)
    
    # Per SVG (hierarchy=None) usa algoritmo avanzato di unificazione
    if hierarchy is None:
        # FASE 1: Crea maschere separate per ogni contorno
        individual_masks = []
        for contour in smoothed_contours:
            temp_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(temp_mask, [contour], 255)
            individual_masks.append(temp_mask)
        
        # FASE 2: Unisci le maschere con operazioni morfologiche progressive
        unified_mask = None
        if individual_masks:
            # Inizia con la prima maschera
            unified_mask = individual_masks[0].copy()
            
            # Unisci progressivamente le altre maschere
            for mask_to_add in individual_masks[1:]:
                # Union diretta
                unified_mask = cv2.bitwise_or(unified_mask, mask_to_add)
                
                # Operazione di connessione per lettere vicine
                kernel_connect = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                unified_mask = cv2.dilate(unified_mask, kernel_connect, iterations=2)
                unified_mask = cv2.erode(unified_mask, kernel_connect, iterations=2)
            
            mask = unified_mask
        else:
            # Fallback se non ci sono maschere individuali
            cv2.fillPoly(mask, smoothed_contours, 255)
        
        # FASE 3: Post-processing avanzato per eliminare spaccature residue
        # 1. Chiusura morfologica per colmare piccoli gap
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
        
        # 2. Riempimento buchi basato su contorni
        contours_found, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours_found:
            # Trova il contorno più grande (dovrebbe essere la scritta principale)
            largest_contour = max(contours_found, key=cv2.contourArea)
            
            # Crea una nuova maschera solo con il contorno più grande riempito
            temp_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(temp_mask, [largest_contour], 255)
            
            # Trova e riempi tutti i buchi interni
            contours_with_holes, hierarchy_holes = cv2.findContours(temp_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            
            final_mask = np.zeros((height, width), dtype=np.uint8)
            for i, contour in enumerate(contours_with_holes):
                # Riempi solo i contorni esterni (hierarchy[i][3] == -1)
                if hierarchy_holes is None or hierarchy_holes[0][i][3] == -1:
                    cv2.fillPoly(final_mask, [contour], 255)
            
            mask = final_mask
        
        # FASE 4: Smoothing finale ottimizzato
        # Blur leggero per eliminare pixelatura
        mask = cv2.GaussianBlur(mask, (5, 5), 1.0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # Operazione finale di pulizia per contorni perfetti
        kernel_final = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_final, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_final, iterations=1)
        
        # FASE 5: Verifica se ci sono ancora spaccature e applica algoritmo avanzato
        # Controlla il numero di componenti connesse
        num_labels, labels = cv2.connectedComponents(mask)
        if num_labels > 2:  # Più di background + una componente = spaccature presenti            
            gap_free_mask = create_gap_free_mask(smoothed_contours, width, height)
            
            # Usa il meglio tra la maschera corrente e quella gap-free
            # Se quella gap-free ha meno componenti, usala
            num_labels_gap_free, _ = cv2.connectedComponents(gap_free_mask)
            if num_labels_gap_free < num_labels:
                mask = gap_free_mask                            
        
    else:
        # Per PDF usa drawContours con hierarchy
        cv2.drawContours(mask, smoothed_contours, -1, 255, -1, lineType=cv2.LINE_AA, hierarchy=hierarchy)
            
    return mask

def create_gap_free_mask(contours, width, height):
    """
    Crea una maschera SVG senza spaccature usando algoritmi geometrici avanzati.
    Approccio multi-fase per eliminare definitivamente le discontinuità.
    """
    if not contours:
        return np.zeros((height, width), dtype=np.uint8)
    
    # FASE 1: Crea maschera base combinando tutti i contorni
    base_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(base_mask, contours, 255)
    
    # FASE 2: Analisi delle componenti connesse
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(base_mask, connectivity=8)
    
    if num_labels <= 2:  # Solo background + una componente = già connesso
        return base_mask
    
    # FASE 3: Trova le componenti principali (esclude background)
    main_components = []
    min_area = (width * height) * 0.001  # Soglia minima 0.1% dell'area totale
    
    for i in range(1, num_labels):  # Salta il background (0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area > min_area:
            component_mask = (labels == i).astype(np.uint8) * 255
            main_components.append(component_mask)
    
    if len(main_components) <= 1:
        return base_mask
    
    # FASE 4: Calcola i centroidi delle componenti principali
    component_centroids = []
    for component in main_components:
        moments = cv2.moments(component)
        if moments['m00'] > 0:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
            component_centroids.append((cx, cy))
    
    # FASE 5: Trova coppie di componenti che dovrebbero essere connesse
    # Usa distanza euclidea e vicinanza per identificare lettere che si toccano
    connections_needed = []
    max_connection_distance = min(width, height) * 0.15  # 15% della dimensione minore
    
    for i in range(len(component_centroids)):
        for j in range(i + 1, len(component_centroids)):
            cx1, cy1 = component_centroids[i]
            cx2, cy2 = component_centroids[j]
            distance = np.sqrt((cx2 - cx1)**2 + (cy2 - cy1)**2)
            
            if distance < max_connection_distance:
                connections_needed.append((i, j, distance))
    
    # FASE 6: Crea connessioni tra componenti vicine
    connected_mask = base_mask.copy()
    
    for i, j, distance in connections_needed:
        # Trova i punti più vicini tra le due componenti
        component1 = main_components[i]
        component2 = main_components[j]
        
        # Estrai i contorni delle componenti
        contours1, _ = cv2.findContours(component1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours2, _ = cv2.findContours(component2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours1 and contours2:
            # Trova i punti più vicini tra i contorni
            min_dist = float('inf')
            closest_points = None
            
            for c1 in contours1:
                for c2 in contours2:
                    for pt1 in c1:
                        for pt2 in c2:
                            dist = np.linalg.norm(pt1[0] - pt2[0])
                            if dist < min_dist:
                                min_dist = dist
                                closest_points = (tuple(pt1[0]), tuple(pt2[0]))
            
            # Disegna una linea di connessione spessa
            if closest_points and min_dist < max_connection_distance:
                pt1, pt2 = closest_points
                # Calcola thickness basato sulla distanza
                thickness = max(3, int(15 - (min_dist / max_connection_distance) * 10))
                cv2.line(connected_mask, pt1, pt2, 255, thickness)
    
    # FASE 7: Post-processing morfologico per smoothing finale
    # Operazione di chiusura per unificare le connessioni
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    connected_mask = cv2.morphologyEx(connected_mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    
    # Operazione di apertura per rimuovere artefatti
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    connected_mask = cv2.morphologyEx(connected_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
    
    # FASE 8: Smooth finale con gaussian blur
    connected_mask = cv2.GaussianBlur(connected_mask, (3, 3), 0.5)
    _, connected_mask = cv2.threshold(connected_mask, 127, 255, cv2.THRESH_BINARY)
    
    return connected_mask

# Rimuovo la vettorizzazione che rallentava invece di velocizzare

def generate_cinematic_path(width, height, path_type, total_frames):
    """
    Genera un percorso cinematografico predefinito che attraversa tutta l'area con BIAS ORIZZONTALE.
    Percorsi ottimizzati per seguire la forma orizzontale della scritta.
    """
    center_x, center_y = width // 2, height // 2
    points = []
    
    # BIAS ORIZZONTALE ULTRA-POTENZIATA: Aumentiamo drasticamente i movimenti orizzontali
    horizontal_scale = 0.8  # AUMENTATO: movimento orizzontale ultra-amplificato per seguire la scritta
    vertical_scale = 0.2   # RIDOTTO: movimento verticale minimizzato per rimanere sulla scritta
    
    if path_type == 'figure_eight':
        # Otto orizzontale che segue la forma della scritta
        for i in range(total_frames):
            t = (i / total_frames) * 4 * np.pi  # Due cicli completi
            x = center_x + (width * horizontal_scale) * np.sin(t)  # Movimento orizzontale ampio
            y = center_y + (height * vertical_scale) * np.sin(2 * t)  # Movimento verticale contenuto
            points.append([x, y])
    
    elif path_type == 'spiral':
        # Spirale orizzontale appiattita per seguire la scritta
        for i in range(total_frames):
            t = (i / total_frames) * 6 * np.pi  # Tre giri completi
            radius_x = (width * horizontal_scale) * (0.3 + 0.7 * np.sin(t * 0.3))
            radius_y = (height * vertical_scale) * (0.3 + 0.7 * np.sin(t * 0.3))
            x = center_x + radius_x * np.cos(t)
            y = center_y + radius_y * np.sin(t)
            points.append([x, y])
    
    elif path_type == 'wave':
        # Onda che segue principalmente la direzione orizzontale della scritta
        for i in range(total_frames):
            progress = i / total_frames
            x = width * 0.05 + (width * 0.9) * progress  # Movimento orizzontale completo
            wave_offset = np.sin(progress * 8 * np.pi) * height * vertical_scale  # Ondulazione verticale contenuta
            y = center_y + wave_offset
            points.append([x, y])
    
    elif path_type == 'circular':
        # Ellisse orizzontale che abbraccia la scritta
        for i in range(total_frames):
            t = (i / total_frames) * 2 * np.pi
            radius_x = width * horizontal_scale   # Ellisse allungata orizzontalmente
            radius_y = height * vertical_scale    # Compressa verticalmente
            x = center_x + radius_x * np.cos(t)
            y = center_y + radius_y * np.sin(t)
            points.append([x, y])
    
    elif path_type == 'cross':
        # Croce con enfasi sui movimenti orizzontali
        quarter = total_frames // 4
        for i in range(total_frames):
            if i < quarter:  # Sinistra -> Centro (movimento orizzontale)
                x = width * 0.05 + (width * horizontal_scale) * (i / quarter)
                y = center_y
            elif i < 2 * quarter:  # Centro -> Destra (movimento orizzontale)
                x = center_x + (width * horizontal_scale) * ((i - quarter) / quarter)
                y = center_y
            elif i < 3 * quarter:  # Centro -> Alto (movimento verticale ridotto)
                x = center_x
                y = center_y - (height * vertical_scale) * ((i - 2 * quarter) / quarter)
            else:  # Alto -> Basso (movimento verticale ridotto)
                x = center_x
                y = center_y - height * vertical_scale + (height * vertical_scale * 2) * ((i - 3 * quarter) / quarter)
            points.append([x, y])
    
    elif path_type == 'horizontal_sweep':
        # NUOVO: Spazzata orizzontale ULTRA-POTENZIATA che segue perfettamente la scritta
        for i in range(total_frames):
            progress = i / total_frames
            # Movimento principale sinistra-destra con ampio range
            base_x = width * 0.05 + (width * 0.9) * (0.5 + 0.5 * np.sin(progress * 2 * np.pi))
            # Aggiunta variazione sinusoidale per movimento più complesso
            x = base_x + width * 0.1 * np.sin(progress * 8 * np.pi)
            # Variazione verticale molto ridotta ma con pattern interessante
            y = center_y + height * 0.08 * np.sin(progress * 12 * np.pi) * np.cos(progress * 4 * np.pi)
            points.append([x, y])
    
    elif path_type == 'horizontal_zigzag':
        # NUOVO: Zigzag orizzontale lungo la scritta
        for i in range(total_frames):
            progress = i / total_frames
            # Movimento a zigzag orizzontale
            x = width * 0.1 + (width * 0.8) * progress
            # Zigzag verticale contenuto
            y = center_y + height * 0.15 * np.sin(progress * 16 * np.pi)
            points.append([x, y])
    
    elif path_type == 'horizontal_wave_complex':
        # NUOVO: Onda orizzontale complessa multi-frequenza
        for i in range(total_frames):
            progress = i / total_frames
            # Movimento orizzontale principale con onde multiple
            x = width * 0.05 + (width * 0.9) * progress + width * 0.05 * np.sin(progress * 20 * np.pi)
            # Combinazione di onde verticali diverse per movimento più "vivo"
            wave1 = np.sin(progress * 6 * np.pi) * height * 0.1
            wave2 = np.sin(progress * 15 * np.pi) * height * 0.05
            wave3 = np.cos(progress * 25 * np.pi) * height * 0.03
            y = center_y + wave1 + wave2 + wave3
            points.append([x, y])
    
    return np.array(points)

def apply_lens_deformation(mask, lenses, frame_index, config, dynamic_params=None, audio_factors=None):
    """
    Applica una deformazione basata su "lenti" che seguono percorsi cinematografici predefiniti.
    Sistema completamente rivisto per movimenti ampi, fluidi e cinematografici con reattività audio.
    """
    h, w = mask.shape
    
    # Ottieni moltiplicatori dinamici se disponibili
    lens_strength_mult = dynamic_params.get('lens_strength_multiplier', 1.0) if dynamic_params else 1.0
    
    # Integra i fattori audio-reattivi se disponibili
    if audio_factors:
        lens_strength_mult *= audio_factors['strength_factor']
    
    map_x_grid, map_y_grid = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    final_map_x = np.copy(map_x_grid)
    final_map_y = np.copy(map_y_grid)

    for lens in lenses:
        dx = map_x_grid - lens['pos'][0]
        dy = map_y_grid - lens['pos'][1]

        if config.WORM_SHAPE_ENABLED:
            # Deformazione a "verme": distorciamo lo spazio di calcolo della distanza
            angle = lens['angle']
            dx_rot = dx * np.cos(angle) - dy * np.sin(angle)
            dy_rot = dx * np.sin(angle) + dy * np.cos(angle)
            
            # Allunghiamo la forma su un asse per creare il "corpo" del verme
            dx_scaled = dx_rot / config.WORM_LENGTH
            
            # CORREZIONE ANTI-SFARFALLIO: Sostituisco noise casuale con pattern sinusoidale predicibile
            # Il noise casuale causava lo sfarfallio, ora uso movimento fluido e prevedibile
            wave_time = frame_index * 0.03 + lens['pulsation_offset']  # Velocità fissa controllata
            sinusoidal_curve = np.sin(dx_rot * 0.01 + wave_time) * 30  # Ampiezza ridotta da 50 a 30
            dy_scaled = dy_rot + sinusoidal_curve
            
            distance = np.sqrt(dx_scaled**2 + dy_scaled**2)
        else:
            distance = np.sqrt(dx**2 + dy**2)

        normalized_distance = distance / (lens['radius'] + 1e-6)
        lens_mask = normalized_distance < 1.0
        
        # Applica moltiplicatore dinamico alla forza della lente
        dynamic_strength = lens['strength'] * lens_strength_mult
        displacement = (1.0 - normalized_distance[lens_mask]) * dynamic_strength
        
        # Applica lo spostamento lungo la linea dal pixel al centro della lente
        final_map_x[lens_mask] += dx[lens_mask] * displacement
        final_map_y[lens_mask] += dy[lens_mask] * displacement

    # SISTEMA AGGIORNATO: Movimento cinematografico + PULSAZIONE DINAMICA ULTRA-POTENZIATA
    for lens in lenses:
        # === PULSAZIONE DINAMICA ULTRA-MIGLIORATA ===
        if config.LENS_PULSATION_ENABLED:
            # Calcola pulsazione con fase unica per ogni lente e frequenze multiple
            pulsation_time = frame_index * config.LENS_PULSATION_SPEED + lens['pulsation_offset']
            
            # Integra fattore audio nella velocità di pulsazione
            if audio_factors:
                pulsation_time *= audio_factors['pulsation_factor']
            
            # CORREZIONE ANTI-SFARFALLIO: Pulsazione semplificata per ridurre caos
            # Rimuovo le pulsazioni secondarie e terziarie che creano sfarfallio
            base_pulsation = np.sin(pulsation_time)
            # secondary_pulsation = 0.3 * np.sin(pulsation_time * 2.7)  # RIMOSSA
            # tertiary_pulsation = 0.15 * np.cos(pulsation_time * 4.1)  # RIMOSSA
            
            total_pulsation = base_pulsation  # Solo pulsazione base per fluidità
            
            # Modula l'ampiezza della pulsazione con l'audio
            pulsation_amplitude = config.LENS_PULSATION_AMPLITUDE
            if audio_factors:
                pulsation_amplitude *= audio_factors['pulsation_factor']
            
            pulsation_factor = 1.0 + pulsation_amplitude * total_pulsation * 0.5  # Ridotta ampiezza
            lens['radius'] = lens['base_radius'] * pulsation_factor
            
            # CORREZIONE: Pulsazione forza molto semplificata
            if config.LENS_FORCE_PULSATION_ENABLED:
                force_pulsation = np.sin(pulsation_time * 1.2)  # Frequenza ridotta da 1.8
                force_pulsation_amplitude = config.LENS_FORCE_PULSATION_AMPLITUDE
                if audio_factors:
                    force_pulsation_amplitude *= audio_factors['strength_factor']
                force_factor = 1.0 + force_pulsation_amplitude * force_pulsation * 0.3  # Ampiezza ridotta
                lens['strength'] = lens['base_strength'] * force_factor
        
        # === MOVIMENTO LUNGO PERCORSI CINEMATOGRAFICI ULTRA-VELOCE ===
        # Velocità configurabile tramite parametri della Config, modulata dall'audio
        movement_speed_multiplier = config.LENS_PATH_SPEED_MULTIPLIER
        if audio_factors:
            movement_speed_multiplier *= audio_factors['speed_factor']
            
        path_progress = ((frame_index + lens['path_offset']) * movement_speed_multiplier) % len(lens['path'])
        current_target = lens['path'][int(path_progress)]
        
        # Interpolazione ultra-fluida tra i punti del percorso
        next_index = (int(path_progress) + 1) % len(lens['path'])
        next_target = lens['path'][next_index]
        interpolation_factor = path_progress - int(path_progress)
        
        # Interpolazione con curva smooth per movimento più naturale
        smooth_factor = 3 * interpolation_factor**2 - 2 * interpolation_factor**3  # Smoothstep
        smooth_target = current_target + (next_target - current_target) * smooth_factor
        
        # Movimento ultra-aggressivo e reattivo verso il target
        direction = smooth_target - lens['pos']
        distance_to_target = np.linalg.norm(direction)
        
        if distance_to_target > 0:
            # CORREZIONE ANTI-SFARFALLIO: Velocità costante invece di adattiva per movimento fluido
            # La velocità adattiva causava accelerazioni brusche che generavano sfarfallio
            base_speed = config.LENS_SPEED_FACTOR * config.LENS_BASE_SPEED_MULTIPLIER
            
            # Modula la velocità con i fattori audio
            if audio_factors:
                base_speed *= audio_factors['speed_factor']
            
            # adaptive_speed = base_speed * (1.0 + 0.5 * min(distance_to_target / 40, 1.5))  # RIMOSSA
            desired_velocity = (direction / distance_to_target) * base_speed  # Velocità costante
            
            # Inerzia più alta per movimento ultra-fluido
            enhanced_inertia = min(0.99, config.LENS_INERTIA + 0.01)  # Aumentata di 1%
            lens['velocity'] = lens['velocity'] * enhanced_inertia + desired_velocity * (1 - enhanced_inertia)
        
        # Aggiorna posizione e angolo con velocità configurabile
        lens['pos'] += lens['velocity']
        lens['angle'] += lens['rotation_speed'] * config.LENS_ROTATION_SPEED_MULTIPLIER
        
        # Assicurati che rimanga nei limiti con margini morbidi
        margin = config.LENS_MIN_RADIUS
        lens['pos'][0] = np.clip(lens['pos'][0], margin, w - margin)
        lens['pos'][1] = np.clip(lens['pos'][1], margin, h - margin)

    deformed_mask = cv2.remap(mask, final_map_x, final_map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return deformed_mask


def apply_organic_deformation(mask, frame_index, params, dynamic_params=None):
    """Applica una deformazione organica super fluida usando calcolo a griglia con parametri dinamici."""
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

def process_background(bg_frame, config):
    """
    Processa il frame di sfondo: lo adatta alle dimensioni video senza crop,
    lo scurisce e ne estrae i contorni per i traccianti.
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
        # Metodo originale con crop
        cropped_bg = bg_frame[config.BG_CROP_Y_START : config.BG_CROP_Y_END, :]
        final_bg = cv2.resize(cropped_bg, (config.WIDTH, config.HEIGHT))
    
    # 2. Scurisce e contrasta
    if config.BG_DARKEN_FACTOR < 1.0:
        # Applica lo scurimento in modo più "morbido"
        final_bg = cv2.addWeighted(final_bg, config.BG_DARKEN_FACTOR, np.zeros_like(final_bg), 1 - config.BG_DARKEN_FACTOR, 0)
    if config.BG_CONTRAST_FACTOR > 1.0:
        final_bg = cv2.convertScaleAbs(final_bg, alpha=config.BG_CONTRAST_FACTOR, beta=0)

    # 3. Estrae i contorni (bordi) per l'effetto tracciante del logo con soglie ottimizzate
    gray_bg = cv2.cvtColor(final_bg, cv2.COLOR_BGR2GRAY)  # Usa il frame processato
    # Applica un leggero blur per ridurre il rumore prima di Canny
    gray_bg = cv2.GaussianBlur(gray_bg, (3, 3), 0)
    logo_edges = cv2.Canny(gray_bg, config.TRACER_THRESHOLD1, config.TRACER_THRESHOLD2)
    
    # 4. NUOVO: Estrae traccianti separati per lo sfondo con soglie diverse
    bg_edges = None
    if hasattr(config, 'BG_TRACER_ENABLED') and config.BG_TRACER_ENABLED:
        # Usa soglie ottimizzate per catturare i contorni del video di sfondo
        bg_edges = cv2.Canny(gray_bg, config.BG_TRACER_THRESHOLD1, config.BG_TRACER_THRESHOLD2)
        # Dilata leggermente per renderli più visibili e organici
        kernel = np.ones((2,2), np.uint8)
        bg_edges = cv2.dilate(bg_edges, kernel, iterations=1)
    
    return final_bg, logo_edges, bg_edges

def render_frame(contours, hierarchy, width, height, frame_index, total_frames, config, bg_frame, texture_image, tracer_history, bg_tracer_history, lenses, audio_data=None):
    """
    Rende un singolo frame dell'animazione, applicando la pipeline di effetti completa.
    """
    # --- 0. Ottieni Parametri Dinamici ---
    dynamic_params = get_dynamic_parameters(frame_index, total_frames)
    
    # --- 0.5. Calcola Fattori Audio-Reattivi ---
    audio_factors = get_audio_reactive_factors(audio_data, frame_index, config)

    # --- 1. Preparazione Sfondo e Traccianti ---
    bg_result = process_background(bg_frame, config)
    if len(bg_result) == 3:
        final_frame, current_logo_edges, current_bg_edges = bg_result
    else:
        final_frame, current_logo_edges = bg_result
        current_bg_edges = None
    
    # --- 2. Creazione Layer Traccianti del Logo (CON PARAMETRI DINAMICI) ---
    if config.TRACER_ENABLED and len(tracer_history) > 0:
        tracer_layer = np.zeros_like(final_frame, dtype=np.float32)
        # Applica moltiplicatore dinamico all'opacità
        dynamic_opacity = config.TRACER_MAX_OPACITY * dynamic_params.get('tracer_opacity_multiplier', 1.0)
        opacities = np.linspace(0, dynamic_opacity, len(tracer_history))
        
        for i, past_edges in enumerate(reversed(tracer_history)):
            # --- NUOVO: Colore dinamico per i traccianti ---
            hue_shift = (frame_index * 0.1 + i * 0.5) % 180
            base_color_hsv = cv2.cvtColor(np.uint8([[config.TRACER_BASE_COLOR]]), cv2.COLOR_BGR2HSV)[0][0]
            new_hue = (base_color_hsv[0] + hue_shift) % 180
            dynamic_color_hsv = np.uint8([[[new_hue, base_color_hsv[1], base_color_hsv[2]]]])
            dynamic_color_bgr = cv2.cvtColor(dynamic_color_hsv, cv2.COLOR_HSV2BGR)[0][0]
            
            # Colora i bordi e applica l'opacità dinamica
            colored_tracer = cv2.cvtColor(past_edges, cv2.COLOR_GRAY2BGR).astype(np.float32)
            colored_tracer[past_edges > 0] = np.array(dynamic_color_bgr, dtype=np.float32)
            tracer_with_opacity = cv2.multiply(colored_tracer, opacities[i])
            tracer_layer = cv2.add(tracer_layer, tracer_with_opacity)
            
        final_frame = cv2.add(final_frame.astype(np.float32), tracer_layer)
        final_frame = np.clip(final_frame, 0, 255).astype(np.uint8)

    # --- 2.5. NUOVO: Creazione Layer Traccianti Sfondo (CON PARAMETRI DINAMICI) ---
    if hasattr(config, 'BG_TRACER_ENABLED') and config.BG_TRACER_ENABLED and len(bg_tracer_history) > 0:
        bg_tracer_layer = np.zeros_like(final_frame, dtype=np.float32)
        # Applica moltiplicatore dinamico all'opacità dello sfondo
        dynamic_bg_opacity = config.BG_TRACER_MAX_OPACITY * dynamic_params.get('bg_tracer_opacity_multiplier', 1.0)
        bg_opacities = np.linspace(0, dynamic_bg_opacity, len(bg_tracer_history))
        
        for i, past_bg_edges in enumerate(reversed(bg_tracer_history)):
            # Colore dinamico per traccianti sfondo (diverso dal logo)
            hue_shift_bg = (frame_index * 0.05 + i * 0.3) % 180  # Velocità diversa
            base_color_hsv_bg = cv2.cvtColor(np.uint8([[config.BG_TRACER_BASE_COLOR]]), cv2.COLOR_BGR2HSV)[0][0]
            new_hue_bg = (base_color_hsv_bg[0] + hue_shift_bg) % 180
            dynamic_color_hsv_bg = np.uint8([[[new_hue_bg, base_color_hsv_bg[1], base_color_hsv_bg[2]]]])
            dynamic_color_bgr_bg = cv2.cvtColor(dynamic_color_hsv_bg, cv2.COLOR_HSV2BGR)[0][0]
            
            # Colora i bordi dello sfondo e applica l'opacità dinamica
            colored_bg_tracer = cv2.cvtColor(past_bg_edges, cv2.COLOR_GRAY2BGR).astype(np.float32)
            colored_bg_tracer[past_bg_edges > 0] = np.array(dynamic_color_bgr_bg, dtype=np.float32)
            bg_tracer_with_opacity = cv2.multiply(colored_bg_tracer, bg_opacities[i])
            bg_tracer_layer = cv2.add(bg_tracer_layer, bg_tracer_with_opacity)
            
        final_frame = cv2.add(final_frame.astype(np.float32), bg_tracer_layer)
        final_frame = np.clip(final_frame, 0, 255).astype(np.uint8)

    # --- 3. Creazione Maschera del Logo ---
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

    # --- 6. Composizione Finale con BLENDING AVANZATO SCRITTA-SFONDO ---
    
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

def initialize_lenses(config):
    """Inizializza una lista di lenti con percorsi cinematografici predefiniti per movimenti ampi e fluidi."""
    lenses = []
    
    # Tipi di percorsi cinematografici - ULTRA-BIAS ORIZZONTALE per seguire la scritta
    horizontal_paths = ['horizontal_sweep', 'horizontal_zigzag', 'horizontal_wave_complex', 'wave']  # Percorsi orizzontali privilegiati
    mixed_paths = ['figure_eight', 'spiral', 'circular', 'cross']  # Percorsi misti
    
    # BIAS ORIZZONTALE: 70% delle lenti usa percorsi orizzontali
    horizontal_lens_count = int(config.NUM_LENSES * 0.7)
    mixed_lens_count = config.NUM_LENSES - horizontal_lens_count
    
    # Lista combinata con bias orizzontale
    path_assignments = []
    # Assegna percorsi orizzontali alla maggior parte delle lenti
    for i in range(horizontal_lens_count):
        path_assignments.append(horizontal_paths[i % len(horizontal_paths)])
    # Aggiungi alcuni percorsi misti per varietà
    for i in range(mixed_lens_count):
        path_assignments.append(mixed_paths[i % len(mixed_paths)])
    
    # Mescola per evitare che tutte le lenti orizzontali siano consecutive
    np.random.shuffle(path_assignments)
    
    # Durata del video in frame (per calcolare i percorsi)
    total_frames = int(config.DURATION_SECONDS * config.FPS)
    
    for i in range(config.NUM_LENSES):
        # Usa il tipo di percorso assegnato con bias orizzontale
        path_type = path_assignments[i]
        
        # Genera il percorso cinematografico completo
        path = generate_cinematic_path(config.WIDTH, config.HEIGHT, path_type, total_frames)
        
        # Posizione iniziale casuala lungo il percorso
        path_offset = np.random.randint(0, len(path))
        initial_pos = path[path_offset]
        
        # NUOVA: Base radius variabile per pulsazioni più interessanti
        base_radius = np.random.uniform(config.LENS_MIN_RADIUS, config.LENS_MAX_RADIUS)
        current_radius = base_radius  # Inizia con il raggio base
        
        # NUOVA: Forza base che verrà modulata dalla pulsazione
        base_strength = np.random.uniform(config.LENS_MIN_STRENGTH, config.LENS_MAX_STRENGTH)
        
        lens = {
            'pos': np.array(initial_pos, dtype=np.float32),
            'velocity': np.array([0.0, 0.0]),  # Inizia ferma, si muove verso il percorso
            'radius': current_radius,
            'base_radius': base_radius,  # Raggio base per pulsazione
            'strength': base_strength,
            'base_strength': base_strength,  # NUOVO: forza base per pulsazione
            'angle': np.random.uniform(0, 2 * np.pi),
            'rotation_speed': np.random.uniform(-0.008, 0.008),  # Rotazione leggermente più veloce
            'pulsation_offset': np.random.uniform(0, 2 * np.pi),  # Offset fase per pulsazione asincrona
            'path': path,  # Percorso cinematografico completo
            'path_offset': path_offset,  # Offset iniziale nel percorso
            'path_type': path_type  # Tipo di percorso per debug
        }
        lenses.append(lens)
    
    print(f"🔮 Inizializzate {config.NUM_LENSES} lenti ULTRA-CINEMATOGRAFICHE:")
    print(f"   📏 {horizontal_lens_count} lenti con percorsi ORIZZONTALI (bias 70%)")
    print(f"   🌀 {mixed_lens_count} lenti con percorsi MISTI per varietà")
    for i, lens in enumerate(lenses):
        movement_type = "ORIZZONTALE" if lens['path_type'] in horizontal_paths else "MISTO"
        print(f"     Lente {i+1}: {lens['path_type']} ({movement_type})")
    
    return lenses

def find_texture_file():
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
    if os.path.exists(Config.TEXTURE_FALLBACK_PATH):
        print(f"🎨 Uso texture fallback: {Config.TEXTURE_FALLBACK_PATH}")
        return Config.TEXTURE_FALLBACK_PATH
    
    print(f"⚠️ Nessuna texture trovata, il logo non sarà texturizzato")
    return None

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

def main():
    """Funzione principale per generare l'animazione del logo."""
    # --- Codici ANSI per colori e stili nel terminale ---
    C_CYAN = '\033[96m'
    C_GREEN = '\033[92m'
    C_YELLOW = '\033[93m'
    C_BLUE = '\033[94m'
    C_MAGENTA = '\033[95m'
    C_RED = '\033[91m'  # Aggiungo colore rosso
    C_BOLD = '\033[1m'
    C_END = '\033[0m'
    SPINNER_CHARS = ['🔮', '✨', '🌟', '💎']
    
    # Mostra le opzioni di blending disponibili
    print_blending_options()
    
    # 🎨 APPLICA PRESET BLENDING AUTOMATICO
    apply_blending_preset(Config)

    # NUOVO: Calcola dimensioni del video dalle dimensioni SVG + padding
    svg_width, svg_height = get_svg_dimensions(Config.SVG_PATH)

    # 📱 FORMATO INSTAGRAM STORIES (9:16)
    if Config.INSTAGRAM_STORIES_MODE:
        if Config.TEST_MODE:
            # Versione ridotta per test: 540x960 (metà di 1080x1920)
            Config.WIDTH = 540
            Config.HEIGHT = 960
        else:
            # Formato Instagram Stories standard: 1080x1920
            Config.WIDTH = 1080
            Config.HEIGHT = 1920
        format_info = "Instagram Stories (9:16)"
    else:
        # Formato tradizionale basato su dimensioni SVG
        Config.WIDTH = svg_width + (Config.SVG_PADDING * 2)
        Config.HEIGHT = svg_height + (Config.SVG_PADDING * 2)
        format_info = "SVG-based"
    
    print(f"{C_BOLD}{C_CYAN}🌊 Avvio rendering Crystal Therapy - SVG CENTRATO...{C_END}")
    print(f"📐 Dimensioni SVG: {svg_width}x{svg_height}")
    print(f"📐 Dimensioni video: {Config.WIDTH}x{Config.HEIGHT} (formato: {format_info})")
    if Config.INSTAGRAM_STORIES_MODE and not Config.TEST_MODE:
        print(f"📱 INSTAGRAM STORIES: Formato verticale ottimizzato per mobile")
    if Config.SVG_PADDING and not Config.INSTAGRAM_STORIES_MODE:
        print(f"🎨 Padding SVG: {Config.SVG_PADDING}px")
    if Config.TEST_MODE:
        print(f"🎬 TEST MODE: 10fps, {Config.DURATION_SECONDS}s, risoluzione ridotta per velocità")
    else:
        print(f"🎬 PRODUZIONE: 30fps, {Config.DURATION_SECONDS}s, risoluzione completa")
    source_type = "SVG vettoriale" if Config.USE_SVG_SOURCE else "PDF rasterizzato"
    print(f"📄 Sorgente: {source_type} con smoothing ottimizzato")
    print(f"🎥 Video sfondo: ORIGINALE senza crop, rallentato {Config.BG_SLOWDOWN_FACTOR}x")
    print(f"✨ Traccianti + Blending + Glow COMPATIBILE")
    print(f"� Variazione dinamica + codec video testati")
    print(f"💎 RENDERING MOVIMENTO GARANTITO per compatibilità VLC/QuickTime!")
    
    # Carica contorni da SVG o PDF
    if Config.USE_SVG_SOURCE:
        if Config.INSTAGRAM_STORIES_MODE:
            # Per Instagram Stories, centra il logo nel formato verticale con spostamento a destra
            horizontal_margin = (Config.WIDTH - svg_width) // 2
            # Riduci un po' il margine sinistro per spostare il logo leggermente a destra
            right_shift = 10 if Config.TEST_MODE else 20
            effective_padding = max(Config.SVG_PADDING, horizontal_margin - right_shift)
            contours, hierarchy = extract_contours_from_svg(Config.SVG_PATH, Config.WIDTH, Config.HEIGHT, effective_padding, Config.SVG_LEFT_PADDING, Config.LOGO_ZOOM_FACTOR)
        else:
            contours, hierarchy = extract_contours_from_svg(Config.SVG_PATH, Config.WIDTH, Config.HEIGHT, Config.SVG_PADDING, Config.SVG_LEFT_PADDING, Config.LOGO_ZOOM_FACTOR)
    else:
        if Config.INSTAGRAM_STORIES_MODE:
            # Per Instagram Stories, centra il logo nel formato verticale con spostamento a destra
            horizontal_margin = (Config.WIDTH - svg_width) // 2
            # Riduci un po' il margine sinistro per spostare il logo leggermente a destra
            right_shift = 10 if Config.TEST_MODE else 20
            effective_padding = max(Config.SVG_PADDING, horizontal_margin - right_shift)
            contours, hierarchy = extract_contours_from_pdf(Config.PDF_PATH, Config.WIDTH, Config.HEIGHT, effective_padding, Config.LOGO_ZOOM_FACTOR)
        else:
            contours, hierarchy = extract_contours_from_pdf(Config.PDF_PATH, Config.WIDTH, Config.HEIGHT, Config.SVG_PADDING, Config.LOGO_ZOOM_FACTOR)

    if not contours:
        source_name = "SVG" if Config.USE_SVG_SOURCE else "PDF"
        print(f"Errore critico: nessun contorno valido trovato nel {source_name}. Uscita.")
        return

    print("Estrazione contorni riuscita.")

    # --- Caricamento Texture (se abilitata) ---
    texture_image = None
    if Config.TEXTURE_ENABLED:
        # Prima cerca la texture automaticamente
        texture_path = find_texture_file()
        # Poi carica la texture trovata (o fallback se non trovata)
        texture_image = load_texture(texture_path, Config.WIDTH, Config.HEIGHT)
        if texture_image is not None:
            print("Texture infusa con l'essenza del Natisone - Creata dal team Alex Ortiga, TV Int, Iaia & Friend.")
    else:
        print("La texturizzazione del logo è disabilitata.")

    # --- Apertura Video di Sfondo ---
    bg_video = cv2.VideoCapture(Config.BACKGROUND_VIDEO_PATH)
    if not bg_video.isOpened():
        print(f"Errore: impossibile aprire il video di sfondo in {Config.BACKGROUND_VIDEO_PATH}")
        # Crea uno sfondo nero di fallback
        bg_video = None
        bg_start_frame = 0
        bg_total_frames = 0  # Aggiungo variabile per fallback
    else:
        # NUOVO: Ottieni informazioni del video di sfondo per il rallentamento
        bg_total_frames = int(bg_video.get(cv2.CAP_PROP_FRAME_COUNT))
        bg_fps = bg_video.get(cv2.CAP_PROP_FPS)
        
        # 🎲 RANDOM START: Calcola frame di inizio casuale (max 2/3 del video)
        bg_start_frame = 0
        if Config.BG_RANDOM_START and bg_total_frames > Config.TOTAL_FRAMES:
            # Calcola quanti frame servono considerando il rallentamento
            frames_needed = int(Config.TOTAL_FRAMES / Config.BG_SLOWDOWN_FACTOR) + 1
            # Assicurati di avere abbastanza frame rimanenti per il rendering
            max_start_frame = max(0, int(bg_total_frames * 2/3) - frames_needed)
            if max_start_frame > 0:
                bg_start_frame = np.random.randint(0, max_start_frame)
                start_time = bg_start_frame / bg_fps
                end_time = start_time + (frames_needed / bg_fps)
                print(f"🎬 Video sfondo: {bg_total_frames} frame @ {bg_fps}fps")
                print(f"🎲 Inizio casuale da frame {bg_start_frame} ({start_time:.1f}s -> {end_time:.1f}s)")
                print(f"📊 Frame necessari: {frames_needed} (con rallentamento {Config.BG_SLOWDOWN_FACTOR}x)")
            else:
                print(f"🎬 Video sfondo: {bg_total_frames} frame @ {bg_fps}fps")
                print(f"⚠️ Video troppo corto per random start")
        else:
            print(f"🎬 Video sfondo: {bg_total_frames} frame @ {bg_fps}fps")
            if not Config.BG_RANDOM_START:
                print(f"🔄 Inizio dal primo frame (random start disabilitato)")
        
        print(f"🐌 RALLENTAMENTO ATTIVATO: Video sfondo {Config.BG_SLOWDOWN_FACTOR}x più lento")
    
    # Setup video writer con codec ottimizzato per WhatsApp
    if Config.WHATSAPP_COMPATIBLE:
        # H.264 è il migliore per WhatsApp
        fourcc = cv2.VideoWriter_fourcc(*'H264')  # Priorità H264 per WhatsApp
        print("🔄 Usando H.264 per compatibilità WhatsApp...")
    else:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Fallback generico
        
    output_filename = get_timestamp_filename()
    out = cv2.VideoWriter(output_filename, fourcc, Config.FPS, (Config.WIDTH, Config.HEIGHT))
    
    if not out.isOpened():
        print("TENTATIVO 1 FALLITO. Provo con mp4v...")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_filename, fourcc, Config.FPS, (Config.WIDTH, Config.HEIGHT))
        
    if not out.isOpened():
        print("TENTATIVO 2 FALLITO. Provo con XVID...")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_filename, fourcc, Config.FPS, (Config.WIDTH, Config.HEIGHT))
        
    if not out.isOpened():
        print("ERRORE CRITICO: Nessun codec video funziona!")
        return
    
    # --- Inizializzazione Effetti ---
    tracer_history = deque(maxlen=Config.TRACER_TRAIL_LENGTH)
    
    # --- NUOVO: Inizializzazione Traccianti Sfondo ---
    bg_tracer_history = deque(maxlen=getattr(Config, 'BG_TRACER_TRAIL_LENGTH', 35))

    # --- Inizializzazione per Effetto Lenti (NUOVO) ---
    lenses = []
    if Config.LENS_DEFORMATION_ENABLED:
        lenses = initialize_lenses(Config)
        print(f"🌊 Liberate {len(lenses)} creature liquide per Alex Ortiga... texturizzizando con TVInt")

    # --- NUOVO: Caricamento e Analisi Audio ---
    audio_data = None
    if Config.AUDIO_ENABLED:
        audio_data = load_audio_analysis(
            Config.AUDIO_FILES, 
            Config.DURATION_SECONDS, 
            Config.FPS,
            Config.AUDIO_RANDOM_SELECTION,
            Config.AUDIO_RANDOM_START
        )
        if audio_data:
            print(f"🎵 Audio caricato: reattività lenti attivata con {len(lenses)} elementi sincronizzati")
            print(f"📂 File selezionato: {audio_data['selected_file']}")
            if audio_data['start_offset'] > 0:
                print(f"⏯️ Inizio da: {audio_data['start_offset']:.1f}s")
        else:
            print("⚠️ Errore nel caricamento audio: rendering senza sincronizzazione")
    else:
        print("🔇 Audio disabilitato nella configurazione")

    print(f"Rendering dell'animazione in corso... ({Config.TOTAL_FRAMES} frame da elaborare)")
    start_time = time.time()
    
    try:
        for i in range(Config.TOTAL_FRAMES):
            # --- Gestione Frame di Sfondo con RALLENTAMENTO ---
            if bg_video:
                # NUOVO: Calcola il frame del video di sfondo rallentato con offset casuale
                bg_frame_index = bg_start_frame + int(i / Config.BG_SLOWDOWN_FACTOR)
                
                # Controllo di sicurezza: assicurati che il frame sia valido
                if bg_frame_index >= bg_total_frames:
                    # Se superiamo la fine, torna al punto di partenza casuale
                    bg_frame_index = bg_start_frame + (bg_frame_index - bg_start_frame) % (bg_total_frames - bg_start_frame)
                
                # Imposta la posizione nel video di sfondo
                bg_video.set(cv2.CAP_PROP_POS_FRAMES, bg_frame_index)
                ret, bg_frame = bg_video.read()
                
                # Doppio controllo di sicurezza
                if not ret:
                    print(f"⚠️ Errore lettura frame {bg_frame_index}, riavvolgendo...")
                    bg_video.set(cv2.CAP_PROP_POS_FRAMES, bg_start_frame)
                    ret, bg_frame = bg_video.read()
                    if not ret:
                        # Ultima risorsa: crea frame nero
                        bg_frame = np.zeros((Config.HEIGHT, Config.WIDTH, 3), dtype=np.uint8)
                # RIMOSSO: Non ridimensionare qui, lo fa process_background
                # bg_frame = cv2.resize(bg_frame, (Config.WIDTH, Config.HEIGHT))
            else:
                # Crea uno sfondo nero se non c'è video
                bg_frame = np.zeros((Config.HEIGHT, Config.WIDTH, 3), dtype=np.uint8)

            frame_result = render_frame(contours, hierarchy, Config.WIDTH, Config.HEIGHT, i, Config.TOTAL_FRAMES, Config, bg_frame, texture_image, tracer_history, bg_tracer_history, lenses, audio_data)
            
            if len(frame_result) == 3:
                frame, current_logo_edges, current_bg_edges = frame_result
            else:
                frame, current_logo_edges = frame_result
                current_bg_edges = None
            
            # Aggiorna la storia dei traccianti
            if Config.TRACER_ENABLED:
                tracer_history.append(current_logo_edges)
            
            # Aggiorna la storia dei traccianti dello sfondo
            if hasattr(Config, 'BG_TRACER_ENABLED') and Config.BG_TRACER_ENABLED and current_bg_edges is not None:
                bg_tracer_history.append(current_bg_edges)
            
            out.write(frame)
            
            # --- Log di Avanzamento Magico (aggiornamento fluido) ---
            elapsed = time.time() - start_time
            fps = (i + 1) / elapsed if elapsed > 0 else 0
            
            # Calcolo ETA con smoothing
            remaining_frames = Config.TOTAL_FRAMES - (i + 1)
            eta_seconds = remaining_frames / fps if fps > 0 else 0
            eta_minutes, eta_sec = divmod(int(eta_seconds), 60)
            eta_str = f"{eta_minutes:02d}:{eta_sec:02d}"

            # Barra di avanzamento fluida con più dettagli
            progress = (i + 1) / Config.TOTAL_FRAMES
            bar_length = 30  # Barra più lunga per maggiore dettaglio
            filled_length = int(bar_length * progress)
            
            # --- Barra colorata dinamica con gradiente ---
            progress_color_map = [C_MAGENTA, C_BLUE, C_CYAN, C_GREEN, C_YELLOW, C_RED]
            color_index = min(int(progress * len(progress_color_map)), len(progress_color_map) - 1)
            bar_color = progress_color_map[color_index]
            
            # Barra con carattere di riempimento più preciso
            partial_char = ['▏', '▎', '▍', '▌', '▋', '▊', '▉', '█']
            partial_fill = (bar_length * progress) - filled_length
            partial_index = int(partial_fill * len(partial_char))
            partial_symbol = partial_char[min(partial_index, len(partial_char) - 1)] if partial_fill > 0 and filled_length < bar_length else ''
            
            bar = f"{bar_color}{'█' * filled_length}{partial_symbol}{C_END}{'░' * (bar_length - filled_length - (1 if partial_symbol else 0))}"
            
            # Spinner organico più fluido
            spinner_organic = ['🌊', '🌀', '💫', '✨', '🔮', '💎', '⭐', '🌟']
            spinner = spinner_organic[i % len(spinner_organic)]
            
            # Frame rate color coding
            fps_color = C_GREEN if fps >= 15 else C_YELLOW if fps >= 8 else C_RED

            log_message = (
                f"\r{spinner} {C_BOLD}{C_GREEN}Natisone Trip{C_END} "
                f"{C_CYAN}[{bar}]{C_END} {C_BOLD}{progress:.1%}{C_END} "
                f"│ {fps_color}⚡{fps:.1f}fps{C_END} "
                f"│ {C_MAGENTA}⏱️{eta_str}{C_END} "
                f"│ {C_YELLOW}🎬{i+1}/{Config.TOTAL_FRAMES}{C_END}"
            )
            print(log_message, end="", flush=True)  # flush=True per aggiornamento immediato
        
        print(f"\n{C_BOLD}{C_GREEN}🌿 Cristallizzazione ULTRA completata con effetti IPNOTICI!{C_END}")
        print(f"💥 Deformazioni organiche ESAGERATE ma ultra-fluide!")
        print(f"� Traccianti DOPPI (logo rosa + sfondo viola) dinamici!")
        print(f"💎 Qualità SUPREMA (1000 DPI, smoothing perfetto)!")
        print(f"🔮 Movimento IPNOTICO e curioso - Alex Ortiga & TV Int ULTIMATE!")
        
    finally:
        # Assicurati sempre di chiudere correttamente i file video
        out.release()
        if bg_video: 
            bg_video.release()
        
        # --- AGGIUNTA AUDIO AL VIDEO ---
        if audio_data:
            print(f"\n{C_BOLD}{C_CYAN}🎵 Aggiungendo audio al video...{C_END}")
            final_output_filename = add_audio_to_video(output_filename, audio_data, Config.DURATION_SECONDS)
        else:
            final_output_filename = output_filename
            
        if Config.TEST_MODE:
            print(f"🧪 TEST - Animazione salvata in: {C_BOLD}{final_output_filename}{C_END}")
        else:
            print(f"🎬 PRODUZIONE - Animazione salvata in: {C_BOLD}{final_output_filename}{C_END}")

        # --- GESTIONE VERSIONAMENTO ---
        try:
            print(f"\n{C_BLUE}🚀 Avvio gestore di versioni...{C_END}")
            source_script_path = os.path.abspath(__file__)
            # Assicurati che il percorso di version_manager.py sia corretto
            version_manager_path = os.path.join(os.path.dirname(source_script_path), 'version_manager.py')
            
            if os.path.exists(version_manager_path):
                result = subprocess.run(
                    [sys.executable, version_manager_path, final_output_filename, source_script_path],
                    capture_output=True,
                    text=True,
                    check=False # Mettiamo a False per gestire l'errore manualmente
                )
                # Stampa sempre stdout e stderr per il debug
                print(result.stdout)
                if result.stderr:
                    # Gestisce il caso "nothing to commit" come un'informazione, non un errore
                    if "nothing to commit" in result.stderr.lower():
                         print(f"{C_GREEN}ℹ️ Nessuna nuova modifica da salvare nel versionamento.{C_END}")
                    else:
                        print(f"{C_YELLOW}Output di errore dal gestore versioni:{C_END}\n{result.stderr}")
            else:
                print(f"{C_YELLOW}ATTENZIONE: version_manager.py non trovato. Saltando il versionamento.{C_END}")

        except Exception as e:
            print(f"{C_YELLOW}Errore inatteso durante il versionamento: {e}{C_END}")

if __name__ == "__main__":
    main()