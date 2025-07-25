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

# --- CONFIGURAZIONE GLOBALE ---

class Config:
    # --- Modalità e Qualità ---
    TEST_MODE = True # Test rapido per verificare le modifiche (SVG/PDF + lenti migliorate)
    
    # --- Compatibilità WhatsApp ---
    WHATSAPP_COMPATIBLE = True  # True = ottimizza per WhatsApp/social media
    CREATE_WHATSAPP_VERSION = True  # True = crea versione aggiuntiva con ffmpeg
    
    # --- Sorgente Logo e Texture ---
    USE_SVG_SOURCE = True  # True = SVG, False = PDF
    SVG_PATH = 'input/logo.svg'  # SVG con tracciato unificato
    PDF_PATH = 'input/logo.pdf'  # Opzione PDF alternativa
    TEXTURE_AUTO_SEARCH = True  # True = cerca automaticamente texture.tif/png/jpg
    TEXTURE_FALLBACK_PATH = 'input/26.png'  # Fallback se non trova texture.*
    TEXTURE_ENABLED = False
    TEXTURE_ALPHA = 0.5 # Leggermente più presente in alta risoluzione

    # --- Parametri Video - ADATTATI ALLE DIMENSIONI SVG + PADDING ---
    # Le dimensioni del video saranno calcolate dall'SVG + padding
    SVG_PADDING = 100  # Padding attorno all'SVG (bei bordi)
    FPS = 10 if TEST_MODE else 30
    DURATION_SECONDS = 5 if TEST_MODE else 60 # Durata normale per il rendering finale
    TOTAL_FRAMES = DURATION_SECONDS * FPS

    # --- Colore e Stile ---
    LOGO_COLOR = (50, 50, 50)   # BGR - Colore scuro per contrasto su sfondo chiaro
    LOGO_ALPHA = 1.0 # Aumentata a 1.0 per un logo solido e visibile
    
    # --- Video di Sfondo - SENZA CROP, COME BACKGROUND ORIGINALE ---
    BACKGROUND_VIDEO_PATH = 'input/sfondo.MOV'
    BG_USE_ORIGINAL_SIZE = True  # NUOVO: usa dimensioni originali senza crop
    BG_SLOWDOWN_FACTOR = 1.2     # NUOVO: rallentamento ridotto (era 2.0 = metà velocità)
    BG_DARKEN_FACTOR = 0.4       # CORRETTO: sfondo visibile ma non invadente (era 0.06)
    BG_CONTRAST_FACTOR = 1.2     # Leggermente aumentato per definizione
    
    # --- Effetto Glow (Bagliore) ---
    GLOW_ENABLED = True
    GLOW_KERNEL_SIZE = 35 if TEST_MODE else 100 # Aumentato per un glow più diffuso in HD
    GLOW_INTENSITY = 0.2

    # --- Deformazione Organica POTENZIATA (MOVIMENTO VISIBILE) ---
    DEFORMATION_ENABLED = True # RIABILITATA per ridare movimento al logo
    DEFORMATION_SPEED = 0.001 # RALLENTATO: da 0.07 a 0.05 per movimento più lento e ampio
    DEFORMATION_SCALE = 0.008 # RIDOTTO: da 0.015 a 0.008 per onde più larghe e spaziose
    DEFORMATION_INTENSITY = 12.0 # RADDOPPIATO: da 5.0 a 12.0 per deformazioni molto più ampie

    # --- Deformazione a Lenti ULTRA-CINEMATOGRAFICHE (MOVIMENTO VIVO E ORIZZONTALE) ---
    LENS_DEFORMATION_ENABLED = True # RIATTIVATA per combo effetti
    NUM_LENSES = 10 # AUMENTATO: più lenti per movimento ultra-denso e spettacolare
    LENS_MIN_STRENGTH = -2.0 # POTENZIATO: effetti ancora più drammatici
    LENS_MAX_STRENGTH = 2.5  # POTENZIATO: deformazioni ultra-spettacolari
    LENS_MIN_RADIUS = 20     # Aumentato per copertura maggiore
    LENS_MAX_RADIUS = 100    # Lenti ancora più grandi per effetti ampi
    LENS_SPEED_FACTOR = 0.001  # VELOCITÀ AUMENTATA per movimento ultra-evidente
    
    # --- PARAMETRI MOVIMENTO LENTI (NUOVI - PRECEDENTEMENTE HARDCODED) ---
    LENS_PATH_SPEED_MULTIPLIER = 8.5  # Velocità con cui le lenti seguono il loro percorso
    LENS_BASE_SPEED_MULTIPLIER = 1.4  # Moltiplicatore per la velocità di base delle lenti
    LENS_ROTATION_SPEED_MULTIPLIER = 7.0  # Velocità di rotazione delle lenti a "verme"
    LENS_INERTIA = 0.75  # Inerzia delle lenti (0.0=scattanti, 1.0=molto fluide)
    LENS_ROTATION_SPEED_MIN = -0.008  # Velocità di rotazione minima
    LENS_ROTATION_SPEED_MAX = 0.008   # Velocità di rotazione massima
    
    # --- PARAMETRI MOVIMENTO ORIZZONTALE E PULSAZIONE ULTRA-POTENZIATI ---
    LENS_HORIZONTAL_BIAS = 2  # AUMENTATO: bias ultra-forte verso movimento orizzontale lungo la scritta
    LENS_PULSATION_ENABLED = True  # Abilita pulsazione/ridimensionamento delle lenti
    LENS_PULSATION_SPEED = 0.005  # RALLENTATO: pulsazione più lenta e calma
    LENS_PULSATION_AMPLITUDE = 0.2  # AUMENTATO: pulsazione più ampia (+/-60% del raggio)
    LENS_FORCE_PULSATION_ENABLED = True  # NUOVO: anche la forza pulsa insieme al raggio
    LENS_FORCE_PULSATION_AMPLITUDE = 0.2  # NUOVO: variazione forza +/-50%
    
    WORM_SHAPE_ENABLED = True # NUOVA OPZIONE per lenti a forma di verme
    WORM_LENGTH = 2.2 # RIDOTTO: da 2.5 a 2.2 per forme più dinamiche
    WORM_COMPLEXITY = 4 # AUMENTATO: da 3 a 4 per movimento più complesso e interessante

    # --- Smussamento Contorni (QUALITÀ ULTRA-ALTA) ---
    SMOOTHING_ENABLED = True
    SMOOTHING_FACTOR = 0.0001 # ULTRA-MIGLIORATO: da 0.0008 a 0.0006 per curve perfette

    # --- Effetto Traccianti Psichedelici (ULTRA-RIDOTTI SULLA SCRITTA) ---
    TRACER_ENABLED = True
    TRACER_TRAIL_LENGTH = 25 # ULTRA-RIDOTTO: da 20 a 15 per scie minime sulla scritta
    TRACER_MAX_OPACITY = 0.03 # ULTRA-RIDOTTO: da 0.25 a 0.15 per traccianti quasi trasparenti
    TRACER_BASE_COLOR = (255, 200, 220) # Colore base (rosa/lavanda)
    TRACER_THRESHOLD1 = 50  # ULTRA-AUMENTATO: da 140 a 160 per catturare meno dettagli
    TRACER_THRESHOLD2 = 350  # ULTRA-AUMENTATO: da 300 a 350 per traccianti ultra-selettivi
    
    # --- Traccianti Sfondo (MIGLIORATI) ---
    BG_TRACER_ENABLED = True
    BG_TRACER_TRAIL_LENGTH = 20 # Scie più lunghe per lo sfondo
    BG_TRACER_MAX_OPACITY = 0.03 # AUMENTATO: da 0.25 a 0.3 per più presenza
    BG_TRACER_BASE_COLOR = (100, 70, 100) # Colore complementare viola-blu
    BG_TRACER_THRESHOLD1 = 30   # Soglie più basse per catturare più contorni dello sfondo
    BG_TRACER_THRESHOLD2 = 100
    
    # --- Blending Avanzato (SISTEMA ULTRA-POTENZIATO) ---
    ADVANCED_BLENDING = True # Abilita il blending avanzato scritta-sfondo
    LOGO_BLEND_FACTOR = 0.6 # DIMINUITO: da 0.5 a 0.3 per più fusione con sfondo
    EDGE_SOFTNESS = 80 # AUMENTATO: da 50 a 65 per transizioni ancora più graduali
    BLEND_TRANSPARENCY = 0.3 # DIMINUITO: da 0.7 a 0.4 per logo più visibile ma integrato
    COLOR_BLENDING_STRENGTH = 0.3 # DIMINUITO: da 0.9 a 0.65 per fusione colori più naturale
    
    # --- Debug e Qualità ---
    DEBUG_MASK = False  # Disabilitato per performance migliori
    
    # --- Variazione Dinamica Parametri (NUOVO SISTEMA) ---
    DYNAMIC_VARIATION_ENABLED = True
    VARIATION_AMPLITUDE = 0.8 # ±10% di variazione massima
    VARIATION_SPEED_SLOW = 0.01  # RALLENTATO: Velocità variazione lenta
    VARIATION_SPEED_MEDIUM = 0.025 # RALLENTATO: Velocità variazione media
    VARIATION_SPEED_FAST = 0.005  # RALLENTATO: Velocità variazione veloce

# --- FUNZIONI DI SUPPORTO ---

def get_dynamic_parameters(frame_index, total_frames):
    """
    Calcola i parametri degli effetti in modo dinamico per il frame corrente.
    Questo crea un'evoluzione armonica degli effetti nel tempo con variazioni casuali.
    """
    # t è il progresso normalizzato dell'animazione (da 0.0 a 1.0)
    t = frame_index / total_frames
    
    params = {}

    # 1. Dinamica del Glow: pulsazione più lenta e armonica
    glow_pulse = np.sin(t * np.pi) # Un solo ciclo completo, molto più lento
    params['glow_intensity'] = Config.GLOW_INTENSITY + (glow_pulse * 0.2) # Variazione più sottile

    # 2. NUOVO: Variazioni dinamiche dei parametri principali (±10%)
    if Config.DYNAMIC_VARIATION_ENABLED:
        # Seed basato sul frame per variazioni coerenti ma pseudo-casuali
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
        # Valori statici se la variazione è disabilitata
        params['deformation_speed'] = Config.DEFORMATION_SPEED
        params['deformation_scale'] = Config.DEFORMATION_SCALE
        params['deformation_intensity'] = Config.DEFORMATION_INTENSITY
        params['lens_speed_factor'] = Config.LENS_SPEED_FACTOR
        params['lens_strength_multiplier'] = 1.0
        params['tracer_opacity_multiplier'] = 1.0
        params['bg_tracer_opacity_multiplier'] = 1.0
    
    return params

def get_timestamp_filename():
    """Genera un nome file con un timestamp magico."""
    now = datetime.datetime.now()
    # Aggiunti caratteri "magici" come richiesto
    magic_chars = ['α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'ι', 'κ', 'λ', 'μ', 'ν', 'ξ', 'ο', 'π', 'ρ', 'σ', 'τ', 'υ', 'φ', 'χ', 'ψ', 'ω', 'ॐ', '☯', '✨', 'Δ', 'Σ', 'Ω']
    magic_char = np.random.choice(magic_chars)
    return f"output/crystalpy_{now.strftime('%Y%m%d_%H%M%S')}_{magic_char}.mp4"

def load_texture(texture_path, width, height):
    """Carica e ridimensiona l'immagine di texture."""
    if not os.path.exists(texture_path):
        print(f"ATTENZIONE: File texture non trovato in '{texture_path}'. Il logo non verrà texturizzato.")
        return None
    try:
        # --- LOG PERSONALIZZATO MAGICO ---
        print("Analisi texture scienziato TV Int dalle acque del Natisone... completata.")
        texture = cv2.imread(texture_path, cv2.IMREAD_COLOR)
        if texture is None:
            raise Exception("cv2.imread ha restituito None.")
        # Ridimensiona la texture per adattarla al frame
        return cv2.resize(texture, (width, height), interpolation=cv2.INTER_LINEAR)
    except Exception as e:
        print(f"Errore durante il caricamento della texture: {e}")
        return None

def get_svg_dimensions(svg_path):
    """
    Estrae le dimensioni dall'SVG per calcolare le dimensioni del video.
    """
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

def extract_contours_from_svg(svg_path, width, height, padding):
    """
    Estrae SOLO I CONTORNI/BORDI da un file SVG, senza riempimento.
    Utilizza rasterizzazione + edge detection per ottenere linee precise.
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
            return extract_contours_from_svg_fallback(svg_path, width, height, padding)
            
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
        return extract_contours_from_svg_fallback(svg_path, width, height, padding)

def extract_contours_from_svg_fallback(svg_path, width, height, padding):
    """
    Metodo per l'estrazione SVG con OPZIONE per SOLO CONTORNI senza riempimento.
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
        scale = min(target_w / svg_width, target_h / svg_height)
        
        # Calcola offset per centrare
        scaled_w = svg_width * scale
        scaled_h = svg_height * scale
        offset_x = (width - scaled_w) / 2
        offset_y = (height - scaled_h) / 2
        
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

def extract_contours_from_pdf(pdf_path, width, height, padding):
    """
    Estrae i contorni da un file PDF e li converte in contorni OpenCV.
    """
    if not PDF_AVAILABLE:
        raise Exception("PyMuPDF non disponibile. Installa con: pip install PyMuPDF")
    
    try:
        print("🎨 Caricamento PDF Crystal Therapy dalle acque del Natisone...")
        
        # Apri il PDF
        doc = fitz.open(pdf_path)
        page = doc[0]  # Prima pagina
        
        # Converti in immagine con alta risoluzione
        zoom_factor = 4  # Aumenta per più dettaglio
        matrix = fitz.Matrix(zoom_factor, zoom_factor)
        pix = page.get_pixmap(matrix=matrix)
        
        # Converti in array numpy
        img_data = pix.tobytes("png")
        import io
        from PIL import Image as PILImage
        
        pil_image = PILImage.open(io.BytesIO(img_data))
        img_array = np.array(pil_image)
        
        # Se l'immagine è in RGBA, convertila in RGB
        if img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        
        # Converti in BGR per OpenCV
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Converti in scala di grigi
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Inverti per avere il logo in bianco su sfondo nero
        gray = 255 - gray
        
        # Soglia per ottenere una maschera binaria
        _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        
        # Trova i contorni
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            raise Exception("Nessun contorno trovato nel PDF.")
        
        # Filtra e processa i contorni
        processed_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Filtra contorni troppo piccoli
                processed_contours.append(contour)
        
        if not processed_contours:
            raise Exception("Nessun contorno valido trovato nel PDF.")
        
        # Ridimensiona i contorni alla risoluzione target
        scale_x = width / binary.shape[1]
        scale_y = height / binary.shape[0]
        
        scaled_contours = []
        for contour in processed_contours:
            scaled_contour = contour.copy().astype(np.float32)
            scaled_contour[:, 0, 0] *= scale_x
            scaled_contour[:, 0, 1] *= scale_y
            scaled_contours.append(scaled_contour.astype(np.int32))
        
        doc.close()
        
        print(f"📐 Estratti {len(scaled_contours)} contorni da PDF")
        print("Estrazione contorni da PDF completata.")
        
        return scaled_contours, None  # Hierarchy non necessaria per PDF
        
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

def apply_lens_deformation(mask, lenses, frame_index, config, dynamic_params=None):
    """
    Applica una deformazione basata su "lenti" che seguono percorsi cinematografici predefiniti.
    Sistema completamente rivisto per movimenti ampi, fluidi e cinematografici.
    """
    h, w = mask.shape
    
    # Ottieni moltiplicatori dinamici se disponibili
    lens_strength_mult = dynamic_params.get('lens_strength_multiplier', 1.0) if dynamic_params else 1.0
    
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
            
            # Aggiungiamo noise per creare le curve del "verme"
            noise_val = pnoise2(
                (lens['pos'][0] + frame_index * 2) * 0.01 * config.WORM_COMPLEXITY, 
                (lens['pos'][1] + frame_index * 2) * 0.01 * config.WORM_COMPLEXITY, 
                octaves=1
            )
            dy_scaled = dy_rot + noise_val * 50
            
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
            
            # Pulsazione del raggio con pattern complesso per più "vita"
            base_pulsation = np.sin(pulsation_time)
            secondary_pulsation = 0.3 * np.sin(pulsation_time * 2.7)  # Frequenza diversa
            tertiary_pulsation = 0.15 * np.cos(pulsation_time * 4.1)  # Ancora più complessa
            
            total_pulsation = base_pulsation + secondary_pulsation + tertiary_pulsation
            pulsation_factor = 1.0 + config.LENS_PULSATION_AMPLITUDE * total_pulsation
            lens['radius'] = lens['base_radius'] * pulsation_factor
            
            # NUOVO: Pulsazione anche della forza per effetto drammatico
            if config.LENS_FORCE_PULSATION_ENABLED:
                force_pulsation_time = pulsation_time * 1.8  # Velocità leggermente diversa
                force_pulsation = np.sin(force_pulsation_time) + 0.5 * np.cos(force_pulsation_time * 1.6)
                force_factor = 1.0 + config.LENS_FORCE_PULSATION_AMPLITUDE * force_pulsation
                lens['strength'] = lens['base_strength'] * force_factor
        
        # === MOVIMENTO LUNGO PERCORSI CINEMATOGRAFICI ULTRA-VELOCE ===
        # Velocità configurabile tramite parametri della Config
        movement_speed_multiplier = config.LENS_PATH_SPEED_MULTIPLIER
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
            # Velocità configurabile con adattamento dinamico alla distanza
            base_speed = config.LENS_SPEED_FACTOR * config.LENS_BASE_SPEED_MULTIPLIER
            adaptive_speed = base_speed * (1.0 + 0.5 * min(distance_to_target / 40, 1.5))
            desired_velocity = (direction / distance_to_target) * adaptive_speed
            
            # Inerzia configurabile per controllo del movimento
            inertia_strength = config.LENS_INERTIA
            lens['velocity'] = lens['velocity'] * inertia_strength + desired_velocity * (1 - inertia_strength)
        
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
        
        # Nuove dimensioni scalate
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Ridimensiona
        scaled_bg = cv2.resize(bg_frame, (new_w, new_h))
        
        # Centro-crop per adattare alle dimensioni esatte
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

def render_frame(contours, hierarchy, width, height, frame_index, total_frames, config, bg_frame, texture_image, tracer_history, bg_tracer_history, lenses):
    """
    Rende un singolo frame dell'animazione, applicando la pipeline di effetti completa.
    """
    # --- 0. Ottieni Parametri Dinamici ---
    dynamic_params = get_dynamic_parameters(frame_index, total_frames)

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

    # --- 4. Applica Deformazione Organica (per movimento di base) ---
    if config.DEFORMATION_ENABLED:
        # Usiamo parametri fissi per un "respiro" costante
        deformation_params = {
            'speed': config.DEFORMATION_SPEED,
            'scale': config.DEFORMATION_SCALE,
            'intensity': config.DEFORMATION_INTENSITY
        }
        logo_mask = apply_organic_deformation(logo_mask, frame_index, deformation_params)

    # --- 5. Applica Deformazione a Lenti (sovrapposta alla prima) ---
    if config.LENS_DEFORMATION_ENABLED:
        logo_mask = apply_lens_deformation(logo_mask, lenses, frame_index, config, dynamic_params)

    # --- 5.5. Estrai Traccianti del Logo (NUOVO per maggiore aderenza) ---
    logo_tracers = extract_logo_tracers(logo_mask, config)
    # Combina i traccianti del logo con quelli dello sfondo per un effetto più ricco
    combined_logo_edges = cv2.add(current_logo_edges, logo_tracers)

    # --- 6. Creazione Layer Logo e Glow ---
    logo_layer = np.zeros_like(final_frame)
    glow_layer = np.zeros_like(final_frame)

    # Applica la texture al logo se disponibile e abilitata
    if config.TEXTURE_ENABLED and texture_image is not None:
        # Crea una base di colore solido
        solid_color_layer = np.zeros_like(final_frame)
        solid_color_layer[logo_mask > 0] = config.LOGO_COLOR
        # Crea la texture mascherata
        textured_logo_masked = cv2.bitwise_and(texture_image, texture_image, mask=logo_mask)
        # Fonde il colore e la texture con alpha
        logo_layer = cv2.addWeighted(solid_color_layer, 1.0 - config.TEXTURE_ALPHA, textured_logo_masked, config.TEXTURE_ALPHA, 0)
    else:
        # Usa colore solido se la texture è disabilitata o non trovata
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
    Applica un blending avanzato tra la scritta e lo sfondo per un effetto più naturale e organico.
    """
    # 1. Crea una maschera sfumata per i bordi del logo
    edge_softness = config.EDGE_SOFTNESS
    # Assicura che il kernel sia dispari
    if edge_softness % 2 == 0:
        edge_softness += 1
    soft_mask = cv2.GaussianBlur(logo_mask.astype(np.float32), (edge_softness, edge_softness), 0) / 255.0
    
    # 2. Converti la maschera in 3 canali per il broadcasting
    soft_mask_3ch = cv2.merge([soft_mask, soft_mask, soft_mask])
    bg_mask_3ch = 1.0 - soft_mask_3ch
    
    # 3. Converti tutto in float32 per calcoli precisi
    bg_frame_f = background_frame.astype(np.float32)
    logo_layer_f = logo_layer.astype(np.float32)
    
    # 4. Estrai colori dello sfondo nell'area del logo per il color blending
    logo_area_bg = cv2.bitwise_and(background_frame, background_frame, mask=logo_mask)
    logo_area_bg_f = logo_area_bg.astype(np.float32)
    
    # 5. Crea una versione del logo che incorpora i colori dello sfondo
    blended_logo = logo_layer_f.copy()
    
    # Mescola i colori del logo con quelli dello sfondo nella zona del logo
    if config.COLOR_BLENDING_STRENGTH > 0:
        color_blend_factor = config.COLOR_BLENDING_STRENGTH
        # Solo nell'area del logo, mescola i colori
        logo_area_mask = logo_mask > 0
        if np.any(logo_area_mask):
            # Media ponderata tra colore logo e colore sfondo
            blended_logo[logo_area_mask] = (
                blended_logo[logo_area_mask] * (1.0 - color_blend_factor) +
                logo_area_bg_f[logo_area_mask] * color_blend_factor
            )
    
    # 6. Applica trasparenza al logo se configurata
    logo_alpha = 1.0 - config.BLEND_TRANSPARENCY
    blended_logo = blended_logo * logo_alpha + bg_frame_f * config.BLEND_TRANSPARENCY
    
    # 7. Applica il blending finale usando le maschere sfumate a 3 canali
    final_result = (
        bg_frame_f * bg_mask_3ch +  # Sfondo nelle aree non-logo (con sfumatura)
        blended_logo * soft_mask_3ch * config.LOGO_BLEND_FACTOR +  # Logo blended
        logo_layer_f * soft_mask_3ch * (1.0 - config.LOGO_BLEND_FACTOR)  # Logo originale
    )
    
    # 8. Riconverti a uint8 e ritorna
    return np.clip(final_result, 0, 255).astype(np.uint8)


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

def main():
    """Funzione principale per generare l'animazione del logo."""
    # --- Codici ANSI per colori e stili nel terminale ---
    C_CYAN = '\033[96m'
    C_GREEN = '\033[92m'
    C_YELLOW = '\033[93m'
    C_BLUE = '\033[94m'
    C_MAGENTA = '\033[95m'
    C_BOLD = '\033[1m'
    C_END = '\033[0m'
    SPINNER_CHARS = ['🔮', '✨', '🌟', '💎']

    # NUOVO: Calcola dimensioni del video dalle dimensioni SVG + padding
    svg_width, svg_height = get_svg_dimensions(Config.SVG_PATH)
    Config.WIDTH = svg_width + (Config.SVG_PADDING * 2)
    Config.HEIGHT = svg_height + (Config.SVG_PADDING * 2)
    
    print(f"{C_BOLD}{C_CYAN}🌊 Avvio rendering Crystal Therapy - SVG CENTRATO...{C_END}")
    print(f"📐 Dimensioni SVG: {svg_width}x{svg_height}")
    print(f"📐 Dimensioni video: {Config.WIDTH}x{Config.HEIGHT} (padding: {Config.SVG_PADDING}px)")
    print(f"� TEST MODE: 30fps, {Config.DURATION_SECONDS}s")
    source_type = "SVG vettoriale" if Config.USE_SVG_SOURCE else "PDF rasterizzato"
    print(f"� Sorgente: {source_type} con smoothing ottimizzato")
    print(f"� Video sfondo: ORIGINALE senza crop, rallentato {Config.BG_SLOWDOWN_FACTOR}x")
    print(f"✨ Traccianti + Blending + Glow COMPATIBILE")
    print(f"� Variazione dinamica + codec video testati")
    print(f"💎 RENDERING MOVIMENTO GARANTITO per compatibilità VLC/QuickTime!")
    
    # Carica contorni da SVG o PDF
    if Config.USE_SVG_SOURCE:
        contours, hierarchy = extract_contours_from_svg(Config.SVG_PATH, Config.WIDTH, Config.HEIGHT, Config.SVG_PADDING)
    else:
        contours, hierarchy = extract_contours_from_pdf(Config.PDF_PATH, Config.WIDTH, Config.HEIGHT, Config.SVG_PADDING)

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
            print("Texture infusa con l'essenza del Natisone - Alex Ortiga.")
    else:
        print("La texturizzazione del logo è disabilitata.")

    # --- Apertura Video di Sfondo ---
    bg_video = cv2.VideoCapture(Config.BACKGROUND_VIDEO_PATH)
    if not bg_video.isOpened():
        print(f"Errore: impossibile aprire il video di sfondo in {Config.BACKGROUND_VIDEO_PATH}")
        # Crea uno sfondo nero di fallback
        bg_video = None
    else:
        # NUOVO: Ottieni informazioni del video di sfondo per il rallentamento
        bg_total_frames = int(bg_video.get(cv2.CAP_PROP_FRAME_COUNT))
        bg_fps = bg_video.get(cv2.CAP_PROP_FPS)
        print(f"🎬 Video sfondo: {bg_total_frames} frame @ {bg_fps}fps")
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
        print(f"🌊 Liberate {len(lenses)} creature liquide dal Natisone per Alex Ortiga.")

    print(f"Rendering dell'animazione in corso... ({Config.TOTAL_FRAMES} frame da elaborare)")
    start_time = time.time()
    
    try:
        for i in range(Config.TOTAL_FRAMES):
            # --- Gestione Frame di Sfondo con RALLENTAMENTO ---
            if bg_video:
                # NUOVO: Calcola il frame del video di sfondo rallentato
                # Frame rallentato: i / BG_SLOWDOWN_FACTOR
                bg_frame_index = int(i / Config.BG_SLOWDOWN_FACTOR)  # Rallentamento configurabile
                
                # Imposta la posizione nel video di sfondo
                bg_video.set(cv2.CAP_PROP_POS_FRAMES, bg_frame_index)
                ret, bg_frame = bg_video.read()
                
                # Se arriviamo alla fine del video, riavvolgi
                if not ret:
                    bg_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, bg_frame = bg_video.read()
                # RIMOSSO: Non ridimensionare qui, lo fa process_background
                # bg_frame = cv2.resize(bg_frame, (Config.WIDTH, Config.HEIGHT))
            else:
                # Crea uno sfondo nero se non c'è video
                bg_frame = np.zeros((Config.HEIGHT, Config.WIDTH, 3), dtype=np.uint8)

            frame_result = render_frame(contours, hierarchy, Config.WIDTH, Config.HEIGHT, i, Config.TOTAL_FRAMES, Config, bg_frame, texture_image, tracer_history, bg_tracer_history, lenses)
            
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
            
            # --- Log di Avanzamento Magico (aggiornamento per frame) ---
            elapsed = time.time() - start_time
            fps = (i + 1) / elapsed if elapsed > 0 else 0
            
            # Calcolo ETA
            remaining_frames = Config.TOTAL_FRAMES - (i + 1)
            eta_seconds = remaining_frames / fps if fps > 0 else 0
            eta_minutes, eta_sec = divmod(int(eta_seconds), 60)
            eta_str = f"{eta_minutes:02d}:{eta_sec:02d}"

            # Barra di avanzamento
            progress = (i + 1) / Config.TOTAL_FRAMES
            bar_length = 25
            filled_length = int(bar_length * progress)
            
            # --- NUOVO: Barra colorata dinamica ---
            progress_color_map = [C_MAGENTA, C_BLUE, C_CYAN, C_GREEN, C_YELLOW]
            color_index = int(progress * (len(progress_color_map) -1))
            bar_color = progress_color_map[color_index]
            bar = f"{bar_color}{'█' * filled_length}{C_END}{'-' * (bar_length - filled_length)}"
            
            # Spinner magico
            spinner = SPINNER_CHARS[i % len(SPINNER_CHARS)]

            log_message = (
                f"\r{spinner} {C_BOLD}{C_GREEN}Cristallizzazione...{C_END} "
                f"{C_CYAN}[{bar}] {C_END}{progress:.1%} "
                f"| {C_YELLOW}FPS: {fps:.2f}{C_END} "
                f"| {C_MAGENTA}ETA: {eta_str}{C_END} "
            )
            print(log_message, end="")
        
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
        print(f"Animazione salvata in: {C_BOLD}{output_filename}{C_END}")

        # --- GESTIONE VERSIONAMENTO ---
        try:
            print(f"\n{C_BLUE}🚀 Avvio gestore di versioni...{C_END}")
            source_script_path = os.path.abspath(__file__)
            # Assicurati che il percorso di version_manager.py sia corretto
            version_manager_path = os.path.join(os.path.dirname(source_script_path), 'version_manager.py')
            
            if os.path.exists(version_manager_path):
                result = subprocess.run(
                    [sys.executable, version_manager_path, output_filename, source_script_path],
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