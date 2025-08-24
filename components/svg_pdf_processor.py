"""
📐 SVG/PDF PROCESSOR - Crystal Therapy
Gestione completa di caricamento e processing di loghi SVG e PDF

Funzioni:
- Estrazione dimensioni da file SVG
- Estrazione contorni da SVG con edge detection
- Estrazione contorni da PDF con rasterizzazione
- Gestione padding, zoom e centratura
- Fallback methods per compatibilità
"""

import cv2
import numpy as np
import os
import xml.etree.ElementTree as ET

# Importazioni condizionali
try:
    from PIL import Image as PILImage, ImageDraw
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import cairosvg
    CAIROSVG_AVAILABLE = True
except ImportError:
    CAIROSVG_AVAILABLE = False

try:
    import fitz  # PyMuPDF per PDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    from svgpathtools import svg2paths2
    SVGPATHTOOLS_AVAILABLE = True
except ImportError:
    SVGPATHTOOLS_AVAILABLE = False


def get_svg_dimensions(svg_path):
    """
    📏 Estrae dimensioni da file SVG.
    
    Args:
        svg_path: Percorso del file SVG
    
    Returns:
        tuple: (width, height) in pixel
    """
    try:
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
    🎨 Estrae SOLO I CONTORNI/BORDI da un file SVG, senza riempimento.
    Utilizza rasterizzazione + edge detection per ottenere linee precise.
    
    Args:
        svg_path: Percorso del file SVG
        width: Larghezza target
        height: Altezza target  
        padding: Padding generale
        left_padding: Padding aggiuntivo dal lato sinistro per SVG
        logo_zoom_factor: Fattore di zoom del logo (1.0=normale, 2.0=doppio, 0.5=metà)
    
    Returns:
        tuple: (contours, hierarchy) per il rendering
    """
    try:
        print("🎨 Caricamento SVG Crystal Therapy dalle acque del Natisone...")
        
        # Prima prova il metodo con cairosvg se disponibile
        if CAIROSVG_AVAILABLE and PIL_AVAILABLE:
            return _extract_contours_svg_cairosvg(svg_path, width, height, padding, left_padding, logo_zoom_factor)
        
        # Fallback al metodo compatibile
        print("⚠️ CairoSVG non disponibile, usando metodo fallback")
        return extract_contours_from_svg_fallback(svg_path, width, height, padding, left_padding, logo_zoom_factor)
            
    except Exception as e:
        print(f"❌ Errore nel caricamento SVG: {e}")
        print("🔄 Tentativo con metodo fallback...")
        return extract_contours_from_svg_fallback(svg_path, width, height, padding, left_padding, logo_zoom_factor)


def _extract_contours_svg_cairosvg(svg_path, width, height, padding, left_padding=0, logo_zoom_factor=1.0):
    """
    🎨 Estrazione contorni SVG usando CairoSVG + edge detection.
    """
    import cairosvg
    import io
    
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
    
    # Applica morphological operations per migliorare i contorni
    kernel = np.ones((2,2), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # Trova contorni dalle edge
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("⚠️ Nessun contorno trovato con edge detection")
        return [], None
    
    # Filtra contorni troppo piccoli (rumore)
    min_area = 100 * scale_factor  # Soglia minima scalata
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    if not filtered_contours:
        print(f"⚠️ Tutti i contorni sotto soglia minima ({min_area})")
        filtered_contours = contours  # Usa tutti se il filtro è troppo aggressivo
    
    # Scala i contorni per adattarli alle dimensioni target
    scale_down = 1.0 / scale_factor
    scaled_contours = []
    for cnt in filtered_contours:
        scaled_cnt = cnt * scale_down
        scaled_contours.append(scaled_cnt.astype(np.int32))
    
    # Applica padding e centratura
    return _apply_svg_positioning(scaled_contours, width, height, padding, left_padding, logo_zoom_factor)


def extract_contours_from_svg_fallback(svg_path, width, height, padding, left_padding=0, logo_zoom_factor=1.0):
    """
    🔄 Metodo fallback per estrazione contorni SVG usando svgpathtools.
    
    Args:
        svg_path: Percorso del file SVG
        width: Larghezza target
        height: Altezza target
        padding: Padding generale
        left_padding: Padding aggiuntivo sinistro
        logo_zoom_factor: Fattore zoom logo
    
    Returns:
        tuple: (contours, hierarchy)
    """
    try:
        print("🔄 Usando metodo fallback con svgpathtools...")
        
        if not SVGPATHTOOLS_AVAILABLE:
            print("❌ svgpathtools non disponibile. Installare con: pip install svgpathtools")
            return [], None
        
        # Carica paths dal SVG
        paths, attributes = svg2paths2(svg_path)
        
        if not paths:
            print("⚠️ Nessun path trovato nel file SVG")
            return [], None
        
        print(f"📐 Trovati {len(paths)} path nel SVG")
        
        # Ottieni dimensioni SVG
        svg_width, svg_height = get_svg_dimensions(svg_path)
        
        # Converti paths in contorni
        all_contours = []
        
        for i, path in enumerate(paths):
            # Converti path in punti
            path_points = []
            
            # Campiona il path con punti sufficienti per i dettagli
            num_samples = max(50, int(path.length() / 5))  # Adatta alla lunghezza del path
            
            for j in range(num_samples + 1):
                t = j / num_samples
                try:
                    point = path.point(t)
                    x = point.real
                    y = point.imag
                    
                    # Scala dalle coordinate SVG alle coordinate target
                    x_scaled = (x / svg_width) * width
                    y_scaled = (y / svg_height) * height
                    
                    path_points.append([x_scaled, y_scaled])
                    
                except:
                    continue  # Salta punti problematici
            
            if len(path_points) >= 3:  # Minimo 3 punti per un contorno valido
                contour = np.array(path_points, dtype=np.int32)
                all_contours.append(contour)
        
        if not all_contours:
            print("⚠️ Nessun contorno valido estratto dai path")
            return [], None
        
        print(f"📝 Estratti {len(all_contours)} contorni dal SVG")
        
        # Applica padding e centratura
        return _apply_svg_positioning(all_contours, width, height, padding, left_padding, logo_zoom_factor)
        
    except Exception as e:
        print(f"❌ Errore nel metodo fallback SVG: {e}")
        return [], None


def _apply_svg_positioning(contours, width, height, padding, left_padding=0, logo_zoom_factor=1.0):
    """
    📐 Applica posizionamento, centratura e zoom ai contorni SVG.
    """
    if not contours:
        return [], None
    
    # Trova bounding box di tutti i contorni
    all_points = np.vstack(contours)
    x_min, y_min = np.min(all_points, axis=0)
    x_max, y_max = np.max(all_points, axis=0)
    
    logo_width = x_max - x_min
    logo_height = y_max - y_min
    
    print(f"📐 Logo SVG: {logo_width:.0f}x{logo_height:.0f}")
    
    # Calcola scala per fit con padding e zoom
    effective_padding = padding + left_padding
    available_width = width - 2 * effective_padding
    available_height = height - 2 * padding
    
    scale_x = available_width / logo_width if logo_width > 0 else 1
    scale_y = available_height / logo_height if logo_height > 0 else 1
    scale = min(scale_x, scale_y) * logo_zoom_factor
    
    # Calcola offset per centratura con left_padding
    scaled_width = logo_width * scale
    scaled_height = logo_height * scale
    
    offset_x = (width - scaled_width) / 2 + left_padding
    offset_y = (height - scaled_height) / 2
    
    # Applica trasformazione ai contorni
    transformed_contours = []
    for contour in contours:
        # Trasla per allineare all'origine
        translated = contour - [x_min, y_min]
        # Scala
        scaled = translated * scale
        # Trasla alla posizione finale
        final_contour = scaled + [offset_x, offset_y]
        
        transformed_contours.append(final_contour.astype(np.int32))
    
    print(f"📐 Logo SVG centrato e ridimensionato ({len(transformed_contours)} contorni)")
    
    # Genera hierarchy fittizia (tutti contorni esterni)
    hierarchy = np.array([[[-1, -1, -1, -1] for _ in transformed_contours]])
    
    return transformed_contours, hierarchy


def extract_contours_from_pdf(pdf_path, width, height, padding, logo_zoom_factor=1.0):
    """
    📄 Estrae i contorni da un file PDF usando il metodo corretto.
    
    Args:
        pdf_path: Percorso del file PDF
        width: Larghezza target
        height: Altezza target
        padding: Padding generale
        logo_zoom_factor: Fattore zoom logo
    
    Returns:
        tuple: (contours, hierarchy)
    """
    if not PYMUPDF_AVAILABLE:
        print("❌ PyMuPDF non disponibile. Installare con: pip install PyMuPDF")
        return [], None
    
    try:
        print("🎨 Caricamento PDF Crystal Therapy dalle acque del Natisone...")
        
        # STEP 1: Rasterizza il PDF
        doc = fitz.open(pdf_path)
        page = doc[0]  # Prima pagina
        
        # Usa scale factor 4 per alta qualità
        mat = fitz.Matrix(4, 4)  
        pix = page.get_pixmap(matrix=mat)
        
        # Converti in array numpy
        img_data = pix.tobytes("png")
        pil_img = PILImage.open(io.BytesIO(img_data))
        img_array = np.array(pil_img)
        
        doc.close()
        
        # STEP 2: Estrai contorni usando edge detection
        # Converti in BGR per OpenCV  
        if img_array.shape[2] == 4:  # RGBA
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        else:  # RGB
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Converti in scala di grigi
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Usa THRESH_BINARY_INV per ottenere contorni neri su sfondo bianco
        _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        
        # Applica morphological operations per pulire l'immagine
        kernel = np.ones((2,2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Trova contorni
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            print("⚠️ Nessun contorno trovato nel PDF")
            return [], None
        
        # STEP 3: Centra e ridimensiona i contorni
        return _apply_pdf_positioning(contours, hierarchy, width, height, padding, logo_zoom_factor)
        
    except Exception as e:
        print(f"❌ Errore nell'estrazione contorni da PDF: {e}")
        return [], None


def _apply_pdf_positioning(contours, hierarchy, width, height, padding, logo_zoom_factor=1.0):
    """
    📐 Applica posizionamento e centratura ai contorni PDF.
    """
    if not contours:
        return [], None
    
    # Filtra contorni troppo piccoli (rimuove rumore)
    min_area = 50
    filtered_contours = []
    filtered_hierarchy = []
    
    for i, contour in enumerate(contours):
        if cv2.contourArea(contour) > min_area:
            filtered_contours.append(contour)
            if hierarchy is not None:
                filtered_hierarchy.append(hierarchy[0][i])
    
    if not filtered_contours:
        print("⚠️ Tutti i contorni filtrati (troppo piccoli)")
        filtered_contours = contours
        filtered_hierarchy = hierarchy[0] if hierarchy is not None else None
    
    # Trova bounding box globale
    all_points = np.vstack(filtered_contours)
    x_min, y_min = np.min(all_points, axis=0)
    x_max, y_max = np.max(all_points, axis=0)
    
    logo_width = x_max - x_min
    logo_height = y_max - y_min
    
    print(f"📐 Logo PDF: {logo_width:.0f}x{logo_height:.0f}")
    
    # Calcola scala per fit
    available_width = width - 2 * padding
    available_height = height - 2 * padding
    
    scale_x = available_width / logo_width if logo_width > 0 else 1
    scale_y = available_height / logo_height if logo_height > 0 else 1
    scale = min(scale_x, scale_y) * logo_zoom_factor
    
    # Calcola offset per centratura
    scaled_width = logo_width * scale
    scaled_height = logo_height * scale
    
    offset_x = (width - scaled_width) / 2
    offset_y = (height - scaled_height) / 2
    
    # Applica trasformazione
    transformed_contours = []
    for contour in filtered_contours:
        # Scala dal PDF alle coordinate video ridotte di 4x
        contour_scaled = contour / 4.0
        
        # Trasla per allineare all'origine
        translated = contour_scaled - [x_min/4, y_min/4]
        
        # Scala finale
        scaled = translated * scale
        
        # Trasla alla posizione finale
        final_contour = scaled + [offset_x, offset_y]
        
        transformed_contours.append(final_contour.astype(np.int32))
    
    # Ricostruisci hierarchy se presente
    final_hierarchy = None
    if filtered_hierarchy:
        final_hierarchy = np.array([filtered_hierarchy])
    
    print(f"📝 Estratti {len(transformed_contours)} contorni dal PDF con gestione buchi")
    print(f"📐 Logo PDF centrato e ridimensionato ({len(transformed_contours)} contorni)")
    print("Estrazione contorni da PDF completata con metodo simple_logo_video.py.")
    
    return transformed_contours, final_hierarchy


# Funzioni di utilità per l'integrazione
def load_logo_contours(config, width, height):
    """
    🎨 Carica i contorni del logo dal file configurato (SVG o PDF).
    
    Args:
        config: Oggetto configurazione
        width: Larghezza target
        height: Altezza target
    
    Returns:
        tuple: (contours, hierarchy, source_info)
    """
    if config.USE_SVG_SOURCE:
        print("📄 Sorgente: SVG con path vettoriali")
        contours, hierarchy = extract_contours_from_svg(
            config.SVG_PATH, width, height, 
            config.SVG_PADDING, config.SVG_LEFT_PADDING, 
            config.LOGO_ZOOM_FACTOR
        )
        source_info = "SVG"
    else:
        print("📄 Sorgente: PDF rasterizzato con smoothing ottimizzato")
        contours, hierarchy = extract_contours_from_pdf(
            config.PDF_PATH, width, height, 
            config.SVG_PADDING, config.LOGO_ZOOM_FACTOR
        )
        source_info = "PDF"
    
    return contours, hierarchy, source_info
