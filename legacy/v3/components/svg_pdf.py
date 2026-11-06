"""
📄 Componente SVG/PDF per CrystalPython3
Gestisce l'estrazione e il processing di contorni da file SVG e PDF.
"""

import cv2
import numpy as np
import xml.etree.ElementTree as ET

# Import condizionali per SVG
try:
    from svgpathtools import svg2paths2
    SVG_PATHTOOLS_AVAILABLE = True
except ImportError:
    SVG_PATHTOOLS_AVAILABLE = False
    print("⚠️ svgpathtools non disponibile. Per supporto SVG: pip install svgpathtools")

# Import condizionale per PDF
try:
    import fitz  # PyMuPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("⚠️ PyMuPDF non disponibile. Per supporto PDF: pip install PyMuPDF")

# Import condizionale per CairoSVG
CAIROSVG_AVAILABLE = None

# Import per spline smoothing
try:
    from scipy.interpolate import splprep, splev
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️ scipy non disponibile. Per smoothing: pip install scipy")


def get_svg_dimensions(svg_path):
    """Estrae dimensioni da file SVG."""
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
            global CAIROSVG_AVAILABLE
            if CAIROSVG_AVAILABLE is None:
                try:
                    import cairosvg
                    CAIROSVG_AVAILABLE = True
                except ImportError:
                    CAIROSVG_AVAILABLE = False
                    print("⚠️ cairosvg non disponibile. Per rendering SVG: pip install cairosvg")
            
            if not CAIROSVG_AVAILABLE:
                return extract_contours_from_svg_fallback(svg_path, width, height, padding, left_padding, logo_zoom_factor)
            
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
    if not SVG_PATHTOOLS_AVAILABLE:
        raise Exception("svgpathtools non disponibile. Installa con: pip install svgpathtools")
    
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
        try:
            from skimage.morphology import skeletonize
            skeleton = skeletonize(edges > 0)
            edges = (skeleton * 255).astype(np.uint8)
        except ImportError:
            # Se skimage non è disponibile, usa erosione alternativa
            kernel_thin = np.ones((3,3), np.uint8)
            edges = cv2.erode(edges, kernel_thin, iterations=1)
        
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
    if not SCIPY_AVAILABLE:
        return contour.astype(np.int32)
    
    if len(contour) < 4: 
        return contour
    
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
    
    # FASE 2: Analisi delle componenti connesse per identificare le spaccature
    num_labels, labels = cv2.connectedComponents(base_mask)
    
    if num_labels <= 2:  # Background + una sola componente = nessuna spaccatura
        return base_mask
    
    # FASE 3: Algoritmo di bridging per connettere componenti vicine
    # Trova tutte le componenti e i loro centroidi
    components = []
    for label in range(1, num_labels):  # Salta background (0)
        component_mask = (labels == label).astype(np.uint8) * 255
        
        # Trova contorno della componente
        contours_comp, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours_comp:
            # Calcola centroide e area
            moments = cv2.moments(contours_comp[0])
            if moments['m00'] > 0:
                cx = int(moments['m10'] / moments['m00'])
                cy = int(moments['m01'] / moments['m00'])
                area = cv2.contourArea(contours_comp[0])
                components.append({
                    'mask': component_mask,
                    'contour': contours_comp[0],
                    'centroid': (cx, cy),
                    'area': area,
                    'label': label
                })
    
    # Ordina componenti per area (più grandi first)
    components.sort(key=lambda x: x['area'], reverse=True)
    
    # FASE 4: Connessione intelligente delle componenti
    connected_mask = np.zeros((height, width), dtype=np.uint8)
    
    if components:
        # Inizia con la componente più grande
        connected_mask = components[0]['mask'].copy()
        
        # Connetti le altre componenti alla principale
        for i in range(1, len(components)):
            comp = components[i]
            
            # Calcola distanza dalla componente principale
            main_centroid = components[0]['centroid']
            comp_centroid = comp['centroid']
            distance = np.sqrt((main_centroid[0] - comp_centroid[0])**2 + 
                             (main_centroid[1] - comp_centroid[1])**2)
            
            # Se la componente è vicina e abbastanza grande, connettila
            min_area_threshold = components[0]['area'] * 0.05  # 5% dell'area principale
            max_distance_threshold = min(width, height) * 0.3  # 30% della dimensione più piccola
            
            if comp['area'] > min_area_threshold and distance < max_distance_threshold:
                # Connetti disegnando un rettangolo tra i centroidi
                cv2.rectangle(connected_mask, 
                            (min(main_centroid[0], comp_centroid[0]) - 10,
                             min(main_centroid[1], comp_centroid[1]) - 5),
                            (max(main_centroid[0], comp_centroid[0]) + 10,
                             max(main_centroid[1], comp_centroid[1]) + 5),
                            255, -1)
                
                # Aggiungi la componente
                connected_mask = cv2.bitwise_or(connected_mask, comp['mask'])
    
    # FASE 5: Post-processing finale
    # Operazioni morfologiche per perfezionare la connessione
    kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    connected_mask = cv2.morphologyEx(connected_mask, cv2.MORPH_CLOSE, kernel_smooth, iterations=3)
    connected_mask = cv2.morphologyEx(connected_mask, cv2.MORPH_OPEN, kernel_smooth, iterations=1)
    
    # Smoothing finale
    connected_mask = cv2.GaussianBlur(connected_mask, (3, 3), 0)
    _, connected_mask = cv2.threshold(connected_mask, 127, 255, cv2.THRESH_BINARY)
    
    return connected_mask
