import cv2
import numpy as np
import fitz  # PyMuPDF
import sys
from datetime import datetime
import random

# --- CONFIGURAZIONE MINIMALE ---

class Config:
    WIDTH = 1920
    HEIGHT = 1080
    FPS = 30
    VIDEO_DURATION_SECONDS = 2
    PDF_PATH = 'input/logo.pdf'
    LOGO_COLOR = (255, 255, 255)
    MAGIC_SYMBOLS = ['🔮', '✨', '🌟', '🌿', '🌊']

# --- FUNZIONI ESSENZIALI ---

def rasterize_pdf_to_image(pdf_path, scale=2):
    """Rasterizza la prima pagina di un PDF in un'immagine."""
    print(f"📄 Rasterizzazione PDF: {pdf_path}")
    try:
        doc = fitz.open(pdf_path)
        page = doc[0]
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        return img
    except Exception as e:
        print(f"❌ Errore durante la lettura del PDF: {e}")
        sys.exit(1)

def extract_contours_from_image(img):
    """Estrae i contorni (con gerarchia per i buchi) da un'immagine."""
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary_img = cv2.threshold(gray_img, 50, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    print(f"✅ Estratti {len(contours)} contorni.")
    return contours, hierarchy

def center_and_scale_contours(contours, target_width, target_height, padding_fraction=0.2):
    """Centra e ridimensiona i contorni per adattarli al canvas."""
    if not contours:
        return []
    all_points = np.vstack([c for c in contours])
    x, y, w, h = cv2.boundingRect(all_points)
    
    contour_center_x = x + w / 2
    contour_center_y = y + h / 2
    canvas_center_x = target_width / 2
    canvas_center_y = target_height / 2
    
    canvas_drawable_width = target_width * (1 - padding_fraction)
    canvas_drawable_height = target_height * (1 - padding_fraction)
    
    scale = min(canvas_drawable_width / w if w > 0 else 1, canvas_drawable_height / h if h > 0 else 1)

    transformed_contours = []
    for contour in contours:
        c_float = contour.astype(np.float32)
        c_float[:, :, 0] = (c_float[:, :, 0] - contour_center_x) * scale + canvas_center_x
        c_float[:, :, 1] = (c_float[:, :, 1] - contour_center_y) * scale + canvas_center_y
        transformed_contours.append(c_float.astype(np.int32))
        
    print(f"📐 Logo centrato e ridimensionato.")
    return transformed_contours

def render_contours_on_canvas(contours, hierarchy, width, height, color):
    """Disegna i contorni su un canvas, gestendo i buchi."""
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    if hierarchy is None:
        cv2.drawContours(canvas, contours, -1, color, thickness=cv2.FILLED)
        return canvas
    
    hierarchy = hierarchy[0]
    for i, contour in enumerate(contours):
        if hierarchy[i][3] == -1:  # Contorno esterno
            cv2.drawContours(canvas, [contour], -1, color, thickness=cv2.FILLED)
        else:  # Buco interno
            cv2.drawContours(canvas, [contour], -1, (0, 0, 0), thickness=cv2.FILLED)
    return canvas

def save_static_video(frame, duration_seconds, fps, width, height):
    """Salva un video statico ripetendo lo stesso frame."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    magic_symbol = random.choice(Config.MAGIC_SYMBOLS)
    output_filename = f"output/static_logo_{timestamp}_{magic_symbol}.mp4"
    
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    if not out.isOpened():
        print("❌ Errore: Impossibile creare il file video.")
        return

    print(f"\n💾 Salvataggio video: {output_filename}")
    num_frames = duration_seconds * fps
    for i in range(num_frames):
        out.write(frame)
        progress = (i + 1) / num_frames
        bar = '█' * int(progress * 20) + '-' * (20 - int(progress * 20))
        sys.stdout.write(f'\rSalvataggio: [{bar}] {progress:.0%}')
        sys.stdout.flush()

    out.release()
    print(f"\n✅ Video salvato con successo!")

# --- ESECUZIONE PRINCIPALE ---

def main():
    print("🎬 Avvio script per video statico del logo...")

    # 1. Carica e processa il logo
    img = rasterize_pdf_to_image(Config.PDF_PATH)
    contours, hierarchy = extract_contours_from_image(img)
    centered_contours = center_and_scale_contours(contours, Config.WIDTH, Config.HEIGHT)

    # 2. Disegna il logo su un frame
    logo_frame = render_contours_on_canvas(centered_contours, hierarchy, Config.WIDTH, Config.HEIGHT, Config.LOGO_COLOR)

    # 3. Salva il video
    save_static_video(logo_frame, Config.VIDEO_DURATION_SECONDS, Config.FPS, Config.WIDTH, Config.HEIGHT)

if __name__ == "__main__":
    main()
