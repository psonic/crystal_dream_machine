"""
🌊 LIVE PREVIEW MODE - Crystal Therapy
Sistema di anteprima in tempo reale per sviluppo creativo

Funzionalità:
- Mostra un frame generato in tempo reale in una finestra
- Auto-refresh ogni 5 secondi
- Hot-reload di sfondo.MOV e texture.jpg quando modificati
- Premere SPAZIO per generare il video completo e fare Git push
- Premere ESC per uscire dalla preview
"""

import cv2
import numpy as np
import os
import time
import threading
from collections import deque

class LivePreview:
    def __init__(self, config, render_frame_func, contours, hierarchy, 
                 width, height, get_background_func, get_texture_func,
                 initialize_lenses_func, load_audio_func=None):
        """
        Inizializza il sistema Live Preview
        
        Args:
            config: Oggetto configurazione
            render_frame_func: Funzione per renderizzare un frame
            contours: Contorni del logo
            hierarchy: Gerarchia contorni
            width, height: Dimensioni video
            get_background_func: Funzione per ottenere frame di sfondo
            get_texture_func: Funzione per caricare texture
            initialize_lenses_func: Funzione per inizializzare lenti
            load_audio_func: Funzione per caricare audio (opzionale)
        """
        self.config = config
        self.render_frame_func = render_frame_func
        self.contours = contours
        self.hierarchy = hierarchy
        self.width = width
        self.height = height
        self.get_background_func = get_background_func
        self.get_texture_func = get_texture_func
        self.initialize_lenses_func = initialize_lenses_func
        self.load_audio_func = load_audio_func
        
        # Stato interno
        self.is_running = False
        self.current_frame = None
        self.frame_counter = 0
        self.last_refresh_time = 0
        self.refresh_interval = 1.0  # 1 secondo per aggiornamenti più fluidi
        self.should_render_video = False
        self.force_refresh = False  # Per forzare il refresh quando si cambiano parametri
        
        # Monitoring file per hot-reload
        self.bg_video_path = config.BACKGROUND_VIDEO_PATH
        self.texture_path = None
        self.last_bg_mtime = 0
        self.last_texture_mtime = 0
        
        # Stato rendering
        self.bg_video = None
        self.texture_image = None
        self.lenses = []
        self.tracer_history = deque(maxlen=config.TRACER_TRAIL_LENGTH)
        self.bg_tracer_history = deque(maxlen=getattr(config, 'BG_TRACER_TRAIL_LENGTH', 35))
        self.audio_data = None
        
        # Trova texture iniziale
        self._find_texture_file()
        
        # Backup parametri originali per reset
        self._backup_original_params()
        
        print("🌊 Live Preview inizializzata!")
        print("   📺 Finestra: Crystal Therapy - Live Preview")
        print("   🔄 Auto-refresh: ogni 1 secondo")
        print("   🎬 SPAZIO: genera video completo + Git push")
        print("   🎚️ CONTROLLI PARAMETRI:")
        print("      Q/A: Deformation Intensity (±2.0)")
        print("      W/S: Glow Intensity (±0.1)")
        print("      E/D: Lens Speed (±0.02)")
        print("      R/F: Background Zoom (±0.1)")
        print("      T/G: Logo Zoom (±0.1)")
        print("      Y/H: Tracer Opacity (±0.01)")
        print("      U/J: Blending Strength (±0.1)")
        print("   🔄 CTRL+R: Reset parametri")
        print("   ❌ ESC: esci dalla preview")
        
    def _find_texture_file(self):
        """Trova il file texture disponibile"""
        base_path = 'input/texture'
        extensions = ['tif', 'png', 'jpg', 'jpeg']
        
        for ext in extensions:
            texture_path = f"{base_path}.{ext}"
            if os.path.exists(texture_path):
                self.texture_path = texture_path
                return
        
        # Fallback
        if os.path.exists(self.config.TEXTURE_FALLBACK_PATH):
            self.texture_path = self.config.TEXTURE_FALLBACK_PATH
    
    def _backup_original_params(self):
        """Salva i parametri originali per il reset"""
        self.original_params = {
            'DEFORMATION_INTENSITY': self.config.DEFORMATION_INTENSITY,
            'GLOW_INTENSITY': self.config.GLOW_INTENSITY,
            'LENS_SPEED_FACTOR': self.config.LENS_SPEED_FACTOR,
            'BG_ZOOM_FACTOR': self.config.BG_ZOOM_FACTOR,
            'LOGO_ZOOM_FACTOR': self.config.LOGO_ZOOM_FACTOR,
            'TRACER_MAX_OPACITY': self.config.TRACER_MAX_OPACITY,
            'BLENDING_STRENGTH': self.config.BLENDING_STRENGTH,
        }
        
    def _reset_params(self):
        """Ripristina i parametri originali"""
        for param, value in self.original_params.items():
            setattr(self.config, param, value)
        self.force_refresh = True
        print("🔄 Parametri ripristinati ai valori originali")
        
    def _handle_parameter_controls(self, key):
        """Gestisce i controlli per modificare i parametri in tempo reale"""
        changed = False
        
        # Deformation Intensity (Q/A)
        if key == ord('q') or key == ord('Q'):
            self.config.DEFORMATION_INTENSITY = max(0.5, self.config.DEFORMATION_INTENSITY + 2.0)
            print(f"🌊 Deformation Intensity: {self.config.DEFORMATION_INTENSITY:.1f}")
            changed = True
        elif key == ord('a') or key == ord('A'):
            self.config.DEFORMATION_INTENSITY = max(0.5, self.config.DEFORMATION_INTENSITY - 2.0)
            print(f"🌊 Deformation Intensity: {self.config.DEFORMATION_INTENSITY:.1f}")
            changed = True
            
        # Glow Intensity (W/S)
        elif key == ord('w') or key == ord('W'):
            self.config.GLOW_INTENSITY = min(1.0, self.config.GLOW_INTENSITY + 0.1)
            print(f"✨ Glow Intensity: {self.config.GLOW_INTENSITY:.2f}")
            changed = True
        elif key == ord('s') or key == ord('S'):
            self.config.GLOW_INTENSITY = max(0.0, self.config.GLOW_INTENSITY - 0.1)
            print(f"✨ Glow Intensity: {self.config.GLOW_INTENSITY:.2f}")
            changed = True
            
        # Lens Speed (E/D)
        elif key == ord('e') or key == ord('E'):
            self.config.LENS_SPEED_FACTOR = min(0.5, self.config.LENS_SPEED_FACTOR + 0.02)
            print(f"🔮 Lens Speed: {self.config.LENS_SPEED_FACTOR:.3f}")
            changed = True
        elif key == ord('d') or key == ord('D'):
            self.config.LENS_SPEED_FACTOR = max(0.005, self.config.LENS_SPEED_FACTOR - 0.02)
            print(f"🔮 Lens Speed: {self.config.LENS_SPEED_FACTOR:.3f}")
            changed = True
            
        # Background Zoom (R/F)
        elif key == ord('r') or key == ord('R'):
            self.config.BG_ZOOM_FACTOR = min(3.0, self.config.BG_ZOOM_FACTOR + 0.1)
            print(f"🎬 Background Zoom: {self.config.BG_ZOOM_FACTOR:.2f}")
            changed = True
        elif key == ord('f') or key == ord('F'):
            self.config.BG_ZOOM_FACTOR = max(0.5, self.config.BG_ZOOM_FACTOR - 0.1)
            print(f"🎬 Background Zoom: {self.config.BG_ZOOM_FACTOR:.2f}")
            changed = True
            
        # Logo Zoom (T/G)
        elif key == ord('t') or key == ord('T'):
            self.config.LOGO_ZOOM_FACTOR = min(3.0, self.config.LOGO_ZOOM_FACTOR + 0.1)
            print(f"📐 Logo Zoom: {self.config.LOGO_ZOOM_FACTOR:.2f}")
            changed = True
        elif key == ord('g') or key == ord('G'):
            self.config.LOGO_ZOOM_FACTOR = max(0.3, self.config.LOGO_ZOOM_FACTOR - 0.1)
            print(f"📐 Logo Zoom: {self.config.LOGO_ZOOM_FACTOR:.2f}")
            changed = True
            
        # Tracer Opacity (Y/H)
        elif key == ord('y') or key == ord('Y'):
            self.config.TRACER_MAX_OPACITY = min(0.2, self.config.TRACER_MAX_OPACITY + 0.01)
            print(f"🌈 Tracer Opacity: {self.config.TRACER_MAX_OPACITY:.3f}")
            changed = True
        elif key == ord('h') or key == ord('H'):
            self.config.TRACER_MAX_OPACITY = max(0.0, self.config.TRACER_MAX_OPACITY - 0.01)
            print(f"🌈 Tracer Opacity: {self.config.TRACER_MAX_OPACITY:.3f}")
            changed = True
            
        # Blending Strength (U/J)
        elif key == ord('u') or key == ord('U'):
            self.config.BLENDING_STRENGTH = min(1.0, self.config.BLENDING_STRENGTH + 0.1)
            print(f"🎨 Blending Strength: {self.config.BLENDING_STRENGTH:.2f}")
            changed = True
        elif key == ord('j') or key == ord('J'):
            self.config.BLENDING_STRENGTH = max(0.0, self.config.BLENDING_STRENGTH - 0.1)
            print(f"🎨 Blending Strength: {self.config.BLENDING_STRENGTH:.2f}")
            changed = True
        
        if changed:
            self.force_refresh = True
            # Se cambiano zoom factors che influenzano i contorni, potremmo dover ricaricare
            if key in [ord('t'), ord('T'), ord('g'), ord('G')]:
                print("⚠️ Cambio zoom logo - Ricaricamento contorni necessario al prossimo rendering completo")
                
        return changed
        
    def _check_file_changes(self):
        """Controlla se i file sono stati modificati"""
        changes = False
        
        # Controlla video di sfondo
        if os.path.exists(self.bg_video_path):
            mtime = os.path.getmtime(self.bg_video_path)
            if mtime != self.last_bg_mtime:
                self.last_bg_mtime = mtime
                changes = True
                print("🎬 Rilevato cambiamento in sfondo.MOV - Ricaricando...")
                
        # Controlla texture
        if self.texture_path and os.path.exists(self.texture_path):
            mtime = os.path.getmtime(self.texture_path)
            if mtime != self.last_texture_mtime:
                self.last_texture_mtime = mtime
                changes = True
                print("🎨 Rilevato cambiamento in texture - Ricaricando...")
        
        return changes
        
    def _reload_resources(self):
        """Ricarica le risorse modificate"""
        try:
            # Ricarica video di sfondo
            if self.bg_video:
                self.bg_video.release()
            self.bg_video = cv2.VideoCapture(self.bg_video_path)
            
            # Ricarica texture
            if self.texture_path:
                self.texture_image = self.get_texture_func(self.texture_path, self.width, self.height)
            
            # Ricarica audio se disponibile
            if self.load_audio_func:
                self.audio_data = self.load_audio_func(
                    self.config.AUDIO_FILES,
                    self.config.DURATION_SECONDS,
                    self.config.FPS,
                    self.config.AUDIO_RANDOM_SELECTION,
                    self.config.AUDIO_RANDOM_START
                )
            
            print("✅ Risorse ricaricate con successo")
            
        except Exception as e:
            print(f"⚠️ Errore nel ricaricamento risorse: {e}")
    
    def _initialize_rendering_state(self):
        """Inizializza lo stato per il rendering"""
        # Carica video di sfondo
        self.bg_video = cv2.VideoCapture(self.bg_video_path)
        
        # Carica texture
        if self.texture_path:
            self.texture_image = self.get_texture_func(self.texture_path, self.width, self.height)
        
        # Inizializza lenti
        if self.config.LENS_DEFORMATION_ENABLED:
            self.lenses = self.initialize_lenses_func(self.config)
        
        # Carica audio se disponibile
        if self.load_audio_func:
            self.audio_data = self.load_audio_func(
                self.config.AUDIO_FILES,
                self.config.DURATION_SECONDS, 
                self.config.FPS,
                self.config.AUDIO_RANDOM_SELECTION,
                self.config.AUDIO_RANDOM_START
            )
        
        # Imposta timestamp iniziali per hot-reload
        if os.path.exists(self.bg_video_path):
            self.last_bg_mtime = os.path.getmtime(self.bg_video_path)
        if self.texture_path and os.path.exists(self.texture_path):
            self.last_texture_mtime = os.path.getmtime(self.texture_path)
            
    def _generate_preview_frame(self):
        """Genera un singolo frame per la preview"""
        try:
            # Ottieni frame di sfondo
            bg_frame = self.get_background_func(self.bg_video, self.frame_counter)
            
            # Renderizza il frame
            frame_result = self.render_frame_func(
                self.contours, self.hierarchy, self.width, self.height,
                self.frame_counter, self.config.TOTAL_FRAMES, self.config,
                bg_frame, self.texture_image, self.tracer_history, 
                self.bg_tracer_history, self.lenses, self.audio_data
            )
            
            # Estrai risultati
            if len(frame_result) == 3:
                frame, current_logo_edges, current_bg_edges = frame_result
            else:
                frame, current_logo_edges = frame_result
                current_bg_edges = None
            
            # Aggiorna traccianti
            if self.config.TRACER_ENABLED:
                self.tracer_history.append(current_logo_edges)
            
            if hasattr(self.config, 'BG_TRACER_ENABLED') and self.config.BG_TRACER_ENABLED and current_bg_edges is not None:
                self.bg_tracer_history.append(current_bg_edges)
            
            # Incrementa contatore frame per animazione
            self.frame_counter = (self.frame_counter + 1) % self.config.TOTAL_FRAMES
            
            return frame
            
        except Exception as e:
            print(f"⚠️ Errore nella generazione frame: {e}")
            # Ritorna frame nero in caso di errore
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
    
    def _add_overlay_info(self, frame):
        """Aggiunge informazioni overlay al frame"""
        overlay = frame.copy()
        
        # Testo informativo
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        # Header
        cv2.putText(overlay, "🌊 CRYSTAL THERAPY - LIVE PREVIEW", (10, 30), 
                   font, font_scale, (0, 255, 255), thickness)
        
        # Informazioni frame
        frame_info = f"Frame: {self.frame_counter}/{self.config.TOTAL_FRAMES}"
        cv2.putText(overlay, frame_info, (10, 60), 
                   font, 0.5, (255, 255, 255), 1)
        
        # Tempo prossimo refresh
        time_since_refresh = time.time() - self.last_refresh_time
        time_to_refresh = max(0, self.refresh_interval - time_since_refresh)
        refresh_info = f"Prossimo refresh: {time_to_refresh:.1f}s"
        cv2.putText(overlay, refresh_info, (10, 80), 
                   font, 0.5, (100, 255, 100), 1)
        
        # Parametri correnti (colonna sinistra)
        y_offset = 110
        param_texts = [
            f"Deform: {self.config.DEFORMATION_INTENSITY:.1f} (Q/A)",
            f"Glow: {self.config.GLOW_INTENSITY:.2f} (W/S)", 
            f"Lens: {self.config.LENS_SPEED_FACTOR:.3f} (E/D)",
            f"BG Zoom: {self.config.BG_ZOOM_FACTOR:.2f} (R/F)",
        ]
        
        for i, text in enumerate(param_texts):
            cv2.putText(overlay, text, (10, y_offset + i * 20), 
                       font, 0.4, (255, 200, 100), 1)
        
        # Parametri correnti (colonna destra)
        param_texts_2 = [
            f"Logo: {self.config.LOGO_ZOOM_FACTOR:.2f} (T/G)",
            f"Tracer: {self.config.TRACER_MAX_OPACITY:.3f} (Y/H)",
            f"Blend: {self.config.BLENDING_STRENGTH:.2f} (U/J)",
            f"CTRL+R: Reset"
        ]
        
        for i, text in enumerate(param_texts_2):
            cv2.putText(overlay, text, (self.width // 2, y_offset + i * 20), 
                       font, 0.4, (255, 200, 100), 1)
        
        # Controlli
        cv2.putText(overlay, "SPAZIO: Genera Video + Git Push", (10, self.height - 40), 
                   font, 0.5, (255, 100, 255), 1)
        cv2.putText(overlay, "ESC: Esci", (10, self.height - 20), 
                   font, 0.5, (100, 100, 255), 1)
        
        return overlay
    
    def run(self):
        """Avvia la modalità Live Preview"""
        print("🌊 Avviando Live Preview...")
        
        # Inizializza stato rendering
        self._initialize_rendering_state()
        
        # Crea finestra
        cv2.namedWindow("Crystal Therapy - Live Preview", cv2.WINDOW_AUTOSIZE)
        
        self.is_running = True
        self.last_refresh_time = time.time()
        
        print("✅ Live Preview attiva!")
        print("   📺 Guarda la finestra per vedere l'anteprima")
        print("   🔄 Il frame si aggiornerà automaticamente ogni 5 secondi")
        
        try:
            while self.is_running:
                current_time = time.time()
                
                # Controlla se è ora di fare refresh
                if (current_time - self.last_refresh_time >= self.refresh_interval) or self.force_refresh:
                    if not self.force_refresh:
                        print("🔄 Auto-refresh frame...")
                    else:
                        print("🎚️ Refresh forzato per cambio parametri...")
                        self.force_refresh = False
                    
                    # Controlla modifiche ai file
                    if self._check_file_changes():
                        self._reload_resources()
                    
                    self.last_refresh_time = current_time
                
                # Genera frame corrente
                if self.current_frame is None or current_time - self.last_refresh_time < 0.1:
                    self.current_frame = self._generate_preview_frame()
                
                # Aggiungi overlay informativo
                display_frame = self._add_overlay_info(self.current_frame)
                
                # Mostra frame
                cv2.imshow("Crystal Therapy - Live Preview", display_frame)
                
                # Gestisci input utente
                key = cv2.waitKey(33) & 0xFF  # ~30 FPS per UI fluida
                
                if key == 27:  # ESC
                    print("❌ Uscita dalla Live Preview...")
                    break
                elif key == 32:  # SPAZIO
                    print("🎬 Richiesta generazione video completo...")
                    self.should_render_video = True
                    break
                elif key == 18:  # CTRL+R
                    self._reset_params()
                elif key != 255:  # Qualsiasi altro tasto
                    # Gestisci controlli parametri
                    if self._handle_parameter_controls(key):
                        pass  # Il metodo già gestisce il feedback
                
        except KeyboardInterrupt:
            print("\n⚠️ Interrotto dall'utente")
        
        finally:
            # Cleanup
            cv2.destroyAllWindows()
            if self.bg_video:
                self.bg_video.release()
            self.is_running = False
            
        return self.should_render_video
    
    def cleanup(self):
        """Pulizia delle risorse"""
        if self.bg_video:
            self.bg_video.release()
        cv2.destroyAllWindows()


def run_preview_mode(config, render_frame_func, contours, hierarchy, width, height,
                    get_background_func, get_texture_func, initialize_lenses_func, 
                    load_audio_func=None):
    """
    Avvia la modalità Live Preview
    
    Returns:
        bool: True se l'utente ha richiesto di generare il video completo
    """
    preview = LivePreview(
        config, render_frame_func, contours, hierarchy, width, height,
        get_background_func, get_texture_func, initialize_lenses_func, load_audio_func
    )
    
    try:
        return preview.run()
    finally:
        preview.cleanup()
