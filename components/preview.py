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
        
        # File di configurazione live
        self.live_params_file = "config"
        self.last_params_mtime = 0
        
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
        
        # Carica parametri iniziali dal file
        if os.path.exists(self.live_params_file):
            self.last_params_mtime = os.path.getmtime(self.live_params_file)
            self._load_live_params()
        
        print("🌊 Live Preview inizializzata!")
        print("   📺 Finestra: Crystal Therapy - Live Preview")
        print("   🔄 Auto-refresh: ogni 1 secondo")
        print("   📝 MODIFICA PARAMETRI: Edita il file 'config' e salvalo!")
        print("   🎬 SPAZIO: genera video completo + Git push")
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
    
    def _load_live_params(self):
        """Carica i parametri dal file config"""
        if not os.path.exists(self.live_params_file):
            return False
            
        try:
            params_changed = False
            with open(self.live_params_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # Conversione dei valori
                        if key == 'DEFORMATION_INTENSITY':
                            new_val = float(value)
                            if new_val != self.config.DEFORMATION_INTENSITY:
                                self.config.DEFORMATION_INTENSITY = new_val
                                params_changed = True
                                
                        elif key == 'DEFORMATION_SPEED':
                            new_val = float(value)
                            if new_val != self.config.DEFORMATION_SPEED:
                                self.config.DEFORMATION_SPEED = new_val
                                params_changed = True
                                
                        elif key == 'DEFORMATION_SCALE':
                            new_val = float(value)
                            if new_val != self.config.DEFORMATION_SCALE:
                                self.config.DEFORMATION_SCALE = new_val
                                params_changed = True
                                
                        elif key == 'GLOW_INTENSITY':
                            new_val = float(value)
                            if new_val != self.config.GLOW_INTENSITY:
                                self.config.GLOW_INTENSITY = new_val
                                params_changed = True
                                
                        elif key == 'GLOW_KERNEL_SIZE':
                            new_val = int(value)
                            if new_val != self.config.GLOW_KERNEL_SIZE:
                                self.config.GLOW_KERNEL_SIZE = new_val
                                params_changed = True
                                
                        elif key == 'LENS_SPEED_FACTOR':
                            new_val = float(value)
                            if new_val != self.config.LENS_SPEED_FACTOR:
                                self.config.LENS_SPEED_FACTOR = new_val
                                params_changed = True
                                
                        elif key == 'NUM_LENSES':
                            new_val = int(value)
                            if new_val != self.config.NUM_LENSES:
                                self.config.NUM_LENSES = new_val
                                params_changed = True
                                print("⚠️ NUM_LENSES cambiato - Ricaricamento lenti necessario")
                                
                        elif key == 'LOGO_ZOOM_FACTOR':
                            new_val = float(value)
                            if new_val != self.config.LOGO_ZOOM_FACTOR:
                                self.config.LOGO_ZOOM_FACTOR = new_val
                                params_changed = True
                                
                        elif key == 'BG_ZOOM_FACTOR':
                            new_val = float(value)
                            if new_val != self.config.BG_ZOOM_FACTOR:
                                self.config.BG_ZOOM_FACTOR = new_val
                                params_changed = True
                                
                        elif key == 'TRACER_MAX_OPACITY':
                            new_val = float(value)
                            if new_val != self.config.TRACER_MAX_OPACITY:
                                self.config.TRACER_MAX_OPACITY = new_val
                                params_changed = True
                                
                        elif key == 'BG_TRACER_MAX_OPACITY':
                            new_val = float(value)
                            if new_val != self.config.BG_TRACER_MAX_OPACITY:
                                self.config.BG_TRACER_MAX_OPACITY = new_val
                                params_changed = True
                                
                        elif key == 'BLENDING_STRENGTH':
                            new_val = float(value)
                            if new_val != self.config.BLENDING_STRENGTH:
                                self.config.BLENDING_STRENGTH = new_val
                                params_changed = True
                                
                        elif key == 'BLEND_TRANSPARENCY':
                            new_val = float(value)
                            if new_val != self.config.BLEND_TRANSPARENCY:
                                self.config.BLEND_TRANSPARENCY = new_val
                                params_changed = True
                                
                        elif key in ['LOGO_COLOR_B', 'LOGO_COLOR_G', 'LOGO_COLOR_R']:
                            new_val = int(value)
                            current_color = list(self.config.LOGO_COLOR)
                            if key == 'LOGO_COLOR_B':
                                current_color[0] = new_val
                            elif key == 'LOGO_COLOR_G':
                                current_color[1] = new_val
                            elif key == 'LOGO_COLOR_R':
                                current_color[2] = new_val
                            
                            new_color = tuple(current_color)
                            if new_color != self.config.LOGO_COLOR:
                                self.config.LOGO_COLOR = new_color
                                params_changed = True
                                
                        elif key == 'TEXTURE_ALPHA':
                            new_val = float(value)
                            if new_val != self.config.TEXTURE_ALPHA:
                                self.config.TEXTURE_ALPHA = new_val
                                params_changed = True
                                
                        elif key == 'TEXTURE_BACKGROUND_ALPHA':
                            new_val = float(value)
                            if new_val != self.config.TEXTURE_BACKGROUND_ALPHA:
                                self.config.TEXTURE_BACKGROUND_ALPHA = new_val
                                params_changed = True
            
            if params_changed:
                print("📝 Parametri aggiornati dal file config")
                
            return params_changed
            
        except Exception as e:
            print(f"⚠️ Errore nel caricamento parametri live: {e}")
            return False
    
    def _check_params_file_changes(self):
        """Controlla se il file dei parametri è stato modificato"""
        if not os.path.exists(self.live_params_file):
            return False
            
        mtime = os.path.getmtime(self.live_params_file)
        if mtime != self.last_params_mtime:
            self.last_params_mtime = mtime
            return self._load_live_params()
        
        return False
        
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
            f"Deform Int: {self.config.DEFORMATION_INTENSITY:.1f}",
            f"Glow: {self.config.GLOW_INTENSITY:.2f}", 
            f"Lens Speed: {self.config.LENS_SPEED_FACTOR:.3f}",
            f"BG Zoom: {self.config.BG_ZOOM_FACTOR:.2f}",
            f"Logo Zoom: {self.config.LOGO_ZOOM_FACTOR:.2f}",
        ]
        
        for i, text in enumerate(param_texts):
            cv2.putText(overlay, text, (10, y_offset + i * 20), 
                       font, 0.4, (255, 200, 100), 1)
        
        # Parametri correnti (colonna destra)
        param_texts_2 = [
            f"Tracer: {self.config.TRACER_MAX_OPACITY:.3f}",
            f"BG Tracer: {self.config.BG_TRACER_MAX_OPACITY:.3f}",
            f"Blend: {self.config.BLENDING_STRENGTH:.2f}",
            f"Texture: {self.config.TEXTURE_ALPHA:.2f}",
            f"Color: {self.config.LOGO_COLOR}",
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
        print("   🔄 Il frame si aggiornerà automaticamente ogni secondo")
        
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
                    
                    # Controlla modifiche al file parametri
                    if self._check_params_file_changes():
                        self.force_refresh = True
                    
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
