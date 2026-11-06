"""
🎵 Componente Audio per CrystalPython3
Gestisce il caricamento, l'analisi e la reattività audio per i video generati.
"""

import numpy as np
import os
import subprocess

# Import condizionale per librosa con verifica step-by-step
AUDIO_AVAILABLE = False
librosa = None

try:
    # Prima verifica numpy (dovrebbe essere già disponibile)
    import numpy as np_test
    
    # Poi prova matplotlib
    try:
        import matplotlib
        print("🎨 Matplotlib disponibile")
    except ImportError:
        print("⚠️ Matplotlib non trovato - tentativo installazione automatica...")
        import subprocess
        import sys
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'matplotlib'])
            import matplotlib
            print("✅ Matplotlib installato con successo")
        except:
            print("❌ Impossibile installare matplotlib automaticamente")
            raise ImportError("matplotlib non disponibile")
    
    # Infine prova librosa
    import librosa
    import librosa.display
    AUDIO_AVAILABLE = True
    print("🎵 Librosa disponibile - Supporto audio attivato!")
    
except ImportError as e:
    AUDIO_AVAILABLE = False
    print(f"⚠️ Audio non disponibile: {e}")
    print("   Per supporto audio completo: pip install librosa matplotlib")
except Exception as e:
    AUDIO_AVAILABLE = False
    print(f"⚠️ Errore nell'inizializzazione audio: {e}")
    print("   Per supporto audio completo: pip install librosa matplotlib")


class AudioSmoothingState:
    """Memorizza lo stato per il smoothing dell'audio reattivo con effetto rimbalzo."""
    def __init__(self):
        self.prev_intensity = None
        self.prev_speed = None
        self.prev_scale = None


# Istanza globale per il smoothing
_audio_smoothing_state = AudioSmoothingState()


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
    if not AUDIO_AVAILABLE:
        print("⚠️ Librosa non disponibile, audio disabilitato")
        return None
    
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
    Supporta sia deformazione organica che stretch con parametri specifici.
    
    Args:
        audio_data: Dati audio preprocessati
        frame_idx: Indice del frame corrente
        config: Configurazione con parametri audio
    
    Returns:
        dict: Parametri dinamici per la deformazione organica (o None se audio disabilitato)
    """
    global _audio_smoothing_state
    
    # Controlla se l'audio è abilitato e se almeno una deformazione è audio-reattiva
    audio_reactive = (
        config.AUDIO_ENABLED and (
            (hasattr(config, 'ORGANIC_AUDIO_REACTIVE') and config.ORGANIC_AUDIO_REACTIVE) or
            (hasattr(config, 'STRETCH_AUDIO_REACTIVE') and config.STRETCH_AUDIO_REACTIVE)
        )
    )
    
    if not audio_data or not audio_reactive:
        return None
    
    # Assicurati che l'indice del frame sia valido
    audio_frame_idx = min(frame_idx, len(audio_data['bass']) - 1)
    
    if audio_frame_idx < 0:
        audio_frame_idx = 0
    
    # Estrai i valori per il frame corrente
    bass = audio_data['bass'][audio_frame_idx]
    mid = audio_data['mid'][audio_frame_idx]
    high = audio_data['high'][audio_frame_idx]
    
    # Determina quale tipo di deformazione è attiva per i parametri base
    base_intensity = 25.0
    base_speed = 0.015
    base_scale = 0.0008
    bass_intensity_factor = 0.22
    bass_speed_factor = 0.03
    mid_scale_factor = 0.002
    smoothing = 0.95
    
    # Usa parametri organic se abilitato
    if hasattr(config, 'ORGANIC_DEFORMATION_ENABLED') and config.ORGANIC_DEFORMATION_ENABLED:
        base_intensity = config.ORGANIC_INTENSITY
        base_speed = config.ORGANIC_SPEED
        base_scale = config.ORGANIC_SCALE
        bass_intensity_factor = config.ORGANIC_BASS_INTENSITY
        bass_speed_factor = config.ORGANIC_BASS_SPEED
        mid_scale_factor = config.ORGANIC_MID_SCALE
        smoothing = config.ORGANIC_SMOOTHING
    # Altrimenti usa parametri stretch se abilitato
    elif hasattr(config, 'STRETCH_DEFORMATION_ENABLED') and config.STRETCH_DEFORMATION_ENABLED:
        base_intensity = config.STRETCH_INTENSITY
        base_speed = config.STRETCH_SPEED
        base_scale = config.STRETCH_SCALE
        bass_intensity_factor = config.STRETCH_BASS_INTENSITY
        bass_speed_factor = config.STRETCH_BASS_SPEED
        mid_scale_factor = config.STRETCH_MID_SCALE
        smoothing = config.STRETCH_SMOOTHING
    
    # Calcola i parametri dinamici raw (in modo delicato)
    raw_intensity = base_intensity + (bass * bass_intensity_factor)
    raw_speed = base_speed + (bass * bass_speed_factor)
    raw_scale = base_scale + (mid * mid_scale_factor)
    
    # Applica smoothing con effetto rimbalzo per movimento più fluido
    
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
                                                    base_intensity * 0.6, 
                                                    base_intensity * 1.4)
    dynamic_params['deformation_speed'] = np.clip(dynamic_params['deformation_speed'], 
                                                base_speed * 0.7, 
                                                base_speed * 1.5)
    dynamic_params['deformation_scale'] = np.clip(dynamic_params['deformation_scale'], 
                                                base_scale * 0.8, 
                                                base_scale * 1.3)
    
    return dynamic_params


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


def load_audio_wrapper(audio_files, duration_seconds, fps, random_selection, random_start):
    """
    🎵 Wrapper per il caricamento audio che gestisce la configurazione.
    
    Args:
        audio_files: Lista dei file audio
        duration_seconds: Durata in secondi
        fps: Frame rate
        random_selection: Selezione casuale
        random_start: Inizio casuale
    
    Returns:
        dict: Dati audio o None
    """
    if not AUDIO_AVAILABLE:
        return None
    
    return load_audio_analysis(
        audio_files, 
        duration_seconds, 
        fps, 
        random_selection, 
        random_start
    )
