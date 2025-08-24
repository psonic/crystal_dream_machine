"""
🎵 AUDIO PROCESSOR - Crystal Therapy
Gestione completa dell'audio: caricamento, analisi e reattività

Funzioni:
- Caricamento e analisi file audio per reattività alle frequenze
- Calcolo fattori di modulazione basati su bass/mid/treble
- Aggiunta audio al video finale con ffmpeg
- Gestione selezione casuale e offset temporali
"""

import os
import subprocess
import numpy as np

# Importazioni condizionali per librosa
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("⚠️ Librosa non disponibile. Per supporto audio: pip install librosa")


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
    
    audio_file = audio_data['selected_file']
    start_offset = audio_data.get('start_offset', 0)
    
    print(f"🎵 Aggiungendo audio da {audio_file} (offset: {start_offset:.1f}s)...")
    
    # Comando ffmpeg per aggiungere audio
    cmd = [
        'ffmpeg', '-y',  # -y per sovrascrivere
        '-i', video_path,  # Input video
        '-ss', str(start_offset),  # Inizio audio con offset
        '-i', audio_file,  # Input audio
        '-t', str(duration),  # Durata
        '-c:v', 'copy',  # Copia video stream senza ricodifica
        '-c:a', 'aac',  # Codec audio
        '-map', '0:v:0',  # Usa il video dal primo input
        '-map', '1:a:0',  # Usa l'audio dal secondo input
        '-shortest',  # Termina quando il più corto finisce
        final_video_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✅ Audio aggiunto con successo: {final_video_path}")
        return final_video_path
    except subprocess.CalledProcessError as e:
        print(f"❌ Errore nell'aggiunta dell'audio: {e}")
        print(f"   Output: {e.stderr}")
        return video_path
    except FileNotFoundError:
        print("❌ ffmpeg non trovato. Installare ffmpeg per aggiungere audio.")
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
    
    if not LIBROSA_AVAILABLE:
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
        print(f"🎲 Audio selezionato casualmente: {os.path.basename(selected_audio)}")
    else:
        selected_audio = existing_files[0]
        print(f"🎵 Audio: {os.path.basename(selected_audio)}")
    
    try:
        # Carica il file audio
        y, sr = librosa.load(selected_audio, sr=None)
        audio_duration = len(y) / sr
        
        print(f"📊 Audio caricato: {audio_duration:.1f}s @ {sr}Hz")
        
        # Calcola l'offset di inizio se richiesto
        start_offset = 0
        if random_start and audio_duration > duration:
            # Inizia da un punto casuale ma lascia abbastanza spazio per la durata del video
            max_start = min(audio_duration - duration, audio_duration * 0.66)  # Max 2/3 del file
            if max_start > 0:
                start_offset = np.random.uniform(0, max_start)
                print(f"🎲 Inizio casuale da {start_offset:.1f}s")
        
        # Estrai il segmento audio per la durata del video
        start_sample = int(start_offset * sr)
        end_sample = int((start_offset + duration) * sr)
        
        if end_sample > len(y):
            end_sample = len(y)
            actual_duration = (end_sample - start_sample) / sr
            print(f"⚠️ Audio più corto del video, usando {actual_duration:.1f}s")
        
        audio_segment = y[start_sample:end_sample]
        
        # Calcola STFT per analisi delle frequenze
        hop_length = int(sr / fps)  # Samples per frame
        n_fft = 2048
        
        stft = librosa.stft(audio_segment, hop_length=hop_length, n_fft=n_fft)
        magnitude = np.abs(stft)
        
        # Analisi delle bande di frequenza
        # Bass: 0-250 Hz
        # Mid: 250-4000 Hz  
        # High: 4000+ Hz
        
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        bass_idx = np.where(freqs <= 250)[0]
        mid_idx = np.where((freqs > 250) & (freqs <= 4000))[0]
        high_idx = np.where(freqs > 4000)[0]
        
        # Estrai energie per banda
        bass_energy = np.mean(magnitude[bass_idx, :], axis=0)
        mid_energy = np.mean(magnitude[mid_idx, :], axis=0)
        high_energy = np.mean(magnitude[high_idx, :], axis=0)
        total_energy = np.mean(magnitude, axis=0)
        
        # Normalizza e applica smoothing
        def normalize_and_smooth(data, window=3):
            # Normalizza 0-1
            data_norm = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)
            # Applica smoothing con rolling average
            if len(data_norm) > window:
                kernel = np.ones(window) / window
                data_smooth = np.convolve(data_norm, kernel, mode='same')
                return data_smooth
            return data_norm
        
        bass_smooth = normalize_and_smooth(bass_energy)
        mid_smooth = normalize_and_smooth(mid_energy)
        high_smooth = normalize_and_smooth(high_energy)
        total_smooth = normalize_and_smooth(total_energy)
        
        print(f"🎚️ Analisi completata: {len(bass_smooth)} frame audio")
        
        return {
            'bass': bass_smooth,
            'mid': mid_smooth,
            'high': high_smooth,
            'total': total_smooth,
            'selected_file': selected_audio,
            'start_offset': start_offset,
            'duration': duration,
            'sample_rate': sr
        }
        
    except Exception as e:
        print(f"❌ Errore nel caricamento audio {selected_audio}: {e}")
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
    
    return factors


def get_organic_deformation_factors(audio_data, frame_idx, config):
    """
    🎵 Calcola i parametri dinamici per la deformazione organica basati sull'audio con effetto rimbalzo.
    
    Args:
        audio_data: Dati audio preprocessati (può essere None)
        frame_idx: Indice del frame corrente
        config: Configurazione con parametri di deformazione
    
    Returns:
        dict: Parametri dinamici per la deformazione organica (o None se audio disabilitato)
    """
    
    if not audio_data or not config.AUDIO_ENABLED or not config.DEFORMATION_AUDIO_REACTIVE:
        return None
    
    # Sicurezza per indici fuori range
    audio_frame_idx = min(frame_idx, len(audio_data['bass']) - 1)
    if audio_frame_idx < 0:
        audio_frame_idx = 0
    
    # Estrai valori dalle diverse bande
    bass = audio_data['bass'][audio_frame_idx]
    mid = audio_data['mid'][audio_frame_idx]
    high = audio_data['high'][audio_frame_idx]
    total = audio_data['total'][audio_frame_idx]
    
    # Applica multiplier per intensificare l'effetto
    multiplier = config.DEFORMATION_AUDIO_MULTIPLIER
    
    # Calcola parametri dinamici con effetto rimbalzo
    # Bass -> ampiezza principale (movimento ampio)
    # Mid -> frequenza di oscillazione (movimento dettagliato)
    # High -> jitter e rumore (movimento fine)
    
    dynamic_params = {
        'amplitude_boost': 1.0 + (bass * multiplier * 0.8),  # Bass influenza l'ampiezza
        'frequency_mod': 1.0 + (mid * multiplier * 0.6),     # Mid influenza la frequenza
        'noise_intensity': high * multiplier * 0.4,           # High aggiunge rumore
        'elastic_bounce': bass * 0.3 + mid * 0.2,            # Effetto rimbalzo elastico
        'organic_flow': total * multiplier * 0.5              # Flusso organico generale
    }
    
    return dynamic_params


def load_audio_wrapper(audio_files, duration_seconds, fps, random_selection, random_start):
    """
    🎵 Wrapper per il caricamento audio con gestione degli errori.
    
    Args:
        audio_files: Lista file audio o stringa singola
        duration_seconds: Durata in secondi
        fps: Frame rate
        random_selection: Selezione casuale del file
        random_start: Inizio casuale nel file
    
    Returns:
        dict: Dati audio o None se errore
    """
    try:
        return load_audio_analysis(audio_files, duration_seconds, fps, random_selection, random_start)
    except Exception as e:
        print(f"⚠️ Errore nel caricamento audio: {e}")
        print("⚠️ Errore nel caricamento audio: rendering senza sincronizzazione")
        return None
