"""
🔭 Componente Lenti per CrystalPython3
Gestisce il sistema di lenti cinematografiche per effetti ottici e distorsioni geometriche.
"""

import numpy as np
import cv2


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
        
        # Posizione iniziale casuale lungo il percorso
        initial_path_position = np.random.randint(0, len(path))
        
        # Parametri casuali per ogni lente (variazione controllata)
        # Distribuzione non uniforme: molte piccole, poche grandi
        # Usa distribuzione power law per favorire dimensioni piccole
        random_factor = np.random.random() ** 2.5  # Esponente 2.5 = forte bias verso piccole
        base_radius = config.LENS_MIN_RADIUS + (config.LENS_MAX_RADIUS - config.LENS_MIN_RADIUS) * random_factor
        base_strength = np.random.uniform(config.LENS_MIN_STRENGTH, config.LENS_MAX_STRENGTH)
        
        lens = {
            'pos': np.array(path[initial_path_position], dtype=np.float32),  # Posizione iniziale dal percorso
            'velocity': np.array([0.0, 0.0], dtype=np.float32),  # Velocità iniziale nulla
            'base_radius': base_radius,  # Raggio base per la pulsazione
            'radius': base_radius,  # Raggio corrente
            'base_strength': base_strength,  # Forza base per la pulsazione
            'strength': base_strength,  # Forza corrente
            'angle': np.random.uniform(0, 2 * np.pi),  # Angolo per la forma a verme
            'rotation_speed': np.random.uniform(-0.02, 0.02),  # Velocità di rotazione della forma
            'path': path,  # Il percorso cinematografico completo
            'path_offset': initial_path_position,  # Offset iniziale nel percorso per sfasamento
            'pulsation_offset': np.random.uniform(0, 2 * np.pi)  # Offset di fase per la pulsazione
        }
        
        lenses.append(lens)
    
    print(f"🔭 Inizializzate {len(lenses)} lenti cinematografiche")
    print(f"   📊 Distribuzione percorsi: {horizontal_lens_count} orizzontali, {mixed_lens_count} misti")
    
    return lenses


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
