# 🌊 Crystal

Generatore di video: prende un logo (SVG o PDF), lo deforma (onde di Perlin, stiramenti,
"lenti" a forma di verme), gli lascia dietro delle scie e lo fonde con un video di
sfondo, opzionalmente reagendo all'audio. È la v4 del *Natisone Trip Generator*:
stesso look, render da ore a minuti, struttura pensata per aggiungere effetti.

---

## 1. Setup (una volta sola)

```bash
brew install ffmpeg cairo                 # se non ci sono già
python3 -m venv .venv                     # il venv c'è già (Python 3.13); quello vecchio
                                          # della v3 (Python 3.9) è in .venv-v3
.venv/bin/python -m pip install -r requirements.txt
```

Tutti i comandi si lanciano **dalla cartella del progetto**:

```bash
source .venv/bin/activate                 # così basta scrivere "python"
```

## 2. Lo studio: la preview live

```bash
python -m crystal studio crystal_therapy  # si apre il browser su http://127.0.0.1:8765
```

È il posto dove si lavora. Il video gira **in tempo reale** (a metà risoluzione,
30 fps) e ogni parametro di `defaults.toml` è uno slider: lo muovi e vedi subito
l'effetto, mentre il video continua (o, da fermo, sul frame che stai guardando).

- **Quello che vedi è quello che esce.** La preview usa lo stesso motore del render
  e lo stesso seed. «Render video» fa il video finale identico, a piena risoluzione.
- **Timeline** con bassi/medi/alti dell'audio sotto: clicca o trascina per andare in un
  punto qualsiasi. Le scie si "scaldano" da sole (meno di un secondo).
- **Audio** sincronizzato (🔊), così senti cosa fa reagire le lenti.
- **📌 Fissa A** + **⇆ Confronta**: congeli un frame e lo confronti a tendina con quello
  live mentre cambi i parametri.
- **Salva** scrive nel preset solo le chiavi diverse dai default, **mantenendo i commenti**
  del file. A sinistra di ogni parametro: viola = scritto nel preset, giallo = cambiato e
  da salvare. ↺ (o doppio clic sul nome) lo toglie dal preset. ⌘Z / ⇧⌘Z annulla e ripete.
- **seed**: 🎲 ne prova uno nuovo (sfondo, punto di partenza, lenti, audio cambiano);
  📌 lo scrive nel preset, così il video è sempre quello.
- Puoi anche modificare il `.toml` nell'editor: quando salvi, lo studio lo ricarica.
- **Frame PNG**, **Render test** e **Render video** partono dalla pagina, con la barra di
  avanzamento e il pulsante per guardare il risultato.
- In alto a destra: risoluzione (25–100%) e fps della preview. `?` mostra le scorciatoie
  (spazio, ←/→, ⇧←/→, P, C, M, /, ⌘S).

`python -m crystal preview` è un sinonimo di `studio`.

### Il pannello di controllo (senza terminale)

Doppio clic su **Crystal** (in `~/Applications`, o Spotlight → "Crystal") apre nel browser
un pannello che mostra cosa sta girando e quanta CPU usa, e permette di:

- avviare uno **studio** (preset, *potenza* = numero di processi, auto-spegnimento) o un **render**;
- **fermare** uno studio/render, con i suoi processi;
- **pulire** eventuali processi rimasti appesi; **⏻ Spegni tutto**.

Dal terminale: `python -m crystal panel`, e `python -m crystal stop` spegne tutto.
L'app si (ri)crea con `tools/make_app.sh` (va rifatta se sposti la cartella del progetto).

Per non scaldare il Mac per niente:
- lo studio **mette in pausa** la preview quando nessuna pagina è aperta e **si chiude da solo**
  dopo 15 minuti senza pagine (`--idle-exit MINUTI`, 0 = mai); mai durante un render;
- `--workers N` limita i processi delle deformazioni (default: core − 2, max 8);
- se lo studio o un render muoiono in qualunque modo (kill, crash, terminale chiuso)
  i loro processi figli si chiudono da soli entro un secondo.

## 3. Fare un video dal terminale

```bash
python -m crystal render                  # video con i default (il look Natisone)
python -m crystal render delirio          # con un preset di presets/
python -m crystal render --test           # prova veloce: metà risoluzione, 15 fps, 6 s
```

Il video finisce in `output/AAAAMMGG_HHMMSS_<preset>.mp4` (i test in `output/test/`).
Accanto c'è un `.json` con **tutta** la configurazione usata, seed compreso.
Per rifare lo stesso identico video:

```bash
python -m crystal render output/<data>_<preset>.json
```

Altri comandi:

- **`python -m crystal frame <preset> --at 12`**: un PNG a risoluzione piena del secondo 12.
- **`python -m crystal render <preset> --test`**: un video breve a risoluzione ridotta.

### Cambiare un parametro al volo

```bash
python -m crystal render --set lenses.count=100 --set output.format=ig_story
python -m crystal render delirio --seed 42          # stesso seed = stesse scelte casuali
python -m crystal render --set 'background.video="input/ghiaccio.MOV"'
```

Le stringhe dentro `--set` vanno tra virgolette doppie, e il tutto tra apici.

## 4. Preset e parametri

La configurazione si applica a strati:

```
crystal/defaults.toml   →   presets/<nome>.toml   →   --set chiave=valore
   (tutto, commentato)        (solo cosa cambia)        (al volo)
```

- **`crystal/defaults.toml`** contiene *tutti* i parametri, ognuno con un commento. È
  il riferimento: aprilo per sapere cosa si può toccare. Non modificarlo per un singolo
  video.
- **`presets/*.toml`** sono i tuoi look. Ci scrivi solo le chiavi che vuoi diverse.
  Per farne uno nuovo copia `presets/natisone.toml`. `python -m crystal presets` li
  elenca.
- Una chiave scritta male (es. `lenses.cont`) dà errore subito, invece di essere
  ignorata in silenzio.

Preset inclusi:

| preset | cosa fa |
|---|---|
| `natisone` | il look dei video di agosto 2025 (= i default) |
| `delirio` | story 9:16, lenti veloci e forti, scie colorate, blend psichedelico, sfondo a caso fra fiumi/ghiaccio/fumo |
| `leggero` | pochi effetti, logo fedele all'SVG, render rapidissimo |
| `crystal_therapy` | l'ultimo config della v3 (27/08/2025): la scritta "Crystal Therapy" dal PDF, 1576×1160, 100 lenti |

Convenzioni:
- **Pixel** = riferiti alla risoluzione piena. In `--test` vengono scalati, quindi il
  test somiglia al finale.
- **Velocità** = "per frame a 30 fps". Cambiando fps il movimento resta uguale.
- **Colori** = `[R, G, B]` da 0 a 255.
- **File** (`background.video`, `audio.files`): un percorso, un glob
  (`"input/fiume*.MOV"`) o una lista. Se sono più di uno ne viene scelto uno a caso
  (in base al seed).

## 5. Gli effetti, in ordine

| sezione | cosa fa | i parametri che contano |
|---|---|---|
| `[output]` | formato e qualità | `format` (`ig_post` 1080×1350, `ig_story` 1080×1920, `square`, `landscape`, `logo` = dimensioni del file logo, `custom`), `duration`, `fps`, `crf`, `encoder` (`x264` o `videotoolbox`, l'encoder hardware del Mac: più veloce, file più grandi) |
| `[logo]` | il logo (SVG o PDF) | `file`, `color`, `zoom`, `left_padding`, `fill` |
| `[background]` | video di sfondo | `video`, `zoom`, `slowdown`, `darken`, `random_start` / `start` |
| `[organic]` | ondulazioni morbide del logo (Perlin) | `intensity` (px), `scale` (frequenza), `speed` |
| `[stretch]` | stiramenti lunghi | `intensity`, `horizontal`, `vertical`, `fine_detail` |
| `[lenses]` | "creature" che deformano il logo passandoci sopra | `count`, `min/max_strength`, `min/max_radius`, `speed`, `path_speed`, `worm_*`, `pulsation_*` |
| `[tracers.logo]` | scie dei bordi del logo | `trail` (frame), `opacity`, `color`, `hue_shift` |
| `[tracers.background]` | scie dei contorni dello sfondo | come sopra + `threshold1/2` (più bassi = più contorni) |
| `[texture]` | un'immagine fusa su logo/sfondo | `path`, `target`, `alpha`, `mode` |
| `[glow]` | bagliore attorno al logo | `size`, `intensity` |
| `[blend]` | come il logo si fonde con lo sfondo | `preset`: `soft` `cinematic` `artistic` `dramatic` `bright` `intense` `psychedelic` `glow` `dark` `geometric` `manual` |
| `[audio]` | reattività all'audio e traccia nel video | `files`, `bass/mid/high` (sensibilità), `smoothing`, `mux` |
| `[variation]` | oscillazioni lentissime automatiche | `enabled`, `amplitude` |

**`logo.fill`**:
- `blob` (default) è il look dei video v3 fatti con `logo.svg`. I path vengono riempiti
  come poligoni e le zone chiuse diventano piene, quindi esce una sagoma "creatura".
  Con un PDF tiene solo la sagoma esterna più grande: per la scritta usa `none`.
- `none` disegna il logo esattamente com'è nel file (così faceva la v3 con il PDF).
- `holes` è come `none`, ma con i buchi interni riempiti.

**Blend mode** (per `blend.mode` con `preset = "manual"` e per `texture.mode`):
`normal multiply screen overlay soft_light hard_light color_dodge color_burn darken
lighten difference exclusion`.

**Audio**:
- I bassi accelerano le lenti, i medi ne aumentano la forza, gli alti la pulsazione.
- Con `organic.audio_reactive = true` i bassi muovono anche le ondulazioni.
- Con i default la reattività è molto delicata. Per sentirla davvero alza
  `audio.bass/mid/high` (es. 0.5) e abbassa `audio.smoothing` (es. 0.6).

## 6. File di input

Tutto in `input/`:
- il logo, SVG o PDF (`input/logo.svg`, `input/logo.pdf`);
- i video `.MOV` di sfondo;
- gli audio `.aif`;
- le texture.

Video e audio sono esclusi da git perché pesano troppo.

## 7. Prestazioni

Sul M4, il render completo di `crystal_therapy` (1576×1160, 30 fps, 40 s = 1200 frame,
100 lenti, tutto attivo) richiede **circa 36–40 secondi**: il tempo reale
(30–33 fps; con `output.encoder = "videotoolbox"` 41 fps). La prima v4 ci metteva 94 s, la v3
andava a ~1 fps già a *432×540*: a piena risoluzione erano ore.

| v3 | v4 |
|---|---|
| rumore di Perlin calcolato pixel per pixel in un loop Python (~1,5 M chiamate per frame) | rumore vettorizzato, su una griglia rada il cui passo segue la frequenza del rumore |
| tre deformazioni = tre `remap` del logo a piena risoluzione | le tre mappe si compongono (su una griglia a 2 px) e il logo si interpola **una volta sola** |
| 60 lenti, ognuna calcolata su tutto il frame | ogni lente lavora solo nel suo riquadro |
| maschera del logo ricostruita da zero a ogni frame | calcolata una volta sola |
| scie: somma di 85 + 105 frame passati a ogni frame | un buffer che sfuma, aggiornato con OpenCV |
| fusione logo/sfondo su tutto il frame | solo sui pixel del logo; armonizzazione e luminanza ridotte a un colore |
| `seek` nel video di sfondo a ogni frame, encoder OpenCV `mp4v` | ffmpeg (con decodifica hardware) legge lo sfondo già scalato e codifica in H.264 in parallelo |
| un solo processo | deformazioni su più processi, composizione di più frame in parallelo su più thread |

A fine render viene stampato quanto pesa ogni fase per frame, così si vede subito
cosa rallenta se si esagera.

**Preview e video finale coincidono anche a risoluzione ridotta**: le misure in pixel si
scalano, e i bordi che generano le scie si cercano sempre alla scala del video finale
(altrimenti a metà risoluzione le scie verrebbero 2-3 volte più fitte).

## 8. Struttura del codice

```
crystal/
  cli.py            comandi: render, frame, studio (= preview), presets
  config.py         caricamento a strati + preset di blending
  defaults.toml     TUTTI i parametri, commentati
  render.py         Renderer: prepara tutto e produce i frame (pipeline in cima al file)
  timeline.py       stato che cambia nel tempo, precalcolato (audio, lenti, fasi)
  media.py          ffmpeg (sfondo, encoding, audio), SVG/PDF → maschera
  effects/
    noise.py        Perlin vettorizzato
    deform.py       organic, stretch, lenti (deformano la maschera del logo)
    compose.py      scie, texture, blend mode, fusione logo/sfondo
  studio/           la preview web
    engine.py       motore della preview: riproduzione, salti, modifiche al volo
    server.py       server HTTP (solo libreria standard), render lanciati dalla pagina
    schema.py       parametri per l'interfaccia, letti da defaults.toml (+ range degli slider)
    presetfile.py   salvataggio dei preset mantenendo i commenti
    static/         la pagina (HTML, CSS, JS senza dipendenze)
presets/            i tuoi look
legacy/v3/          il codice della v3 (Natisone Trip Generator) com'era, con il tuo ultimo config
```

Idea chiave: **la Timeline precalcola tutto ciò che evolve nel tempo**, quindi
deformare il frame *k* è una funzione pura e si può fare in parallelo. Le scie invece
dipendono dai frame precedenti: si aggiornano in ordine (`Renderer.advance`), ma ogni
frame dipende solo dallo stato delle scie prima di lui, quindi la composizione
(`Renderer.compose`) di frame diversi gira in parallelo. Render, `frame` e studio usano
le stesse primitive (`Renderer.step`).

### Aggiungere un effetto

- **Una deformazione del logo**:
  1. in `effects/deform.py` scrivi un metodo che restituisce lo spostamento (dx, dy),
     componilo in `Deformer.maps()` e aggiungi il nome a `Deformer.STAGES`;
  2. aggiungi la sezione `[nome]` (con `enabled`) in `defaults.toml`: nello studio
     compare da sola; per un range dello slider migliore aggiungi la chiave a
     `HINTS` in `studio/schema.py`;
  3. se ti serve qualcosa che varia nel tempo, calcolalo in `Timeline` e leggilo con
     l'indice `k`.
- **Un effetto sul frame** (colore, distorsione dello sfondo, overlay...): aggiungilo
  in `Renderer.frames()` / `_composite()` in `render.py`, con la sua sezione in
  `defaults.toml`.

## 9. Differenze rispetto alla v3

- **Niente più commit/tag/push git automatici** prima di ogni render. Il `.json`
  accanto al video fa la stessa cosa senza toccare il repo.
- **Parametri rimossi perché non facevano niente**:
  - `AUDIO_BOOST_FACTOR`, `LOGO_ALPHA`, `LOGO_BLEND_FACTOR`;
  - `WORM_COMPLEXITY`, `LENS_HORIZONTAL_BIAS`;
  - `AUDIO_*_INFLUENCE`, `ORGANIC_AUDIO_MULTIPLIER`, `TRACER_THRESHOLD1/2`;
  - tutta la parte `STRETCH_ANTIALIASING` / `SHADER_*` (codice mai eseguito);
  - `SMOOTHING_*`, `WHATSAPP_*`: l'output H.264 yuv420p va già bene per WhatsApp e
    Instagram.
- **Parametri che prima venivano ignorati e ora funzionano**:
  - `audio.smoothing`;
  - `lenses.rotation_min/max`;
  - `stretch.audio_reactive`.
- **Velocità legate al tempo, non al numero di frame**: test e finale si muovono
  uguale.
- **Antialiasing più pulito**:
  - mappe float32 e maschera con bordi morbidi;
  - corretto il bug che in v3 faceva apparire pixel colorati sui bordi;
  - il logo è centrato correttamente (in v3 era spostato di ~25 px in alto a sinistra).
- **`--test` scala tutto**, anche lenti, glow e deformazioni. In v3 il test a 432 px
  "incollava" il logo molto più del render finale.
- **La preview live è nel browser** (`python -m crystal studio`): tempo reale, slider per
  ogni parametro, salvataggio nel preset. Prima era una finestra OpenCV con un file di
  parametri a parte.
- **La v3 completa è in `legacy/v3/`**, compreso il `config` con i ritocchi non
  committati (tradotti nel preset `crystal_therapy`). Per farla girare di nuovo:
  `cd legacy/v3 && ln -s ../../input input && ../../.venv-v3/bin/python natisone_trip_generator.py --test`.
  Attenzione: senza `--test` fa commit, tag e push su GitHub da sola.
