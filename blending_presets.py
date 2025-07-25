"""
🎨 PRESET DI BLENDING PER CRYSTAL THERAPY
=====================================

Copia uno di questi preset nella classe Config del file crystal_fiume_funziona_davvero.py
per ottenere diversi effetti di blending tra logo e sfondo.

Come usare:
1. Scegli un preset qui sotto
2. Copia i parametri nella sezione "NUOVI PARAMETRI BLENDING CONFIGURABILI" della classe Config
3. Esegui lo script e goditi l'effetto!

"""

# 🎬 PRESET CINEMATOGRAFICO
# Perfetto per video professionali con integrazione naturale
CINEMATIC_PRESET = {
    'BLENDING_MODE': 'overlay',
    'BLENDING_STRENGTH': 0.8,
    'EDGE_DETECTION_ENABLED': True,
    'EDGE_BLUR_RADIUS': 15,
    'ADAPTIVE_BLENDING': True,
    'COLOR_HARMONIZATION': True,
    'LUMINANCE_MATCHING': False,
    'BLEND_TRANSPARENCY': 0.2,
    'COLOR_BLENDING_STRENGTH': 0.4
}

# 🌟 PRESET ARTISTICO
# Effetto creativo e surreale con differenze cromatiche
ARTISTIC_PRESET = {
    'BLENDING_MODE': 'difference',
    'BLENDING_STRENGTH': 0.9,
    'EDGE_DETECTION_ENABLED': True,
    'EDGE_BLUR_RADIUS': 21,
    'ADAPTIVE_BLENDING': False,
    'COLOR_HARMONIZATION': False,
    'LUMINANCE_MATCHING': False,
    'BLEND_TRANSPARENCY': 0.1,
    'COLOR_BLENDING_STRENGTH': 0.2
}

# 🌙 PRESET SOFT/ELEGANTE
# Integrazione molto delicata e naturale
SOFT_PRESET = {
    'BLENDING_MODE': 'soft_light',
    'BLENDING_STRENGTH': 0.6,
    'EDGE_DETECTION_ENABLED': True,
    'EDGE_BLUR_RADIUS': 25,
    'ADAPTIVE_BLENDING': True,
    'COLOR_HARMONIZATION': True,
    'LUMINANCE_MATCHING': True,
    'BLEND_TRANSPARENCY': 0.4,
    'COLOR_BLENDING_STRENGTH': 0.5
}

# ⚡ PRESET DRAMMATICO
# Effetto intenso e contrastato
DRAMATIC_PRESET = {
    'BLENDING_MODE': 'multiply',
    'BLENDING_STRENGTH': 0.85,
    'EDGE_DETECTION_ENABLED': True,
    'EDGE_BLUR_RADIUS': 11,
    'ADAPTIVE_BLENDING': True,
    'COLOR_HARMONIZATION': False,
    'LUMINANCE_MATCHING': True,
    'BLEND_TRANSPARENCY': 0.15,
    'COLOR_BLENDING_STRENGTH': 0.3
}

# ✨ PRESET LUMINOSO
# Logo che si fonde con la luce dello sfondo
BRIGHT_PRESET = {
    'BLENDING_MODE': 'screen',
    'BLENDING_STRENGTH': 0.7,
    'EDGE_DETECTION_ENABLED': False,
    'EDGE_BLUR_RADIUS': 19,
    'ADAPTIVE_BLENDING': True,
    'COLOR_HARMONIZATION': True,
    'LUMINANCE_MATCHING': False,
    'BLEND_TRANSPARENCY': 0.3,
    'COLOR_BLENDING_STRENGTH': 0.4
}

# 🔥 PRESET INTENSO
# Luce dura e contrasti forti
INTENSE_PRESET = {
    'BLENDING_MODE': 'hard_light',
    'BLENDING_STRENGTH': 0.9,
    'EDGE_DETECTION_ENABLED': True,
    'EDGE_BLUR_RADIUS': 13,
    'ADAPTIVE_BLENDING': False,
    'COLOR_HARMONIZATION': False,
    'LUMINANCE_MATCHING': False,
    'BLEND_TRANSPARENCY': 0.1,
    'COLOR_BLENDING_STRENGTH': 0.25
}

# 🌈 PRESET PSICHEDELICO
# Effetto esclusione per colori invertiti
PSYCHEDELIC_PRESET = {
    'BLENDING_MODE': 'exclusion',
    'BLENDING_STRENGTH': 0.95,
    'EDGE_DETECTION_ENABLED': True,
    'EDGE_BLUR_RADIUS': 17,
    'ADAPTIVE_BLENDING': False,
    'COLOR_HARMONIZATION': False,
    'LUMINANCE_MATCHING': False,
    'BLEND_TRANSPARENCY': 0.05,
    'COLOR_BLENDING_STRENGTH': 0.1
}

# 💡 PRESET GLOW
# Effetto bagliore con color dodge
GLOW_PRESET = {
    'BLENDING_MODE': 'color_dodge',
    'BLENDING_STRENGTH': 0.75,
    'EDGE_DETECTION_ENABLED': True,
    'EDGE_BLUR_RADIUS': 23,
    'ADAPTIVE_BLENDING': True,
    'COLOR_HARMONIZATION': True,
    'LUMINANCE_MATCHING': False,
    'BLEND_TRANSPARENCY': 0.2,
    'COLOR_BLENDING_STRENGTH': 0.35
}

# 🖤 PRESET SCURO
# Effetto bruciatura per toni scuri
DARK_PRESET = {
    'BLENDING_MODE': 'color_burn',
    'BLENDING_STRENGTH': 0.8,
    'EDGE_DETECTION_ENABLED': True,
    'EDGE_BLUR_RADIUS': 15,
    'ADAPTIVE_BLENDING': True,
    'COLOR_HARMONIZATION': False,
    'LUMINANCE_MATCHING': True,
    'BLEND_TRANSPARENCY': 0.25,
    'COLOR_BLENDING_STRENGTH': 0.4
}

# 📐 PRESET GEOMETRICO
# Blending normale ma con bordi netti per effetti geometrici
GEOMETRIC_PRESET = {
    'BLENDING_MODE': 'normal',
    'BLENDING_STRENGTH': 1.0,
    'EDGE_DETECTION_ENABLED': True,
    'EDGE_BLUR_RADIUS': 7,
    'ADAPTIVE_BLENDING': False,
    'COLOR_HARMONIZATION': False,
    'LUMINANCE_MATCHING': False,
    'BLEND_TRANSPARENCY': 0.0,
    'COLOR_BLENDING_STRENGTH': 0.0
}

print("🎨 GUIDA RAPIDA AI PRESET:")
print("=" * 40)
print("🎬 CINEMATIC_PRESET    - Per video professionali")
print("🌟 ARTISTIC_PRESET     - Effetto creativo e surreale") 
print("🌙 SOFT_PRESET         - Integrazione delicata")
print("⚡ DRAMATIC_PRESET     - Effetto intenso")
print("✨ BRIGHT_PRESET       - Logo luminoso")
print("🔥 INTENSE_PRESET      - Contrasti forti")
print("🌈 PSYCHEDELIC_PRESET  - Colori invertiti")
print("💡 GLOW_PRESET         - Effetto bagliore")
print("🖤 DARK_PRESET         - Toni scuri")
print("📐 GEOMETRIC_PRESET    - Bordi netti")
print()
print("💡 TIP: Modifica i valori per personalizzare!")
