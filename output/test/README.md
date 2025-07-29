# 🧪 Test Files / File di Test

Questa cartella contiene tutti i file generati in **TEST MODE**.

## Caratteristiche Test Mode:
- 🎬 **FPS**: 10fps (invece di 30fps)
- 📱 **Risoluzione ridotta**: Per rendering più veloce
- 🏷️ **Nome file**: Include `_TEST` nel nome
- ⚡ **Scopo**: Test rapidi e sperimentazione parametri

## Struttura File:
```
crystalpy_YYYYMMDD_HHMMSS_TEST_[simbolo].mp4
crystalpy_YYYYMMDD_HHMMSS_TEST_[simbolo]_with_audio.mp4
```

## Come Attivare Test Mode:
Nel file `natisone_trip_generator.py`, imposta:
```python
Config.TEST_MODE = True
```

---
*File generati automaticamente dal Natisone Trip Generator* ✨
