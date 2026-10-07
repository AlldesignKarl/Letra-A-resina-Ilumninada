# AlldesignKarl · Letra A de resina iluminada — vídeo promocional

Vídeo vertical (1080×1920, 30 fps, 19,8 s) para Reels / TikTok / Stories / WhatsApp.

## Archivos finales (`video/`)

| Archivo | Uso |
|---|---|
| `AlldesignKarl_Letra_A_1080x1920.mp4` | Máxima calidad (Instagram, TikTok, Facebook, web) |
| `AlldesignKarl_Letra_A_WhatsApp_720p.mp4` | Versión ligera (~8 MB) para enviar por WhatsApp |
| `portada.jpg` | Portada / miniatura del Reel |

## Guion

1. **0–4,4 s · Día**: luz de ventana con rayos de sol y polvo flotando, un brillo recorre la resina.
   Texto: *ALLDESIGNKARL · Flores naturales atrapadas en resina*.
2. **4,4–7,9 s · Anochece**: la noche "cae" de arriba abajo (atardecer dorado → hora azul → noche).
   Texto: *Y cuando cae la noche…*
3. **8,3–9,6 s · Se enciende**: las lucecitas LED se encienden una a una, de abajo arriba, y la luz llena la letra.
4. **9,6 s · Estallido de luz**: destello, rayos y chispas. Pasa a la foto real de noche.
   Texto: *…la magia se enciende · TU INICIAL · TUS FLORES · TU LUZ*.
5. **14,2–19,8 s · Cierre**: el logo *AlldesignKarl* se escribe con un trazo de luz, y aparecen
   alldesignkarl.com · Vanessa@alldesignkarl.com · 614 65 37 36 · *Pide la tuya personalizada*.

Música y efectos de sonido sintetizados desde cero (originales, sin derechos de autor).

## Cómo regenerarlo (`tools/`)

```bash
python3 tools/upscale.py RealESRGAN_x4plus.pth dia_src.png dia_x4.png   # (y noche) super-resolución x4
python3 tools/prep.py <carpeta_hd> build/data.npz                        # máscaras, alineado, LEDs
python3 tools/audio.py build/data.npz build/music.wav                    # banda sonora
python3 tools/render.py build/data.npz <carpeta_fuentes> build/video.mp4 # vídeo
```

Fuentes tipográficas: Cormorant Garamond, Montserrat y Great Vibes (Google Fonts, licencia OFL).
Los textos y tiempos están en `tools/timeline.py` y `build_text()` de `tools/render.py`.
