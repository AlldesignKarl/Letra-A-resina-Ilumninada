"""Shared timing for picture and sound (seconds)."""
import numpy as np

FPS = 30
DUR = 19.8
W, H = 1080, 1920

# scene anchors
NIGHT_T0, NIGHT_T1 = 4.4, 7.9      # night falls (top to bottom)
FIRST_SPARK = 8.30                  # first tiny LED
LED_T0, LED_T1 = 8.42, 9.30         # LED cascade
FILL_T0, FILL_T1 = 9.15, 9.60       # whole letter lights up
FLASH = 9.58                        # light burst peak
MORPH_T0, MORPH_T1 = 9.45, 10.45    # day-scene -> real night photo
END_T0, END_T1 = 14.2, 15.35        # camera pulls back to end card
LOGO_T0, LOGO_T1 = 15.0, 15.95      # logo is "written" with light
ROWS_T = [16.1, 16.25, 16.4]        # contact rows
CTA_T = 16.7
SHIMMER_T = 17.6
FADE_OUT = 19.4


def led_schedule(led_world_xy):
    """Ignition time per LED: bottom to top with a little randomness, accelerating."""
    rng = np.random.default_rng(7)
    y = led_world_xy[:, 1] + rng.normal(0, 60, len(led_world_xy))
    order = np.argsort(-y)
    n = len(order)
    times = np.empty(n)
    for rank, idx in enumerate(order):
        x = rank / max(n - 1, 1)
        times[idx] = LED_T0 + (LED_T1 - LED_T0) * x ** 0.75
    return times
