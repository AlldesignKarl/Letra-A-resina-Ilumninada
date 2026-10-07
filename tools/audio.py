"""Synthesize the soundtrack (ambient pads, music box, LED chimes, impact, whooshes).

Usage: python3 audio.py data.npz out.wav
Everything is generated here, so the music is original and royalty-free.
"""
import sys
import wave

import numpy as np
from scipy.signal import butter, fftconvolve, sosfilt

import timeline as TL

SR = 48000
N = int(TL.DUR * SR)
rng = np.random.default_rng(5)
mix = np.zeros((N, 2), np.float64)


def hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def place(sig, t0, gain=1.0, pan=0.0):
    i0 = int(t0 * SR)
    if sig.ndim == 1:
        l, r = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
        sig = np.stack([sig * l, sig * r], 1) * np.sqrt(2)
    i1 = min(i0 + len(sig), N)
    if i1 > i0 >= 0:
        mix[i0:i1] += sig[:i1 - i0] * gain


def env_ar(n, att, rel):
    e = np.ones(n)
    a, r = int(att * SR), int(rel * SR)
    if a:
        e[:a] = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, a))
    if r:
        e[-r:] *= 0.5 + 0.5 * np.cos(np.linspace(0, np.pi, r))
    return e


def pad(notes, t0, t1, gain, att=1.0, rel=1.2, bright=1.0, harm=7):
    n = int((t1 - t0 + rel) * SR)
    t = np.arange(n) / SR
    out = np.zeros((n, 2))
    for m in notes:
        f = hz(m)
        for ch, det in enumerate((-4, 4)):
            ff = f * 2 ** (det / 1200)
            vib = 1 + 0.0015 * np.sin(2 * np.pi * (0.18 + 0.05 * ch) * t + m)
            ph = 2 * np.pi * ff * np.cumsum(vib) / SR
            s = np.zeros(n)
            for k in range(1, harm + 1):
                if f * k > 9000:
                    break
                s += np.sin(k * ph + 0.7 * k) / k ** (2.2 - 0.5 * bright)
            out[:, ch] += s
    out /= len(notes) ** 0.6
    e = env_ar(n, att, rel)
    # slow swell
    e *= 0.85 + 0.15 * np.sin(2 * np.pi * 0.12 * t)
    place(out * e[:, None], t0, gain)


def bell(m, t0, gain, decay=1.4, pan=0.0, ratio=3.0, index=2.2, partial=True):
    f = hz(m)
    n = int(decay * 5 * SR)
    t = np.arange(n) / SR
    I = index * np.exp(-t / 0.08)
    s = np.sin(2 * np.pi * f * t + I * np.sin(2 * np.pi * f * ratio * t))
    if partial:
        s += 0.25 * np.sin(2 * np.pi * f * 2.76 * t) * np.exp(-t / (decay * 0.3))
    e = np.exp(-t / decay) * (1 - np.exp(-t / 0.002))
    place(s * e, t0, gain, pan)


def noise_band(n, lo, hi, order=2):
    sos = butter(order, [lo, hi], btype="band", fs=SR, output="sos")
    return sosfilt(sos, rng.normal(0, 1, n))


def riser(t0, t1, gain):
    n = int((t1 - t0) * SR)
    x = np.linspace(0, 1, n)
    bands = [(200, 600), (500, 1500), (1200, 3500), (3000, 8000), (6000, 14000)]
    out = np.zeros((n, 2))
    for i, (lo, hi) in enumerate(bands):
        c = i / (len(bands) - 1)
        w = np.exp(-((x - c) / 0.28) ** 2)
        for ch in range(2):
            out[:, ch] += noise_band(n, lo, hi) * w
    out *= (x ** 2.2)[:, None]
    # rising tone underneath
    f = 110 * 2 ** (2.5 * x)
    tone = np.sin(2 * np.pi * np.cumsum(f) / SR) * x ** 3 * 0.35
    out += tone[:, None]
    place(out / np.abs(out).max(), t0, gain)


def reverse_cymbal(t_hit, length, gain):
    n = int(length * SR)
    x = np.linspace(0, 1, n)
    s = np.stack([noise_band(n, 4000, 16000, 4), noise_band(n, 4000, 16000, 4)], 1)
    s *= (x ** 3.5)[:, None]
    place(s / np.abs(s).max(), t_hit - length, gain)


def boom(t0, gain):
    n = int(2.0 * SR)
    t = np.arange(n) / SR
    f = 38 + 60 * np.exp(-t / 0.12)
    s = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.55)
    s = np.tanh(s * 1.8) / np.tanh(1.8)
    click = noise_band(n, 80, 2500) * np.exp(-t / 0.015) * 0.5
    place((s + click) * (1 - np.exp(-t / 0.003)), t0, gain)


def whoosh(t0, length, gain, lo=400, hi=5000):
    n = int(length * SR)
    x = np.linspace(0, 1, n)
    e = np.sin(np.pi * x) ** 2
    nb = noise_band(n, lo, hi)
    pan = np.linspace(-0.8, 0.8, n)
    l, r = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
    s = np.stack([nb * l, nb * r], 1) * e[:, None]
    place(s / np.abs(s).max(), t0, gain)


# ----------------------------------------------------------------- score
D9 = [50, 57, 62, 66, 69, 73, 76]
Bm9 = [47, 54, 62, 66, 69, 73]
G7s = [43, 50, 59, 62, 66, 73]
Bm11 = [47, 54, 62, 64, 69, 73]
G9 = [43, 50, 59, 62, 66, 69, 73]
D9_low = [38, 50, 57, 62, 66, 69, 73, 76]

# day
pad(D9, 0.0, 5.1, 0.11, att=1.6, rel=1.0, bright=0.9)
motif = [78, 81, 85, 81, 88, 85, 81, 78, 76, 81, 85, 88, 90, 88, 85]
for i, m in enumerate(motif):
    bell(m, 0.35 + i * 0.32, 0.075 if i % 4 else 0.095, decay=1.1, pan=0.35 * np.sin(i * 1.3))
# dusk: darker harmony, slower and quieter arpeggio
pad(Bm9, 4.9, 6.7, 0.10, att=0.9, rel=0.9, bright=0.6)
pad(G7s, 6.5, 8.6, 0.09, att=0.9, rel=0.6, bright=0.3)
for i, m in enumerate([71, 74, 78, 81, 78, 74, 71, 67]):
    t = 5.15 + i * 0.42
    bell(m, t, 0.06 * (1 - i / 9), decay=1.2, pan=-0.3 + 0.08 * i)
riser(7.0, TL.FLASH, 0.10)

# LED cascade chimes (synced with the picture)
d = np.load(sys.argv[1])
H = d["H"] / d["H"][2, 2]
leds = d["leds"][:, :2]
p = np.c_[leds, np.ones(len(leds))] @ H.T
times = TL.led_schedule(p[:, :2] / p[:, 2:3])
pool = [74, 76, 78, 81, 83, 86, 88, 90, 93, 95, 98]
order = np.argsort(times)
bell(93, TL.FIRST_SPARK, 0.07, decay=0.9, pan=-0.4)
for rank, i in enumerate(order):
    m = pool[min(int(rank / len(order) * len(pool)), len(pool) - 1)]
    bell(m, times[i], 0.05, decay=0.7, pan=float(rng.uniform(-0.7, 0.7)), index=1.6)
reverse_cymbal(TL.FLASH, 1.1, 0.10)

# the letter lights up
boom(TL.FLASH, 0.55)
pad(D9_low + [81, 85], TL.FLASH, 14.4, 0.15, att=0.04, rel=1.4, bright=1.1)
for k in range(12):
    bell(int(rng.choice([86, 88, 90, 93, 95, 98])), TL.FLASH + k * 0.045 + rng.uniform(0, 0.03), 0.045,
         decay=1.4, pan=float(rng.uniform(-0.8, 0.8)))
mel = [(10.3, 81), (10.62, 85), (10.94, 88), (11.26, 90), (11.9, 88), (12.22, 85), (12.54, 81), (12.86, 83),
       (13.5, 85), (13.82, 81), (14.14, 78)]
for t, m in mel:
    bell(m, t, 0.08, decay=1.5, pan=0.2 * np.sin(t))
pad(Bm11, 11.9, 14.4, 0.08, att=0.8, rel=1.0, bright=0.8)

# end card
whoosh(TL.END_T0 - 0.15, 1.2, 0.12)
pad(G9, 14.2, 15.9, 0.11, att=0.5, rel=0.9, bright=0.9)
pad(D9_low, 15.6, TL.DUR, 0.14, att=0.9, rel=1.6, bright=1.0)
gl = [74, 76, 78, 81, 83, 86, 88, 90, 93, 95, 98]
for k, m in enumerate(gl):
    bell(m, TL.LOGO_T0 + k * (TL.LOGO_T1 - TL.LOGO_T0) / len(gl), 0.045, decay=1.2, pan=-0.6 + 1.2 * k / len(gl))
for t, m in zip(TL.ROWS_T, (81, 85, 88)):
    bell(m, t, 0.07, decay=0.9, ratio=2.0, index=1.0)
bell(90, TL.CTA_T + 0.1, 0.06, decay=1.6)
bell(86, TL.CTA_T + 0.1, 0.05, decay=1.6)
for k, m in enumerate((93, 95, 98, 100)):
    bell(m, TL.SHIMMER_T + k * 0.07, 0.035, decay=1.0, pan=-0.3 + 0.2 * k)
bell(74, 17.9, 0.07, decay=2.4)
bell(81, 17.95, 0.05, decay=2.4)

# ----------------------------------------------------------------- reverb + master
ir_n = int(2.6 * SR)
tt = np.arange(ir_n) / SR
ir = np.stack([rng.normal(0, 1, ir_n), rng.normal(0, 1, ir_n)], 1) * np.exp(-tt / 0.42)[:, None]
ir = sosfilt(butter(2, 5500, fs=SR, output="sos"), ir, axis=0)
ir[: int(0.012 * SR)] = 0
ir /= np.sqrt((ir ** 2).sum(0))
wet = np.stack([fftconvolve(mix[:, c], ir[:, c])[:N] for c in range(2)], 1)
out = mix * 0.8 + wet * 0.55
out = sosfilt(butter(2, 28, btype="high", fs=SR, output="sos"), out, axis=0)
fade_n = int((TL.DUR - TL.FADE_OUT + 0.4) * SR)
out[-fade_n:] *= np.linspace(1, 0, fade_n)[:, None] ** 1.5
out[: int(0.05 * SR)] *= np.linspace(0, 1, int(0.05 * SR))[:, None]
out /= np.abs(out).max()
out = np.tanh(out * 1.3) / np.tanh(1.3) * 0.89
pcm = (out * 32767).astype(np.int16)
with wave.open(sys.argv[2], "wb") as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes(pcm.tobytes())
print("ok", pcm.shape)
