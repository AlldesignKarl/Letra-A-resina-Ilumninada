"""Render the AlldesignKarl promo video (1080x1920, 30 fps).

Usage:
  python3 render.py data.npz fonts_dir out.mp4            # full video (no audio)
  python3 render.py data.npz fonts_dir outdir --stills 1,5.5,9.6
"""
import math
import sys
from multiprocessing import Pool

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.interpolate import PchipInterpolator

import timeline as TL

W, H, FPS, DUR = TL.W, TL.H, TL.FPS, TL.DUR
BASE_S = H / 1094.0          # screen px per world px at zoom 1
LETTER_C = np.array([447.0, 630.0])
LR = 4                        # low-res factor for glow/rays work
LW, LH = W // LR, H // LR

G = {}                        # per-process globals


# ----------------------------------------------------------------------------- helpers
def clamp01(x):
    return min(max(x, 0.0), 1.0)


def smooth(x):
    x = clamp01(x)
    return x * x * (3 - 2 * x)


def ease_out(x):
    x = clamp01(x)
    return 1 - (1 - x) ** 3


def ease_in(x):
    x = clamp01(x)
    return x ** 3


def ease_io(x):
    x = clamp01(x)
    return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


def win(t, a, b):
    return smooth((t - a) / (b - a)) if b != a else float(t >= a)


def mat_cam(cx, cy, z):
    s = BASE_S * z
    return np.array([[s, 0, W / 2 - s * cx], [0, s, H / 2 - s * cy], [0, 0, 1]], np.float64)


def warp(img, M, border=cv2.BORDER_REFLECT_101, size=(W, H)):
    return cv2.warpPerspective(img, M, size, flags=cv2.INTER_LINEAR, borderMode=border)


def apply_h(M, pts):
    p = np.c_[pts, np.ones(len(pts))] @ M.T
    return p[:, :2] / p[:, 2:3]


def splat(img, x, y, sigma, color, amp=1.0, disc=False):
    """Additive soft dot with sub-pixel position."""
    r = int(math.ceil(sigma * (1.6 if disc else 3.0))) + 1
    x0, y0 = int(math.floor(x)) - r, int(math.floor(y)) - r
    x1, y1 = x0 + 2 * r + 2, y0 + 2 * r + 2
    hh, ww = img.shape[:2]
    if x1 <= 0 or y1 <= 0 or x0 >= ww or y0 >= hh:
        return
    ys = np.arange(y0, y1, dtype=np.float32)[:, None] - y
    xs = np.arange(x0, x1, dtype=np.float32)[None, :] - x
    d2 = xs * xs + ys * ys
    if disc:
        d = np.sqrt(d2)
        k = np.clip((sigma - d) / max(sigma * 0.22, 0.8) + 0.5, 0, 1)
        k = k * (0.75 + 0.25 * np.clip(d / sigma, 0, 1))   # slightly brighter rim
    else:
        k = np.exp(-d2 / (2 * sigma * sigma))
    cx0, cy0 = max(x0, 0), max(y0, 0)
    cx1, cy1 = min(x1, ww), min(y1, hh)
    k = k[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
    img[cy0:cy1, cx0:cx1] += (k * amp)[..., None] * np.asarray(color, np.float32)


def make_flare(size=161):
    c = size // 2
    y, x = np.mgrid[-c:c + 1, -c:c + 1].astype(np.float32)
    r = np.sqrt(x * x + y * y) + 1e-3
    core = np.exp(-r * r / (2 * 2.2 ** 2)) + 0.35 * np.exp(-r * r / (2 * 7 ** 2))
    streak = np.exp(-np.abs(x) / 22) * np.exp(-(y / 1.1) ** 2) + np.exp(-np.abs(y) / 22) * np.exp(-(x / 1.1) ** 2)
    xd, yd = (x + y) / math.sqrt(2), (x - y) / math.sqrt(2)
    diag = np.exp(-np.abs(xd) / 9) * np.exp(-(yd / 0.9) ** 2) + np.exp(-np.abs(yd) / 9) * np.exp(-(xd / 0.9) ** 2)
    f = core + 0.55 * streak + 0.18 * diag
    f *= np.clip(1 - r / c, 0, 1) ** 1.5
    return f.astype(np.float32)


def add_sprite(img, spr, x, y, scale, color, amp):
    if amp <= 1e-3:
        return
    if abs(scale - 1) > 0.02:
        n = max(9, int(spr.shape[0] * scale) | 1)
        spr = cv2.resize(spr, (n, n), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    n = spr.shape[0]
    x0, y0 = int(round(x)) - n // 2, int(round(y)) - n // 2
    hh, ww = img.shape[:2]
    cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x0 + n, ww), min(y0 + n, hh)
    if cx1 <= cx0 or cy1 <= cy0:
        return
    k = spr[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
    img[cy0:cy1, cx0:cx1] += (k * amp)[..., None] * np.asarray(color, np.float32)


# ----------------------------------------------------------------------------- camera
def build_camera():
    # t, cx, cy, log(zoom)
    keys = [
        (0.0, 452, 570, 1.05),
        (4.9, 447, 560, 1.15),
        (8.3, 447, 562, 1.20),
        (9.6, 447, 566, 1.235),
        (10.9, 448, 552, 1.15),
        (14.15, 447, 548, 1.205),
        (15.35, 447, 992, 0.655),
        (DUR, 447, 990, 0.635),
    ]
    k = np.array(keys, np.float64)
    fx = PchipInterpolator(k[:, 0], k[:, 1])
    fy = PchipInterpolator(k[:, 0], k[:, 2])
    fz = PchipInterpolator(k[:, 0], np.log(k[:, 3]))

    def cam(t):
        z = math.exp(float(fz(t)))
        # tiny punch-in on the light burst
        z *= 1 + 0.03 * math.exp(-((t - TL.FLASH - 0.05) / 0.16) ** 2)
        return float(fx(t)), float(fy(t)), z
    return cam


# ----------------------------------------------------------------------------- text
class Glyphs:
    """A line of text pre-rendered per character, animatable per character."""

    def __init__(self, text, font, cx, cy, tracking=0.0, pad=70):
        self.text = text
        asc, desc = font.getmetrics()
        widths = [font.getlength(text[:i]) + i * tracking for i in range(len(text) + 1)]
        total = widths[-1] - tracking
        self.w = int(total + 2 * pad)
        self.h = int(asc + desc + 2 * pad)
        self.x0 = int(round(cx - self.w / 2))
        self.y0 = int(round(cy - (asc + desc) / 2 - pad))
        self.chars = []
        for i, ch in enumerate(text):
            if ch == " ":
                continue
            im = Image.new("L", (self.w, self.h), 0)
            ImageDraw.Draw(im).text((pad + widths[i], pad), ch, font=font, fill=255)
            a = np.asarray(im, np.float32) / 255
            ys, xs = np.nonzero(a > 0)
            if len(xs) == 0:
                continue
            bx0, bx1 = max(xs.min() - 2, 0), min(xs.max() + 3, self.w)
            by0, by1 = max(ys.min() - 2, 0), min(ys.max() + 3, self.h)
            self.chars.append((i, bx0, by0, a[by0:by1, bx0:bx1].copy()))
        self.n = len(text)

    def alpha(self, t, t_in, dur_in=0.7, stagger=0.035, rise=36, blur_in=7.0,
              t_out=None, dur_out=0.5, stagger_out=0.012, rise_out=24, blur_out=6.0, reverse=False):
        buf = np.zeros((self.h, self.w), np.float32)
        any_vis = False
        for i, bx0, by0, spr in self.chars:
            j = (self.n - 1 - i) if reverse else i
            e = ease_out((t - t_in - j * stagger) / dur_in)
            if e <= 0:
                continue
            a, dy, bl = e, (1 - e) * rise, (1 - e) * blur_in
            if t_out is not None:
                e2 = ease_in((t - t_out - i * stagger_out) / dur_out)
                a *= 1 - e2
                dy -= e2 * rise_out
                bl += e2 * blur_out
            if a <= 0.003:
                continue
            any_vis = True
            s = spr
            if bl > 0.35:
                p = int(bl * 3) + 1
                s = cv2.GaussianBlur(np.pad(s, p), (0, 0), bl)
                ox, oy = bx0 - p, by0 - p
            else:
                ox, oy = bx0, by0
            oy += int(round(dy))
            hh, ww = s.shape
            cx0, cy0, cx1, cy1 = max(ox, 0), max(oy, 0), min(ox + ww, self.w), min(oy + hh, self.h)
            if cx1 > cx0 and cy1 > cy0:
                region = buf[cy0:cy1, cx0:cx1]
                np.maximum(region, s[cy0 - oy:cy1 - oy, cx0 - ox:cx1 - ox] * a, out=region)
        return buf if any_vis else None


def over(frame, x0, y0, alpha, color):
    """Composite a color (scalar rgb or HxWx3 field) through alpha onto frame (sRGB float)."""
    h, w = alpha.shape
    fx0, fy0, fx1, fy1 = max(x0, 0), max(y0, 0), min(x0 + w, W), min(y0 + h, H)
    if fx1 <= fx0 or fy1 <= fy0:
        return
    a = alpha[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0][..., None]
    if isinstance(color, np.ndarray) and color.ndim == 3:
        c = color[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0]
    else:
        c = np.asarray(color, np.float32)
    reg = frame[fy0:fy1, fx0:fx1]
    reg *= 1 - a
    reg += c * a


def screen_add(frame, x0, y0, alpha, color, amp=1.0):
    h, w = alpha.shape
    fx0, fy0, fx1, fy1 = max(x0, 0), max(y0, 0), min(x0 + w, W), min(y0 + h, H)
    if fx1 <= fx0 or fy1 <= fy0:
        return
    a = alpha[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0][..., None] * amp
    reg = frame[fy0:fy1, fx0:fx1]
    c = np.asarray(color, np.float32) * a
    reg[:] = 1 - (1 - reg) * (1 - np.clip(c, 0, 1))


def gold_field(h, w, top=None, t_shimmer=None, x_off=0):
    top = (1.0, 0.94, 0.74)
    mid = (0.94, 0.77, 0.42)
    bot = (0.70, 0.50, 0.22)
    y = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    c = np.empty((h, w, 3), np.float32)
    for k in range(3):
        col = np.where(y < 0.5, top[k] + (mid[k] - top[k]) * (y / 0.5), mid[k] + (bot[k] - mid[k]) * ((y - 0.5) / 0.5))
        c[..., k] = col
    if t_shimmer is not None:
        xs = np.arange(w, dtype=np.float32)[None, :] + x_off
        ys = np.arange(h, dtype=np.float32)[:, None]
        band = np.exp(-(((xs + ys * 0.45) - t_shimmer) / 38.0) ** 2)
        c = c + (1 - c) * band[..., None] * 0.85
    return c


def line_icon(kind, size=72, color=1.0):
    """Gold line icons drawn at 4x then reduced: globe, mail, phone (inside a circle)."""
    S4 = size * 4
    im = Image.new("L", (S4, S4), 0)
    d = ImageDraw.Draw(im)
    lw = 7
    d.ellipse((lw, lw, S4 - lw, S4 - lw), outline=255, width=lw)
    c = S4 / 2
    if kind == "globe":
        r = S4 * 0.26
        d.ellipse((c - r, c - r, c + r, c + r), outline=255, width=lw)
        d.ellipse((c - r * 0.45, c - r, c + r * 0.45, c + r), outline=255, width=lw)
        d.line((c - r, c, c + r, c), fill=255, width=lw)
        d.line((c - r * 0.87, c - r * 0.5, c + r * 0.87, c - r * 0.5), fill=255, width=lw - 2)
        d.line((c - r * 0.87, c + r * 0.5, c + r * 0.87, c + r * 0.5), fill=255, width=lw - 2)
    elif kind == "mail":
        w2, h2 = S4 * 0.27, S4 * 0.19
        d.rounded_rectangle((c - w2, c - h2, c + w2, c + h2), radius=10, outline=255, width=lw)
        d.line((c - w2 + 6, c - h2 + 6, c, c + h2 * 0.25, c + w2 - 6, c - h2 + 6), fill=255, width=lw, joint="curve")
    elif kind == "phone":
        w2, h2 = S4 * 0.15, S4 * 0.27
        d.rounded_rectangle((c - w2, c - h2, c + w2, c + h2), radius=18, outline=255, width=lw)
        d.line((c - w2 * 0.35, c - h2 + 22, c + w2 * 0.35, c - h2 + 22), fill=255, width=lw - 1)
        d.ellipse((c - 8, c + h2 - 34, c + 8, c + h2 - 18), fill=255)
    a = np.asarray(im.resize((size, size), Image.LANCZOS), np.float32) / 255
    return a


def ornament(width=420, height=40):
    S4 = 4
    im = Image.new("L", (width * S4, height * S4), 0)
    d = ImageDraw.Draw(im)
    cy = height * S4 / 2
    cx = width * S4 / 2
    d.line((0, cy, cx - 46, cy), fill=255, width=6)
    d.line((cx + 46, cy, width * S4, cy), fill=255, width=6)
    r = 26
    d.polygon([(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)], fill=255)
    a = np.asarray(im.resize((width, height), Image.LANCZOS), np.float32) / 255
    # taper line ends
    x = np.abs(np.linspace(-1, 1, width, dtype=np.float32))[None, :]
    return a * np.clip((1 - x) / 0.35, 0, 1)


def build_text(fonts_dir):
    F = fonts_dir.rstrip("/") + "/"

    def font(name, size, var=None):
        f = ImageFont.truetype(F + name, size)
        if var:
            f.set_variation_by_name(var)
        return f

    it = lambda s: font("CormorantGaramond-Italic[wght].ttf", s, "Medium Italic")
    it_sb = lambda s: font("CormorantGaramond-Italic[wght].ttf", s, "SemiBold Italic")
    sans = lambda s, v="Medium": font("Montserrat[wght].ttf", s, v)
    T = {}
    T["brand"] = Glyphs("ALLDESIGNKARL", sans(30, "SemiBold"), W / 2, 262, tracking=13)
    T["day1"] = Glyphs("Flores naturales", it(96), W / 2, 352)
    T["day2"] = Glyphs("atrapadas en resina", it(96), W / 2, 448)
    T["dusk"] = Glyphs("Y cuando cae la noche…", it(86), W / 2, 372)
    T["night1"] = Glyphs("…la magia se enciende", it_sb(98), W / 2, 330)
    T["night2"] = Glyphs("TU INICIAL · TUS FLORES · TU LUZ", sans(27, "Medium"), W / 2, 436, tracking=7)
    T["logo"] = Glyphs("AlldesignKarl", font("GreatVibes-Regular.ttf", 158), W / 2, 1030, pad=90)
    T["tag"] = Glyphs("LETRAS DE RESINA ILUMINADAS", sans(25, "Medium"), W / 2, 1186, tracking=8)
    T["cta"] = Glyphs("Pide la tuya personalizada", it_sb(58), W / 2, 1612)

    rows = [("globe", "alldesignkarl.com"), ("mail", "Vanessa@alldesignkarl.com"), ("phone", "614 65 37 36")]
    f_row = sans(42, "Medium")
    icon = 72
    gap = 30
    widest = max(f_row.getlength(s) for _, s in rows)
    block_w = icon + gap + widest
    bx = W / 2 - block_w / 2
    T["rows"] = []
    for k, (kind, s) in enumerate(rows):
        cy = 1300 + k * 96
        tw = f_row.getlength(s)
        g = Glyphs(s, f_row, bx + icon + gap + tw / 2, cy)
        T["rows"].append((line_icon(kind, icon), int(bx), int(cy - icon / 2), g))
    T["orn"] = ornament()
    return T


def draw_text(frame, t):
    T = G["text"]
    dark = (0.16, 0.12, 0.10)
    cream = (0.98, 0.94, 0.87)

    def comp(g, a, color, halo=None, halo_amp=0.0, glow=None, glow_amp=0.0, sig=14):
        if a is None:
            return
        if halo is not None:
            hb = cv2.GaussianBlur(a, (0, 0), sig)
            over(frame, g.x0, g.y0, np.clip(hb * halo_amp, 0, 1), halo)
        if glow is not None:
            gb = cv2.GaussianBlur(a, (0, 0), sig)
            screen_add(frame, g.x0, g.y0, gb, glow, glow_amp)
        over(frame, g.x0, g.y0, a, color)

    # --- day
    if t < 5.3:
        g = T["brand"]
        a = g.alpha(t, 0.45, dur_in=0.9, stagger=0.045, rise=0, blur_in=10, t_out=4.35, dur_out=0.5, stagger_out=0.0)
        comp(g, a if a is None else a * 0.85, dark, halo=(1, 1, 1), halo_amp=0.55)
        g = T["day1"]
        a = g.alpha(t, 0.85, t_out=4.4, stagger_out=0.015)
        comp(g, a, dark, halo=(1, 1, 1), halo_amp=0.7, sig=16)
        g = T["day2"]
        a = g.alpha(t, 1.25, t_out=4.5, stagger_out=0.015)
        comp(g, a, dark, halo=(1, 1, 1), halo_amp=0.7, sig=16)
    # --- dusk
    if 5.5 < t < 9.2:
        g = T["dusk"]
        a = g.alpha(t, 5.95, dur_in=0.8, stagger=0.04, t_out=8.55, dur_out=0.45)
        comp(g, a, cream, halo=(0.02, 0.03, 0.07), halo_amp=0.55, sig=18)
    # --- night
    if 9.9 < t < 14.8:
        g = T["night1"]
        a = g.alpha(t, 10.15, dur_in=0.75, stagger=0.04, rise=30, blur_in=9, t_out=13.85, dur_out=0.5)
        if a is not None:
            col = gold_field(g.h, g.w, t_shimmer=(t - 11.3) * 900 - 200)
            comp(g, a, col, halo=(0.0, 0.0, 0.02), halo_amp=0.5, sig=20)
            gb = cv2.GaussianBlur(a, (0, 0), 10)
            screen_add(frame, g.x0, g.y0, gb, (1.0, 0.72, 0.30), 0.55)
        g = T["night2"]
        a = g.alpha(t, 10.9, dur_in=0.8, stagger=0.02, rise=0, blur_in=8, t_out=13.95, dur_out=0.45, stagger_out=0.0)
        comp(g, a if a is None else a * 0.92, cream, halo=(0, 0, 0.02), halo_amp=0.5, sig=12)
    # --- end card
    if t > 14.9:
        draw_end_card(frame, t)


def draw_end_card(frame, t):
    T = G["text"]
    cream = (0.98, 0.94, 0.87)
    # logo written with light
    g = T["logo"]
    a = g.alpha(t, TL.LOGO_T0 - 0.1, dur_in=0.9, stagger=0.0, rise=0, blur_in=0)
    if a is not None:
        p = ease_io((t - TL.LOGO_T0) / (TL.LOGO_T1 - TL.LOGO_T0))
        xs_on = np.nonzero(a.max(0) > 0.05)[0]
        lx0, lx1 = xs_on.min(), xs_on.max()
        front = lx0 - 40 + (lx1 - lx0 + 80) * p
        x = np.arange(g.w, dtype=np.float32)
        wipe = np.clip((front - x) / 55.0, 0, 1)[None, :]
        a = a * wipe
        sh = (t - TL.SHIMMER_T) * 1100 if t > TL.SHIMMER_T - 0.3 else None
        col = gold_field(g.h, g.w, t_shimmer=sh)
        gb = cv2.GaussianBlur(a, (0, 0), 16)
        over(frame, g.x0, g.y0, np.clip(gb * 0.6, 0, 1), (0.0, 0.0, 0.01))
        screen_add(frame, g.x0, g.y0, gb, (1.0, 0.70, 0.30), 0.45)
        over(frame, g.x0, g.y0, a, col)
        # the pen of light
        if 0 < p < 1:
            col_ys = np.nonzero(a[:, int(np.clip(front - 30, 0, g.w - 1))] > 0.3)[0]
            fy = g.y0 + (col_ys.mean() if len(col_ys) else g.h / 2)
            fl = np.zeros((161, 161, 3), np.float32)
            add_sprite(fl, G["flare"], 80, 80, 1.0, (1.0, 0.85, 0.55), 1.0)
            screen_add(frame, int(g.x0 + front - 30 - 80), int(fy - 80), np.clip(fl.max(2), 0, 1), (1.0, 0.9, 0.7), 1.0)
    # ornament
    e = ease_out((t - 15.65) / 0.8)
    if e > 0:
        o = T["orn"]
        h, w = o.shape
        x = np.abs(np.arange(w, dtype=np.float32) - w / 2)[None, :]
        m = np.clip((e * w / 2 - x) / 20.0, 0, 1)
        over(frame, int(W / 2 - w / 2), int(1128 - h / 2), o * m * 0.95, (0.90, 0.74, 0.45))
    g = T["tag"]
    a = g.alpha(t, 15.85, dur_in=0.7, stagger=0.012, rise=0, blur_in=6)
    if a is not None:
        over(frame, g.x0, g.y0, a * 0.8, cream)
    # contact rows
    for k, (icon, ix, iy, g) in enumerate(T["rows"]):
        t0 = TL.ROWS_T[k]
        e = ease_out((t - t0) / 0.6)
        if e <= 0:
            continue
        dy = int(round((1 - e) * 34))
        sc = 0.55 + 0.45 * ease_out((t - t0) / 0.45)
        n = int(icon.shape[0] * sc) | 1
        ic = cv2.resize(icon, (n, n), interpolation=cv2.INTER_AREA)
        off = (icon.shape[0] - n) // 2
        gb = cv2.GaussianBlur(np.pad(ic, 20), (0, 0), 8)
        screen_add(frame, ix + off - 20, iy + off - 20 + dy, gb, (1.0, 0.7, 0.3), 0.35 * e)
        over(frame, ix + off, iy + off + dy, ic * e, (0.93, 0.78, 0.48))
        a = g.alpha(t, t0 + 0.08, dur_in=0.6, stagger=0.008, rise=0, blur_in=6)
        if a is not None:
            sh = np.zeros_like(a)
            sh[6:, 3:] = a[:-6, :-3]
            over(frame, g.x0, g.y0 + dy, cv2.GaussianBlur(sh, (0, 0), 5) * 0.6, (0, 0, 0))
            over(frame, g.x0, g.y0 + dy, a, (0.98, 0.95, 0.90))
    # call to action
    g = T["cta"]
    a = g.alpha(t, TL.CTA_T, dur_in=0.8, stagger=0.02, rise=22, blur_in=6)
    if a is not None:
        col = gold_field(g.h, g.w, t_shimmer=(t - 18.2) * 900 - 100)
        gb = cv2.GaussianBlur(a, (0, 0), 12)
        screen_add(frame, g.x0, g.y0, gb, (1.0, 0.7, 0.3), 0.4)
        over(frame, g.x0, g.y0, a, col)
        # small twinkling stars both sides
        xs_on = np.nonzero(a.max(0) > 0.1)[0]
        if len(xs_on):
            e = ease_out((t - TL.CTA_T - 0.5) / 0.5)
            for side, x in ((0, g.x0 + xs_on.min() - 34), (1, g.x0 + xs_on.max() + 34)):
                tw = 0.65 + 0.35 * math.sin(t * 5.0 + side * 2.1)
                fl = np.zeros((81, 81, 3), np.float32)
                add_sprite(fl, G["flare"], 40, 40, 0.5, (1, 0.85, 0.5), 1.0)
                screen_add(frame, int(x - 40), int(g.y0 + g.h / 2 - 40 + 4), np.clip(fl.max(2), 0, 1),
                           (1.0, 0.86, 0.55), e * tw)


# ----------------------------------------------------------------------------- scene
def init_worker(data_path, fonts_dir):
    d = np.load(data_path)
    G["day"] = d["day"]
    G["night"] = d["night_ext"]
    G["pad"] = int(d["ext_pad"])
    G["K"] = int(d["K"])
    G["m1"] = d["m1"]
    G["m2"] = d["m2"]
    G["H"] = d["H"] / d["H"][2, 2]
    G["S"] = d["S"]
    G["Hinv"] = np.linalg.inv(G["H"])
    leds = d["leds"]
    G["leds"] = leds
    led_world = apply_h(G["H"], leds[:, :2])
    G["led_t"] = TL.led_schedule(led_world)
    rng = np.random.default_rng(11)
    G["led_f"] = rng.uniform(0.6, 1.6, len(leds))
    G["led_ph"] = rng.uniform(0, 2 * np.pi, len(leds))
    G["led_star"] = rng.random(len(leds)) < 0.45
    K = G["K"]
    # spill light map (world, 2x): glow around the letter + pool of light on the table
    sp = cv2.GaussianBlur(G["m1"], (0, 0), 30 * K)
    sp /= sp.max()
    pool = np.zeros_like(sp)
    cv2.ellipse(pool, (447 * K, 905 * K), (290 * K, 45 * K), 0, 0, 360, 1.0, -1)
    pool = cv2.GaussianBlur(pool, (0, 0), 26 * K)
    # light falls on the surroundings, not as a flat tint over the letter itself
    G["spill"] = np.clip(sp * 1.4 * (1 - G["m1"]) + pool * 0.9, 0, 1.6)
    G["flare"] = make_flare()
    G["cam"] = build_camera()
    G["text"] = build_text(fonts_dir)
    # particles
    rng = np.random.default_rng(3)
    n = 90
    G["dust"] = dict(x=rng.uniform(0, W, n), y=rng.uniform(0, H * 0.85, n), vx=rng.uniform(-6, 14, n),
                     vy=rng.uniform(-10, 8, n), s=rng.uniform(0.9, 2.8, n), a=rng.uniform(0.3, 1.0, n),
                     ph=rng.uniform(0, 6.28, n), blur=rng.random(n) < 0.15)
    n = 46
    G["bokeh"] = dict(x=rng.uniform(-40, W + 40, n), y=rng.uniform(0, H, n), v=rng.uniform(22, 70, n),
                      sw=rng.uniform(8, 34, n), sf=rng.uniform(0.15, 0.5, n), r=rng.uniform(3, 26, n),
                      a=rng.uniform(0.25, 1.0, n), ph=rng.uniform(0, 6.28, n), t0=rng.uniform(0, 1.2, n),
                      hue=rng.uniform(0, 1, n))
    # sparks from the letter for the light burst
    ys, xs = np.nonzero(G["m1"][::4, ::4] > 0.5)
    idx = rng.choice(len(xs), 150, replace=False)
    wx, wy = xs[idx] * 4 / K, ys[idx] * 4 / K
    dirv = np.c_[wx - LETTER_C[0], wy - LETTER_C[1]]
    dirv /= np.linalg.norm(dirv, axis=1, keepdims=True) + 1e-6
    ang = rng.normal(0, 0.35, len(idx))
    ca, sa = np.cos(ang), np.sin(ang)
    dirv = np.c_[dirv[:, 0] * ca - dirv[:, 1] * sa, dirv[:, 0] * sa + dirv[:, 1] * ca]
    spd = rng.uniform(250, 1300, len(idx))
    G["sparks"] = dict(wx=wx, wy=wy, vx=dirv[:, 0] * spd, vy=dirv[:, 1] * spd - 120, life=rng.uniform(0.5, 1.5, len(idx)),
                       s=rng.uniform(1.2, 2.6, len(idx)), a=rng.uniform(0.6, 1.8, len(idx)))
    G["grain"] = [np.random.default_rng(100 + i).normal(0, 1, (H, W)).astype(np.float32) for i in range(6)]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    G["vig"] = ((xx / W - 0.5) ** 2 * 1.3 + (yy / H - 0.5) ** 2).astype(np.float32)
    G["xx_lr"], G["yy_lr"] = np.mgrid[0:LH, 0:LW][1].astype(np.float32) * LR, np.mgrid[0:LH, 0:LW][0].astype(np.float32) * LR


def w2_at(t):
    """Placement of the night photo in world space (homography -> natural)."""
    q = ease_io((t - TL.MORPH_T0) / (TL.MORPH_T1 - TL.MORPH_T0))
    return G["H"] * (1 - q) + G["S"] * q


def night_rows(t):
    """Per-row grading multipliers: night falls from the top of the frame down."""
    P = smooth((t - TL.NIGHT_T0) / (TL.NIGHT_T1 - TL.NIGHT_T0))
    d = np.linspace(0, 1, H, dtype=np.float32)
    Kf = 0.6
    p = np.clip(P * (1 + Kf) - Kf * d, 0, 1)
    xp = [0.0, 0.3, 0.65, 1.0]
    exp_ = np.interp(p, xp, [1.0, 0.95, 0.42, 0.050])
    r = np.interp(p, xp, [1.0, 1.16, 0.72, 0.52])
    g = np.interp(p, xp, [1.0, 0.92, 0.82, 0.68])
    b = np.interp(p, xp, [1.0, 0.70, 1.10, 1.30])
    sat = np.interp(p, xp, [1.0, 1.08, 0.78, 0.55])
    mult = np.stack([r, g, b], 1) * exp_[:, None]
    lift = np.array([0.0012, 0.0022, 0.0060], np.float32)[None, :] * p[:, None]
    return P, p, mult.astype(np.float32)[:, None, :], sat.astype(np.float32)[:, None, None], lift[:, None, :]


def god_rays(t, P):
    sx, sy = -0.3 * W, -0.25 * H
    x, y = G["xx_lr"], G["yy_lr"]
    th = np.arctan2(y - sy, x - sx)
    r = np.hypot(x - sx, y - sy)
    b = np.zeros_like(th)
    for f, ph, sp, a in ((23, 0.3, 0.05, 1.0), (41, 1.7, -0.08, 0.7), (67, 2.9, 0.11, 0.45), (13, 4.1, 0.03, 0.8)):
        b += a * (0.5 + 0.5 * np.sin(f * th + ph + sp * t * 6.28))
    b = np.clip((b / 2.95 - 0.38) * 2.2, 0, 1) ** 1.6
    wedge = np.clip((th - 0.25) / 0.25, 0, 1) * np.clip((1.25 - th) / 0.3, 0, 1)
    fall = np.exp(-(r / (1.25 * H)) ** 2)
    f = cv2.GaussianBlur((b * wedge * fall).astype(np.float32), (0, 0), 2.0)
    return f


def render_frame(fi):
    t = fi / FPS
    cam = G["cam"]
    cx, cy, z = cam(t)
    C = mat_cam(cx, cy, z)
    K = G["K"]
    D = np.diag([1.0 / K, 1.0 / K, 1.0])
    pad = G["pad"]
    E = np.array([[1.0 / K, 0, -pad / K], [0, 1.0 / K, -pad / K], [0, 0, 1]])
    W2 = w2_at(t)
    M_night = C @ W2 @ E
    morph_q = win(t, TL.MORPH_T0 + 0.05, TL.MORPH_T1)
    lit_ramp = win(t, TL.LED_T0 - 0.1, TL.FILL_T1)

    # ------------------------------------------------ base plate
    if morph_q < 1.0:
        P1 = W2 @ G["Hinv"]
        M_day = C @ P1 @ D
        img = warp(G["day"], M_day)
        P, p, mult, sat, lift = night_rows(t)
        if P > 0:
            lum = img @ np.array([0.2126, 0.7152, 0.0722], np.float32)
            img = lum[..., None] + (img - lum[..., None]) * sat
            img = img * mult + lift
            # deeper night: a touch more contrast as it gets dark
            gam = 1 + 0.22 * p[:, None, None]
            img = np.power(np.maximum(img, 0) / 0.6, gam) * 0.6
        # sunlight, dust, glossy glint (day only)
        day_amt = 1 - smooth((P - 0.15) / 0.55)
        if day_amt > 0.01:
            rays = god_rays(t, P)
            warm = np.array([1.0, 0.86, 0.66]) * (1 - P) + np.array([1.0, 0.55, 0.22]) * P
            ray_up = cv2.resize(rays, (W, H), interpolation=cv2.INTER_LINEAR)
            img += ray_up[..., None] * (0.16 * day_amt * (1 + 1.2 * P)) * warm.astype(np.float32)
            dd = G["dust"]
            for i in range(len(dd["x"])):
                x = (dd["x"][i] + dd["vx"][i] * t + 18 * math.sin(0.4 * t + dd["ph"][i])) % W
                y = (dd["y"][i] + dd["vy"][i] * t + 12 * math.sin(0.55 * t + 2 * dd["ph"][i])) % H
                lit = rays[min(int(y / LR), LH - 1), min(int(x / LR), LW - 1)]
                tw = 0.6 + 0.4 * math.sin(2.3 * t + dd["ph"][i] * 3)
                amp = dd["a"][i] * (0.25 + 1.6 * lit) * tw * day_amt * 0.55
                if dd["blur"][i]:
                    splat(img, x, y, dd["s"][i] * 4, warm, amp * 0.25, disc=True)
                else:
                    splat(img, x, y, dd["s"][i], warm, amp)
            if 1.9 < t < 3.6:
                m1s = warp(G["m1"], M_day, border=cv2.BORDER_CONSTANT)
                xs = np.arange(W, dtype=np.float32)[None, :]
                ys = np.arange(H, dtype=np.float32)[:, None]
                v = (xs * 0.8 + ys * 0.45) / W
                c = -0.4 + (t - 1.9) / 1.5 * 2.2
                band = np.exp(-((v - c) / 0.07) ** 2) + 0.5 * np.exp(-((v - c + 0.16) / 0.025) ** 2)
                img += (band * m1s * 0.35)[..., None] * np.array([1.0, 0.97, 0.92], np.float32)
        # warm light spills from the letter into the dark room
        if lit_ramp > 0:
            sp = warp(G["spill"], M_day, border=cv2.BORDER_CONSTANT)
            img += sp[..., None] * (lit_ramp * 0.07) * np.array([1.0, 0.50, 0.12], np.float32)
        base = img
    if t >= TL.FIRST_SPARK - 0.05:
        night = warp(G["night"], M_night)
        if morph_q < 1.0:
            m2s = warp(G["m2"], C @ W2 @ D, border=cv2.BORDER_CONSTANT)
            led_scr = apply_h(C @ W2, G["leds"][:, :2])
            canvas = np.zeros((LH, LW), np.float32)
            s_scr = BASE_S * z
            for i, (x, y) in enumerate(led_scr):
                dt = t - G["led_t"][i]
                if dt <= 0:
                    continue
                r = 72 * s_scr * ease_out(dt / 0.6) / LR
                cv2.circle(canvas, (int(x / LR), int(y / LR)), max(int(r), 1), 1.0, -1, lineType=cv2.LINE_AA)
            canvas = cv2.GaussianBlur(canvas, (0, 0), 7)
            R = cv2.resize(canvas, (W, H), interpolation=cv2.INTER_LINEAR)
            fill = win(t, TL.FILL_T0, TL.FILL_T1)
            R = np.maximum(R, fill) * m2s
            lit = night * (0.45 + 0.55 * fill)
            img = base * (1 - R[..., None]) + lit * R[..., None]
            img = img * (1 - morph_q) + night * morph_q
        else:
            img = night
    else:
        img = base

    # ------------------------------------------------ light effects (linear)
    if t >= TL.FIRST_SPARK - 0.05:
        Wn = C @ W2
        led_scr = apply_h(Wn, G["leds"][:, :2])
        s_scr = BASE_S * z
        night_fx = 1 - win(t, 18.8, 19.6) * 0.3
        for i, (x, y) in enumerate(led_scr):
            dt = t - G["led_t"][i]
            if dt < 0:
                continue
            pop = 2.6 * math.exp(-dt / 0.14) * (1 - math.exp(-dt / 0.02))
            steady = 0.55 + 0.3 * math.sin(6.28 * G["led_f"][i] * t + G["led_ph"][i])
            amp = (pop + steady * min(dt / 0.2, 1)) * night_fx
            splat(img, x, y, 2.2 * s_scr / 2.1, (1.0, 0.82, 0.55), amp * 0.9)
            star = pop * 0.5
            if G["led_star"][i]:
                star += max(0.0, math.sin(6.28 * G["led_f"][i] * 0.37 * t + G["led_ph"][i] * 2)) ** 6 * 0.55
            if star > 0.02:
                add_sprite(img, G["flare"], x, y, 0.55 + 0.25 * min(pop, 1), (1.0, 0.80, 0.50), star * night_fx)
        # first spark of the cascade
        dt = t - TL.FIRST_SPARK
        if 0 <= dt < 0.6:
            fx, fy = apply_h(Wn, np.array([[293.0, 789.0]]))[0]
            add_sprite(img, G["flare"], fx, fy, 0.9, (1.0, 0.85, 0.6), 1.6 * math.exp(-dt / 0.2) * (1 - math.exp(-dt / 0.02)))

    flash = math.exp(-((t - TL.FLASH) / 0.13) ** 2) if abs(t - TL.FLASH) < 0.6 else 0.0
    if flash > 0.001:
        img = img * (1 + 0.8 * flash) + flash * 0.04 * np.array([1.0, 0.75, 0.45], np.float32)

    # sparks burst
    tau = t - TL.FLASH + 0.04
    if 0 < tau < 1.6:
        sk = G["sparks"]
        kd = 2.6
        for i in range(len(sk["wx"])):
            if tau > sk["life"][i]:
                continue
            fade = (1 - tau / sk["life"][i]) ** 1.5
            wpt = apply_h(C, np.array([[sk["wx"][i], sk["wy"][i]]]))[0]
            for j, back in enumerate((0.0, 0.012, 0.024, 0.036)):
                tt = max(tau - back, 0)
                dx = sk["vx"][i] * (1 - math.exp(-kd * tt)) / kd
                dy = sk["vy"][i] * (1 - math.exp(-kd * tt)) / kd + 90 * tt * tt
                splat(img, wpt[0] + dx, wpt[1] + dy, sk["s"][i], (1.0, 0.78, 0.42), sk["a"][i] * fade * (1 - j * 0.22))

    # floating golden bokeh
    if t > TL.FLASH - 0.1:
        bk = G["bokeh"]
        tb = t - TL.FLASH
        for i in range(len(bk["x"])):
            e = smooth((tb - bk["t0"][i]) / 0.8)
            if e <= 0:
                continue
            y = (bk["y"][i] - bk["v"][i] * tb) % (H + 120) - 60
            x = bk["x"][i] + bk["sw"][i] * math.sin(6.28 * bk["sf"][i] * tb + bk["ph"][i])
            tw = 0.7 + 0.3 * math.sin(3.1 * tb + bk["ph"][i] * 2)
            r = bk["r"][i]
            col = (1.0, 0.62 + 0.18 * bk["hue"][i], 0.25 + 0.2 * bk["hue"][i])
            a = bk["a"][i] * e * tw * (0.05 + 0.32 / (1 + r / 6))
            if r > 6:
                splat(img, x, y, r, col, a * 0.55, disc=True)
            else:
                splat(img, x, y, r * 0.5, col, a * 2.2)

    # ------------------------------------------------ rays, bloom, grade
    small = cv2.resize(img, (LW, LH), interpolation=cv2.INTER_AREA)
    lum = small.max(2)
    night_amt = win(t, 6.0, 8.6)
    thr = 0.92 - 0.5 * night_amt
    bright = small * (np.clip((lum - thr) / 0.5, 0, 1) ** 1.5)[..., None]
    b1 = cv2.GaussianBlur(bright, (0, 0), 3)
    b2 = cv2.GaussianBlur(bright, (0, 0), 14)
    bloom = 0.10 + 0.35 * night_amt + 0.55 * flash
    acc = b1 * 0.45 + b2 * 0.55
    ray_amt = 0.85 * win(t, 9.25, TL.FLASH) * (1 - 0.82 * win(t, TL.FLASH, 11.2)) * (1 - win(t, 14.0, 15.0) * 0.6)
    if ray_amt > 0.01:
        cxs, cys = apply_h(C, LETTER_C[None])[0] / LR
        ra = np.zeros_like(bright)
        src = cv2.GaussianBlur(bright, (0, 0), 1.5)
        n = 14
        for k in range(n):
            sc = 1 + 0.55 * k / n
            M = np.float32([[sc, 0, cxs * (1 - sc)], [0, sc, cys * (1 - sc)]])
            ra += cv2.warpAffine(src, M, (LW, LH), flags=cv2.INTER_LINEAR) * (1 - k / n)
        acc = acc * bloom + ra * (ray_amt * 0.22)
    else:
        acc = acc * bloom
    img += cv2.resize(acc, (W, H), interpolation=cv2.INTER_LINEAR)

    # end-card darkening of the lower half
    endk = win(t, TL.END_T0 + 0.2, TL.END_T1 + 0.2)
    if endk > 0:
        yy = np.arange(H, dtype=np.float32)
        gk = (np.clip((yy - 0.43 * H) / (0.2 * H), 0, 1) ** 1.3 * 0.93 * endk)[:, None, None]
        img = img * (1 - gk) + np.array([0.004, 0.005, 0.011], np.float32) * gk

    vig = 0.22 + 0.35 * night_amt
    img *= np.clip(1 - vig * G["vig"], 0, 1)[..., None]
    # exposure fade in / out
    fade = ease_out(t / 0.9) * (1 - win(t, TL.FADE_OUT, DUR))
    img *= fade
    # soft highlight shoulder, then display gamma
    a = 0.78
    over_ = np.maximum(img - a, 0)
    img = np.minimum(img, a) + (1 - a) * (1 - np.exp(-over_ / (1 - a)))
    out = np.power(np.clip(img, 0, 1), 1 / 2.2)
    # focus pull at the start
    if t < 0.8:
        sg = 14 * (1 - ease_out(t / 0.8))
        if sg > 0.3:
            out = cv2.GaussianBlur(out, (0, 0), sg)

    draw_text(out, t)
    out *= 1 - win(t, TL.FADE_OUT, DUR)

    gr = G["grain"][fi % len(G["grain"])]
    out += gr[..., None] * (0.006 + 0.008 * night_amt)
    return (np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8)


# ----------------------------------------------------------------------------- main
def main():
    data, fonts, out = sys.argv[1:4]
    if "--stills" in sys.argv:
        ts = [float(x) for x in sys.argv[sys.argv.index("--stills") + 1].split(",")]
        init_worker(data, fonts)
        for t in ts:
            fr = render_frame(int(round(t * FPS)))
            Image.fromarray(fr).save(f"{out}/still_{t:05.2f}.png")
            print("still", t)
        return
    import subprocess
    n = int(round(DUR * FPS))
    ff = subprocess.Popen([
        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
        "-r", str(FPS), "-i", "-",
        "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
        "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-profile:v", "high",
        "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
        "-movflags", "+faststart", out], stdin=subprocess.PIPE)
    with Pool(4, initializer=init_worker, initargs=(data, fonts)) as pool:
        for i, fr in enumerate(pool.imap(render_frame, range(n), chunksize=2)):
            ff.stdin.write(fr.tobytes())
            if i % 30 == 0:
                print(f"frame {i}/{n}", flush=True)
    ff.stdin.close()
    ff.wait()


if __name__ == "__main__":
    main()
