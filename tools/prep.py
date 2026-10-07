"""Prepare 2x working images, letter masks, LED points and alignment.

World coordinates = pixel coordinates of the original day photo (896x1094).
Outputs go to build/ (npz with everything render.py needs).
"""
import sys
import numpy as np
import cv2
from PIL import Image

HD, OUT = sys.argv[1], sys.argv[2]
K = 2  # working resolution = 2x original

# Letter outlines, original pixel coordinates (traced by hand)
A1_OUTER = [(312, 385), (335, 362), (580, 367), (597, 377), (672, 790), (667, 895), (475, 897),
            (480, 778), (425, 775), (415, 887), (227, 882), (220, 810)]
A1_HOLE = [(445, 555), (458, 557), (492, 662), (400, 662)]
A2_OUTER = [(345, 305), (352, 298), (572, 298), (580, 307), (672, 737), (667, 797), (485, 800),
            (485, 693), (427, 690), (427, 797), (245, 797), (245, 730)]
A2_HOLE = [(450, 480), (462, 480), (490, 588), (424, 588)]

# Matching points night->day used for alignment
P_DAY = np.float32([(312, 385), (597, 380), (227, 882), (667, 895), (448, 556), (425, 776), (480, 778)])
P_NIGHT = np.float32([(346, 305), (580, 307), (245, 797), (667, 797), (456, 480), (427, 690), (485, 693)])


def to_lin(a):
    return np.power(np.clip(a, 0, 1), 2.2).astype(np.float32)


def load_2x(name, size):
    x4 = np.asarray(Image.open(f"{HD}/{name}_x4.png").convert("RGB"), np.float32) / 255
    w, h = size
    sr = cv2.resize(x4, (w * K, h * K), interpolation=cv2.INTER_AREA)
    src = np.asarray(Image.open(f"{HD}/{name}_src.png").convert("RGB"), np.float32) / 255
    lz = cv2.resize(src, (w * K, h * K), interpolation=cv2.INTER_LANCZOS4)
    # mostly super-resolved, a touch of the original keeps natural texture
    return np.clip(0.82 * sr + 0.18 * lz, 0, 1)


def poly_mask(outer, hole, shape, feather):
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.int32(np.array(outer) * K)], 255)
    cv2.fillPoly(m, [np.int32(np.array(hole) * K)], 0)
    m = m.astype(np.float32) / 255
    return cv2.GaussianBlur(m, (0, 0), feather)


day = load_2x("dia", (896, 1094))
night = load_2x("noche", (891, 1099))

# remove the cut-off "ARTESANIA EN RES..." text at the bottom right of the day photo
wm = np.zeros(day.shape[:2], np.uint8)
cv2.line(wm, (592 * K, 1038 * K), (830 * K, 1102 * K), 255, 30 * K)
wm = cv2.dilate(wm, np.ones((5, 5), np.uint8))
d8 = (day * 255 + 0.5).astype(np.uint8)
d8 = cv2.inpaint(d8, wm, 6, cv2.INPAINT_TELEA)
# keep some marble texture: blend inpaint with blurred noise-free version
day = d8.astype(np.float32) / 255

m1 = poly_mask(A1_OUTER, A1_HOLE, day.shape[:2], 2.0)
m2 = poly_mask(A2_OUTER, A2_HOLE, night.shape[:2], 2.0)

# Homography night->world (letters coincide) and a natural upright scale+shift
H, _ = cv2.findHomography(P_NIGHT, P_DAY, 0)
A = np.zeros((len(P_NIGHT) * 2, 3))
b = np.zeros(len(P_NIGHT) * 2)
for i, ((x2, y2), (x1, y1)) in enumerate(zip(P_NIGHT, P_DAY)):
    A[2 * i] = (x2, 1, 0); b[2 * i] = x1
    A[2 * i + 1] = (y2, 0, 1); b[2 * i + 1] = y1
s, tx, ty = np.linalg.lstsq(A, b, rcond=None)[0]
S = np.array([[s, 0, tx], [0, s, ty], [0, 0, 1]])

# LED points in the night photo
n0 = np.asarray(Image.open(f"{HD}/noche_src.png").convert("RGB"), np.float32) / 255
lm = np.zeros(n0.shape[:2], np.uint8)
cv2.fillPoly(lm, [np.int32(A2_OUTER)], 1)
cv2.fillPoly(lm, [np.int32(A2_HOLE)], 0)
lm = cv2.erode(lm, np.ones((7, 7), np.uint8))
L = n0.mean(2)
score = (L - cv2.GaussianBlur(L, (0, 0), 6)) * (n0.min(2) > 0.62) * lm
mx = cv2.dilate(score, np.ones((15, 15), np.uint8))
ys, xs = np.where((score == mx) & (score > 0.07))
leds = np.array([(x, y, score[y, x]) for y, x in zip(ys, xs)], np.float32)

# Extended night canvas for wide end-card framing: reflect, blur and darken outside
PAD = 1000
ext = cv2.copyMakeBorder(night, PAD, PAD, PAD, PAD, cv2.BORDER_REFLECT_101)
blur = cv2.GaussianBlur(cv2.resize(ext, None, fx=0.125, fy=0.125, interpolation=cv2.INTER_AREA), (0, 0), 6)
blur = cv2.resize(blur, (ext.shape[1], ext.shape[0]), interpolation=cv2.INTER_LINEAR)
inside = np.zeros(ext.shape[:2], np.uint8)
inside[PAD:-PAD, PAD:-PAD] = 1
dist = cv2.distanceTransform(1 - inside, cv2.DIST_L2, 5)
w = np.clip(dist / 120.0, 0, 1)[..., None]
dark = np.exp(-dist / 500.0)[..., None] * 0.55
navy = np.array([0.035, 0.045, 0.075], np.float32)
ext = ext * (1 - w) + (blur * dark + navy * (1 - dark) * 0.6) * w

np.savez(OUT,
         day=to_lin(day), night_ext=to_lin(ext), ext_pad=PAD,
         m1=m1, m2=m2, H=H, S=S, leds=leds, K=K)
print("H", H, "\nS", S, "\nleds", len(leds))
