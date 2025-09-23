
import cv2 as cv
import numpy as np

# -------- Tunables (kept as constants for clarity) --------
GREEN_HSV_LOW  = (30, 45, 35)    # green gate in HSV (slightly wider to catch shaded turf)
GREEN_HSV_HIGH = (102, 255, 255)
WHITE_HSV_LOW  = (0, 0, 175)     # white border (faint to bright)
WHITE_HSV_HIGH = (180, 85, 255)
BARRIER_KSIZE  = (3, 3)          # thickness of the 'moat' around white line
EXG_BANDS      = 30              # horizontal bands for Otsu per band
BOTTOM_CUT_R   = 0.97            # cut a bottom strip to avoid floor leak
CLAHE_CLIP     = 2.5             # V-channel equalization
CLAHE_TILE     = (8, 8)
FENCE_CLOSE_W  = 0.018           # width (ratio) of closing kernel to overcome vertical fence bars
FENCE_CLOSE_H  = 0.010           # height (ratio) of closing kernel
HOLE_FILL_R    = 0.002           # max ratio of holes inside the field mask to auto-fill

# -------- Small helpers --------
def kpercent(w: int, h: int, wr: float, hr: float, ellipse=False):
    """Structuring element with size in image percents."""
    kw = max(3, int(round(w * wr)) | 1)
    kh = max(3, int(round(h * hr)) | 1)
    shape = cv.MORPH_ELLIPSE if ellipse else cv.MORPH_RECT
    return cv.getStructuringElement(shape, (kw, kh))

def clahe_v(img_rgb):
    """Equalize illumination on V channel."""
    hsv = cv.cvtColor(img_rgb, cv.COLOR_RGB2HSV)
    clahe = cv.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
    hsv[..., 2] = clahe.apply(hsv[..., 2])
    return cv.cvtColor(hsv, cv.COLOR_HSV2RGB)

def green_mask(img_rgb: np.ndarray) -> np.ndarray:
    """Robust green: HSV gate ∩ ExG (per-band Otsu) ∩ Lab b/a gate; small open."""
    h, w, _ = img_rgb.shape
    hsv = cv.cvtColor(img_rgb, cv.COLOR_RGB2HSV)
    hsv_gate = cv.inRange(hsv, GREEN_HSV_LOW, GREEN_HSV_HIGH)

    rgb = img_rgb.astype(np.float32) / 255.0
    exg = 2 * rgb[..., 1] - rgb[..., 0] - rgb[..., 2]
    exg = cv.GaussianBlur(exg, (0, 0), 1.0)
    exg_u8 = cv.normalize(exg, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

    # Otsu per horizontal band to adapt to illumination gradient
    mask_exg = np.zeros((h, w), np.uint8)
    y_edges = np.linspace(0, h, EXG_BANDS + 1, dtype=int)
    x0, x1 = int(0.10 * w), int(0.90 * w)  # ignore extreme sides for thresholding
    for y0, y1 in zip(y_edges[:-1], y_edges[1:]):
        band = exg_u8[y0:y1, x0:x1]
        if band.size == 0:
            continue
        thr, _ = cv.threshold(band, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        mask_exg[y0:y1, :] = (exg_u8[y0:y1, :] > thr).astype(np.uint8) * 255

    lab = cv.cvtColor(img_rgb, cv.COLOR_RGB2Lab)
    lab_gate = cv.inRange(lab, (0, 88, 118), (255, 140, 182))

    base = cv.bitwise_and(hsv_gate, mask_exg)
    base = cv.bitwise_and(base, lab_gate)
    base = cv.morphologyEx(base, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)), 1)
    return base

def fence_transparency(mask_u8: np.ndarray, w: int, h: int) -> np.ndarray:
    """Fill the vertical grid and gate gaps so the distant field stays connected."""
    if not np.any(mask_u8):
        return mask_u8

    hor = cv.getStructuringElement(
        cv.MORPH_RECT,
        (max(3, int(round(w * FENCE_CLOSE_W)) | 1), max(3, int(round(h * FENCE_CLOSE_H)) | 1)),
    )
    filled = cv.morphologyEx(mask_u8, cv.MORPH_CLOSE, hor, 1)
    soft = cv.morphologyEx(filled, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)), 1)

    inv = cv.bitwise_not(soft)
    num, labels, stats, _ = cv.connectedComponentsWithStats(inv, 8)
    if num > 1:
        max_hole = int(round(w * h * HOLE_FILL_R))
        for lbl in range(1, num):
            if stats[lbl, cv.CC_STAT_AREA] <= max_hole:
                soft[labels == lbl] = 255
    return soft

def white_barrier(img_rgb: np.ndarray, green_mask_u8: np.ndarray) -> np.ndarray:
    """Detect white line that touches non-green region → dilate to 'moat' mask."""
    hsv = cv.cvtColor(img_rgb, cv.COLOR_RGB2HSV)
    white = cv.inRange(hsv, WHITE_HSV_LOW, WHITE_HSV_HIGH)
    not_green = cv.bitwise_not((green_mask_u8 > 0).astype(np.uint8) * 255)
    touch = cv.dilate(not_green, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)), 1)
    border = cv.bitwise_and(white, touch)
    return cv.dilate(border, cv.getStructuringElement(cv.MORPH_ELLIPSE, BARRIER_KSIZE), 1)

def center_connected(mask_u8: np.ndarray, ker, h: int, w: int) -> np.ndarray:
    """Close small gaps, then keep the region reachable from image center."""
    m = cv.morphologyEx(mask_u8, cv.MORPH_CLOSE, ker, 2)
    cy0, cy1 = int(0.45*h), int(0.60*h)
    cx0, cx1 = int(0.30*w), int(0.70*w)
    seed = ((cx0+cx1)//2, (cy0+cy1)//2)

    tmp = m.copy()
    if tmp[seed[1], seed[0]] == 0:
        ys, xs = np.where(tmp > 0)
        if ys.size:
            i = np.argmin((xs - seed[0])**2 + (ys - seed[1])**2)
            seed = (int(xs[i]), int(ys[i]))

    ffm = np.zeros((h+2, w+2), np.uint8)
    cv.floodFill(tmp, ffm, seedPoint=seed, newVal=255)
    conn = cv.bitwise_and(tmp, m)

    num, labels, stats, _ = cv.connectedComponentsWithStats(conn, 8)
    if num > 1:
        largest = 1 + np.argmax(stats[1:, cv.CC_STAT_AREA])
        conn = (labels == largest).astype(np.uint8) * 255
    return conn

# -------- Main pipeline --------
def fb_field_detector(filename: str) -> None:
    img_bgr = cv.imread(filename)
    assert img_bgr is not None, "File not found."
    img = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    h, w, _ = img.shape

    img_eq = clahe_v(img)
    base   = green_mask(img_eq)
    base   = fence_transparency(base, w, h)
    moat   = white_barrier(img_eq, base)
    base   = cv.bitwise_and(base, cv.bitwise_not(moat))

    # Two scales of connectivity: conservative (hull) and wide (support)
    k_narrow = kpercent(w, h, 1/32, 5/h, ellipse=False)
    k_wide   = kpercent(w, h, 1/16, 5/h, ellipse=False)
    conn_n   = center_connected(base, k_narrow, h, w)
    conn_w   = center_connected(base, k_wide,   h, w)

    # Trim very bottom rows (floor/steps)
    cut = int(BOTTOM_CUT_R * h)
    conn_n[cut:, :] = 0
    conn_w[cut:, :] = 0

    # Hull from conservative mask
    cnts, _ = cv.findContours(conn_n, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    assert cnts, "Field contour not found"
    cnt  = max(cnts, key=cv.contourArea)
    hull = cv.convexHull(cnt)

    # Intersect with wide support for smoother lower edge under the fence
    hull_mask = np.zeros_like(conn_n); cv.fillPoly(hull_mask, [hull], 255)
    support = cv.dilate(conn_w, kpercent(w, h, 0.008, 0.008, ellipse=True), 1)
    color_sup = cv.morphologyEx(base, cv.MORPH_CLOSE, kpercent(w, h, 0.012, 0.012, ellipse=True), 1)
    inter   = cv.bitwise_and(hull_mask, support)
    inter   = cv.bitwise_and(inter, color_sup)

    # Final hull
    cnts, _ = cv.findContours(inter, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not cnts:
        cnts, _ = cv.findContours(hull_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnt  = max(cnts, key=cv.contourArea)
    final = cv.convexHull(cnt)

    # Draw
    overlay = img.copy()
    cv.polylines(overlay, [final], True, (255, 0, 0), max(2, w//800))
    cv.imwrite(f"processed_{filename}", cv.cvtColor(overlay, cv.COLOR_RGB2BGR))

if __name__ == "__main__":
    fb_field_detector("panorama_1.jpg")
    fb_field_detector("panorama_frame_2.jpg")
    fb_field_detector("panorama_frame_3.jpg")
