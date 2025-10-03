import cv2 as cv
import numpy as np
from pathlib import Path

# ==============================
#   ГЛОБАЛЬНЫЕ НАСТРОЙКИ
# ==============================

DEBUG = True  # Если False — промежуточные шаги не будут сохраняться

GREEN_HSV_LOW  = (35, 55, 35)
GREEN_HSV_HIGH = (95, 255, 255)

WHITE_HSV_LOW  = (0, 0, 175)
WHITE_HSV_HIGH = (180, 85, 255)

BARRIER_KSIZE  = (5, 5)
EXG_BANDS      = 30
BOTTOM_CUT_R   = 0.93

CROP_LEFT_FRAC  = 0.025
CROP_RIGHT_FRAC = 0.025


CLAHE_CLIP     = 2.5
CLAHE_TILE     = (8, 8)




# ==============================
#   ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==============================

def _to_bgr(img):
    """Гарантировать BGR для cv.imwrite"""
    if img.ndim == 2:  # маска
        return cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    return img

def _overlay_mask(img_rgb, mask_u8, color=(0, 255, 0), alpha=0.5):
    """Наложить полупрозрачную маску на RGB-изображение."""
    vis = img_rgb.copy()
    if mask_u8.ndim == 3:
        mask_u8 = cv.cvtColor(mask_u8, cv.COLOR_BGR2GRAY)
    m = (mask_u8 > 0).astype(np.uint8) * 255
    color_img = np.zeros_like(vis)
    color_img[:] = color
    color_img = cv.bitwise_and(color_img, color_img, mask=m)
    return cv.addWeighted(vis, 1.0, color_img, alpha, 0.0)

class Debugger:
    """Сохраняет шаги в debug/<stem>/NN_name.png"""
    def __init__(self, stem: str, base_rgb: np.ndarray):
        self.dir = Path("debug") / stem
        self.dir.mkdir(parents=True, exist_ok=True)
        self.step = 0
        self.base = base_rgb

    def save(self, name: str, img, *, overlay=False, rgb=False):
        if not DEBUG:
            return
        if overlay:
            vis = _overlay_mask(self.base, img)
            bgr = cv.cvtColor(vis, cv.COLOR_RGB2BGR)
        else:
            if rgb and img.ndim == 3:
                bgr = cv.cvtColor(img, cv.COLOR_RGB2BGR)
            else:
                bgr = _to_bgr(img)
        cv.imwrite(str(self.dir / f"{self.step:02d}_{name}.png"), bgr)
        self.step += 1


# ==============================
#   ПРЕОБРАЗОВАНИЯ
# ==============================

def kpercent(w: int, h: int, wr: float, hr: float, ellipse=False):
    """Создаёт морфологическое ядро, размер которого зависит от % изображения."""
    kw = max(3, int(round(w * wr)) | 1)
    kh = max(3, int(round(h * hr)) | 1)
    shape = cv.MORPH_ELLIPSE if ellipse else cv.MORPH_RECT
    return cv.getStructuringElement(shape, (kw, kh))

def clahe_v(img_rgb):
    """Выравниваем освещение по каналу яркости V."""
    hsv = cv.cvtColor(img_rgb, cv.COLOR_RGB2HSV)
    clahe = cv.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
    hsv[..., 2] = clahe.apply(hsv[..., 2])
    return cv.cvtColor(hsv, cv.COLOR_HSV2RGB)


# ==============================
#   МАСКА ЗЕЛЕНОГО
# ==============================

def green_mask(img_rgb: np.ndarray, dbg: Debugger | None = None) -> np.ndarray:
    """Объединение HSV-гейта, ExG с Otsu и Lab-фильтра."""
    h, w, _ = img_rgb.shape

    # HSV фильтр
    hsv = cv.cvtColor(img_rgb, cv.COLOR_RGB2HSV)
    hsv_gate = cv.inRange(hsv, GREEN_HSV_LOW, GREEN_HSV_HIGH)
    if dbg:
        dbg.save("10_hsv_gate_mask", hsv_gate)
        dbg.save("11_hsv_gate_overlay", hsv_gate, overlay=True)

    # Excess Green
    rgb = img_rgb.astype(np.float32) / 255.0
    exg = 2 * rgb[..., 1] - rgb[..., 0] - rgb[..., 2]
    exg = cv.GaussianBlur(exg, (0, 0), 1.0)
    exg_u8 = cv.normalize(exg, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    if dbg:
        dbg.save("12_exg_u8", exg_u8)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    exg_u8_eq = clahe.apply(exg_u8)
    if dbg:
        dbg.save("11_exg_u8_eq", exg_u8_eq)

    blurred = cv.GaussianBlur(exg_u8_eq, (0, 0), sigmaX=3)
    highpass = cv.addWeighted(exg_u8_eq, 1.5, blurred, -0.5, 0)
    if dbg:
        dbg.save("12_exg_highpass", highpass)

    # вычисляем порог Отсу в крупных полосах
    bands = max(10, min(EXG_BANDS, h // 20))
    y_edges = np.linspace(0, h, bands + 1, dtype=int)
    x0, x1 = int(0.10 * w), int(0.90 * w)

    centers, thr_vals = [], []
    for y0, y1 in zip(y_edges[:-1], y_edges[1:]):
        band = exg_u8[y0:y1, x0:x1]
        if band.size < 100:
            continue
        thr, _ = cv.threshold(band, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        centers.append((y0 + y1) * 0.5)
        thr_vals.append(float(thr))

    centers = np.array(centers, dtype=np.float32)
    thr_vals = np.array(thr_vals, dtype=np.float32)

    # сгладим пороги вдоль Y (без SciPy: свёртка простым окном)
    win = max(3, (len(thr_vals) // 7) | 1)
    kernel = np.ones(win, dtype=np.float32) / win
    thr_smooth = np.convolve(thr_vals, kernel, mode="same")

    # интерполируем порог для каждой строки
    ys = np.arange(h, dtype=np.float32)
    thr_y = np.interp(ys, centers, thr_smooth, left=thr_smooth[0], right=thr_smooth[-1])

    # применяем построчно: пиксель > локального порога?
    mask_exg = (exg_u8 > thr_y[:, None]).astype(np.uint8) * 255

    # тонкое вертикальное закрытие швов
    k_vert = max(3, (int(h * 0.015) | 1))
    mask_exg = cv.morphologyEx(mask_exg, cv.MORPH_CLOSE,
                               cv.getStructuringElement(cv.MORPH_RECT, (1, k_vert)), 1)

    # ломаем узкие горизонтальные мостики
    # break_kernel = cv.getStructuringElement(cv.MORPH_RECT, (21, 3))
    # mask_exg = cv.morphologyEx(mask_exg, cv.MORPH_OPEN, break_kernel, iterations=1)

    # --- коричневая/земляная полоса ---
    hsv = cv.cvtColor(img_rgb, cv.COLOR_RGB2HSV)
    # Hue ~10–25 (оранж/коричневый), Saturation высокое, Value среднее
    brown = cv.inRange(hsv, (5, 50, 50), (25, 255, 200))
    brown = cv.morphologyEx(brown, cv.MORPH_CLOSE,
                            cv.getStructuringElement(cv.MORPH_RECT, (15, 5)), 2)
    if dbg:
        dbg.save("13a_brown_mask", brown)
        dbg.save("13b_brown_mask_overlay", brown, overlay=True)

    # вычитаем из зелёного
    mask_exg = cv.bitwise_and(mask_exg, cv.bitwise_not(brown))
    if dbg:
        dbg.save("13c_exg_minus_brown", mask_exg)
        dbg.save("13d_exg_minus_brown_overlay", mask_exg, overlay=True)

    if dbg:
        dbg.save("13_exg_mask_smooth", mask_exg)
        dbg.save("14_exg_mask_smooth_overlay", mask_exg, overlay=True)

    # Lab
    lab = cv.cvtColor(img_rgb, cv.COLOR_RGB2Lab)
    lab_gate = cv.inRange(lab, (0, 0, 120), (255, 118, 165))
    if dbg:
        dbg.save("15_lab_gate_mask", lab_gate)
        dbg.save("16_lab_gate_overlay", lab_gate, overlay=True)

    # Пересечение трёх масок
    base = cv.bitwise_and(hsv_gate, mask_exg)
    base = cv.bitwise_and(base, lab_gate)
    if dbg:
        dbg.save("17_base_raw", base)
        dbg.save("18_base_raw_overlay", base, overlay=True)

    # Открытие для удаления шумов
    base = cv.morphologyEx(
        base, cv.MORPH_OPEN,
        cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)), 1
    )
    if dbg:
        dbg.save("19_base_opened", base)
        dbg.save("20_base_opened_overlay", base, overlay=True)

    return base


# ==============================
#   РОВ ВОКРУГ БЕЛЫХ ЛИНИЙ
# ==============================

def white_barrier(img_rgb: np.ndarray, green_mask_u8: np.ndarray,
                  dbg: Debugger | None = None) -> np.ndarray:
    hsv = cv.cvtColor(img_rgb, cv.COLOR_RGB2HSV)
    white = cv.inRange(hsv, WHITE_HSV_LOW, WHITE_HSV_HIGH)
    if dbg:
        dbg.save("30_white_mask", white)
        dbg.save("31_white_overlay", white, overlay=True)

    not_green = cv.bitwise_not((green_mask_u8 > 0).astype(np.uint8) * 255)
    touch = cv.dilate(not_green,
                      cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)), 1)
    if dbg:
        dbg.save("32_not_green", not_green)
        dbg.save("33_touch_dilated", touch)

    border = cv.bitwise_and(white, touch)
    moat = cv.dilate(border,
                     cv.getStructuringElement(cv.MORPH_ELLIPSE, BARRIER_KSIZE),
                     1)


    if dbg:
        dbg.save("34_border_white∩not_green", border)
        dbg.save("35_moat_dilated", moat)
        dbg.save("36_moat_overlay", moat, overlay=True)
    return moat


# ==============================
#   СВЯЗНАЯ ОБЛАСТЬ ОТ ЦЕНТРА
# ==============================

def center_connected(mask_u8: np.ndarray, ker, h: int, w: int,
                     dbg: Debugger | None = None, tag: str = "") -> np.ndarray:
    m = cv.morphologyEx(mask_u8, cv.MORPH_CLOSE, ker, 2)
    if dbg:
        dbg.save(f"40_close_{tag}", m)

    # центр
    cy0, cy1 = int(0.45*h), int(0.60*h)
    cx0, cx1 = int(0.30*w), int(0.70*w)
    seed = ((cx0+cx1)//2, (cy0+cy1)//2)

    tmp = m.copy()
    if tmp[seed[1], seed[0]] == 0:
        ys, xs = np.where(tmp > 0)
        if ys.size:
            i = np.argmin((xs - seed[0])**2 + (ys - seed[1])**2)
            seed = (int(xs[i]), int(ys[i]))

    if dbg:
        seed_vis = cv.cvtColor(tmp, cv.COLOR_GRAY2BGR)
        cv.circle(seed_vis, seed, 4, (0, 0, 255), -1)
        dbg.save(f"41_seed_{tag}", seed_vis)

    ffm = np.zeros((h+2, w+2), np.uint8)
    cv.floodFill(tmp, ffm, seedPoint=seed, newVal=255)
    if dbg:
        dbg.save(f"42_flood_{tag}", tmp)

    conn = cv.bitwise_and(tmp, m)
    if dbg:
        dbg.save(f"43_conn_{tag}", conn)

    # оставить только крупнейшую компоненту
    num, labels, stats, _ = cv.connectedComponentsWithStats(conn, 8)
    if num > 1:
        largest = 1 + np.argmax(stats[1:, cv.CC_STAT_AREA])
        conn = (labels == largest).astype(np.uint8) * 255
    if dbg:
        dbg.save(f"44_conn_largest_{tag}", conn)
    return conn

# ==============================
#   ГЛАВНЫЙ ПАЙПЛАЙН
# ==============================

def fb_field_detector(filename: str) -> None:
    img_bgr = cv.imread(filename)
    assert img_bgr is not None, "File not found."
    img = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    h, w, _ = img.shape

    # Обрезка по краям
    x0 = int(w * CROP_LEFT_FRAC)
    x1 = int(w * (1.0 - CROP_RIGHT_FRAC))
    img = img[:, x0:x1, :]
    w = x1 - x0

    # Создаём отладчик
    stem = Path(filename).stem
    dbg = Debugger(stem, img)
    dbg.save("00_input_rgb", img, rgb=True)


    # Выравнивание освещения
    img_eq = clahe_v(img)
    dbg.save("01_img_eq_after_CLAHE", img_eq, rgb=True)

    # Базовая маска поля
    base = green_mask(img_eq, dbg)
    moat = white_barrier(img_eq, base, dbg)
    base2 = cv.bitwise_and(base, cv.bitwise_not(moat))

    edge_mask = np.zeros_like(base2)
    edge_thickness = int(0.04 * w)  # % ширины по бокам
    cv.rectangle(edge_mask, (0, 0), (edge_thickness, h), 255, -1)
    cv.rectangle(edge_mask, (w - edge_thickness, 0), (w, h), 255, -1)

    # вычитаем из маски поля
    base2 = cv.bitwise_and(base2, cv.bitwise_not(edge_mask))

    dbg.save("37_base_minus_moat", base2)
    dbg.save("38_base_minus_moat_overlay", base2, overlay=True)

    # Две связности
    k_narrow = kpercent(w, h, 1/20, 5/h, ellipse=False)
    k_wide   = kpercent(w, h, 1/16, 5/h, ellipse=False)
    conn_n   = center_connected(base2, k_narrow, h, w, dbg, tag="narrow")
    conn_w   = center_connected(base2, k_wide,   h, w, dbg, tag="wide")

    # Обрезаем низ
    cut = int(BOTTOM_CUT_R * h)
    conn_n[cut:, :] = 0
    conn_w[cut:, :] = 0
    dbg.save("45_conn_n_cut", conn_n)
    dbg.save("46_conn_w_cut", conn_w)

    # Hull
    cnts, _ = cv.findContours(conn_n, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    assert cnts, "Field contour not found"
    cnt = max(cnts, key=cv.contourArea)
    hull = cv.convexHull(cnt)
    hull_mask = np.zeros_like(conn_n)
    cv.fillPoly(hull_mask, [hull], 255)
    dbg.save("47_hull_mask", hull_mask)
    dbg.save("48_hull_overlay", hull_mask, overlay=True)

    # Wide support
    support = cv.dilate(conn_w, kpercent(w, h, 0.008, 0.008, ellipse=True), 1)
    dbg.save("49_support_dilated", support)

    inter = cv.bitwise_and(hull_mask, support)
    dbg.save("50_intersection_hull×support", inter)
    dbg.save("51_intersection_overlay", inter, overlay=True)

    # Финальный контур
    cnts, _ = cv.findContours(inter, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnt = max(cnts, key=cv.contourArea)
    final = cv.convexHull(cnt)

    overlay = img.copy()
    cv.polylines(overlay, [final], True, (255, 0, 0), max(2, w//800))
    dbg.save("99_final_overlay", overlay, rgb=True)

    cv.imwrite(f"processed_{filename}", cv.cvtColor(overlay, cv.COLOR_RGB2BGR))


# ==============================
#   ЗАПУСК
# ==============================

if __name__ == "__main__":
    #fb_field_detector("panorama_1.jpg")
    #fb_field_detector("panorama_frame_2.jpg")
    #fb_field_detector("panorama_frame_3.jpg")
    #fb_field_detector("panorama_frame_50.jpg")
    fb_field_detector("panorama_frame_51.jpg")