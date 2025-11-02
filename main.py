import cv2 as cv
import numpy as np
from pathlib import Path

# ==============================
#   НАСТРОЙКИ
# ==============================

DEBUG = True
SAVE_RESULTS = True


# ==============================
#   ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==============================

class Debugger:
    """Класс для отладочного вывода"""

    def __init__(self, stem: str, base_rgb: np.ndarray):
        self.dir = Path("debug") / stem
        self.dir.mkdir(parents=True, exist_ok=True)
        self.step = 0
        self.base = base_rgb

    def save(self, name: str, img, overlay=False):
        if not DEBUG:
            return

        if overlay:
            vis = self.base.copy()
            if img.ndim == 2:
                mask = (img > 0).astype(np.uint8) * 255
                color_mask = np.zeros_like(vis)
                color_mask[:, :, 1] = mask  # Зелёный канал для маски
                vis = cv.addWeighted(vis, 0.7, color_mask, 0.3, 0)
            bgr = cv.cvtColor(vis, cv.COLOR_RGB2BGR)
        else:
            if img.ndim == 3:
                bgr = cv.cvtColor(img, cv.COLOR_RGB2BGR)
            else:
                bgr = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

        cv.imwrite(str(self.dir / f"{self.step:02d}_{name}.png"), bgr)
        self.step += 1


# ==============================
#   ОБРАБОТКА EXG
# ==============================

def compute_exg(img_rgb):
    """Вычисление Excess Green Index"""
    # Нормализация в диапазон [0, 1]
    rgb_norm = img_rgb.astype(np.float32) / 255.0
    r, g, b = rgb_norm[:, :, 0], rgb_norm[:, :, 1], rgb_norm[:, :, 2]

    # ExG = 2*G - R - B
    exg = 2.0 * g - r - b

    # Нормализация в диапазон [0, 255]
    # ExG теоретически в диапазоне [-2, 2], но практически [-1, 1]
    exg_norm = (exg + 1.0) * 127.5
    exg_u8 = np.clip(exg_norm, 0, 255).astype(np.uint8)

    return exg_u8


def process_exg_otsu(exg_u8, dbg=None):
    """Обработка ExG с использованием Otsu и адаптацией для разных условий освещения"""
    h, w = exg_u8.shape

    # Предварительная фильтрация шума
    exg_filtered = cv.GaussianBlur(exg_u8, (5, 5), 1.5)

    if dbg:
        dbg.save("10_exg_raw", exg_u8)
        dbg.save("11_exg_filtered", exg_filtered)

    # Глобальный Otsu для начальной маски
    otsu_val, mask_otsu = cv.threshold(exg_filtered, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    if dbg:
        dbg.save("12_otsu_global", mask_otsu)
        dbg.save("13_otsu_overlay", mask_otsu, overlay=True)

    # Проверяем, достаточно ли зелёных пикселей найдено
    green_pixels = cv.countNonZero(mask_otsu)
    total_pixels = h * w
    green_ratio = green_pixels / total_pixels

    # Если зелёных пикселей слишком мало (< 10%), снижаем порог
    if green_ratio < 0.1:
        # Пробуем более низкий порог
        adaptive_threshold = otsu_val * 0.7
        mask_refined = (exg_filtered > adaptive_threshold).astype(np.uint8) * 255
    elif green_ratio > 0.7:
        # Если слишком много зелёного (> 70%), повышаем порог
        adaptive_threshold = otsu_val * 1.1
        mask_refined = (exg_filtered > adaptive_threshold).astype(np.uint8) * 255
    else:
        # Нормальный случай: адаптивное уточнение
        green_pixels_mask = mask_otsu > 0
        if np.any(green_pixels_mask):
            mean_green = np.mean(exg_filtered[green_pixels_mask])
            std_green = np.std(exg_filtered[green_pixels_mask])

            # Новый порог: немного ниже среднего зелёного
            adaptive_threshold = mean_green - 0.5 * std_green
            adaptive_threshold = max(adaptive_threshold, otsu_val * 0.8)  # Не ниже 80% от Otsu
        else:
            adaptive_threshold = otsu_val * 0.8

        # Применяем уточнённый порог
        mask_refined = (exg_filtered > adaptive_threshold).astype(np.uint8) * 255

    if dbg:
        dbg.save("14_refined_mask", mask_refined)
        dbg.save("15_refined_overlay", mask_refined, overlay=True)

    return mask_refined


def clean_mask(mask, dbg=None):
    """Очистка маски от шума и артефактов"""
    h, w = mask.shape

    # 1. Удаление мелких шумов
    kernel_small = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    cleaned = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel_small)

    # 2. Заполнение мелких дыр
    kernel_medium = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
    closed = cv.morphologyEx(cleaned, cv.MORPH_CLOSE, kernel_medium)

    if dbg:
        dbg.save("20_cleaned", closed)

    # 3. Удаление областей, касающихся краёв изображения (кроме небольших касаний)
    # Создаём маску краёв
    edge_mask = np.zeros((h, w), dtype=np.uint8)
    edge_thickness = 2
    edge_mask[:edge_thickness, :] = 255  # верх
    edge_mask[-edge_thickness:, :] = 255  # низ
    edge_mask[:, :edge_thickness] = 255  # лево
    edge_mask[:, -edge_thickness:] = 255  # право

    # Находим компоненты, касающиеся краёв
    edge_touching = cv.bitwise_and(closed, edge_mask)

    # Удаляем только маленькие краевые артефакты
    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(edge_touching, 8)
    for i in range(1, num_labels):
        if stats[i, cv.CC_STAT_AREA] < (h * w * 0.01):  # Меньше 1% изображения
            closed[labels == i] = 0

    if dbg:
        dbg.save("21_edge_cleaned", closed)
        dbg.save("22_cleaned_overlay", closed, overlay=True)

    return closed


def find_field_components(mask, dbg=None):
    """Поиск всех компонент поля, включая фрагменты за сеткой забора"""
    h, w = mask.shape

    # Находим все компоненты
    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(mask, 8)

    if num_labels <= 1:
        return mask

    # Собираем все значимые компоненты
    components = []
    for i in range(1, num_labels):
        area = stats[i, cv.CC_STAT_AREA]
        x = stats[i, cv.CC_STAT_LEFT]
        y = stats[i, cv.CC_STAT_TOP]
        width = stats[i, cv.CC_STAT_WIDTH]
        height = stats[i, cv.CC_STAT_HEIGHT]
        cx, cy = centroids[i]

        # Пропускаем только очень маленькие шумовые области
        if area < (h * w * 0.001):  # Минимум 0.1% от изображения
            continue

        components.append({
            'label': i,
            'area': area,
            'x': x, 'y': y,
            'width': width, 'height': height,
            'cx': cx, 'cy': cy,
            'bbox': (x, y, x + width, y + height)
        })

    if not components:
        return mask

    # Анализируем геометрическое расположение компонент
    # Поле обычно занимает центральную горизонтальную полосу изображения

    # Находим основную горизонтальную полосу где находится поле
    y_centers = [c['cy'] for c in components]
    y_median = np.median(y_centers)

    # Определяем вертикальные границы поля (примерно)
    field_y_min = y_median - h * 0.4
    field_y_max = y_median + h * 0.4

    # Создаём маску результата
    result_mask = np.zeros_like(mask)

    # Добавляем все компоненты, которые:
    # 1. Находятся в центральной горизонтальной полосе
    # 2. Имеют достаточную площадь
    # 3. Имеют разумное соотношение сторон (не слишком узкие вертикальные полосы)

    for comp in components:
        cy = comp['cy']
        aspect_ratio = comp['width'] / comp['height'] if comp['height'] > 0 else 0

        # Условия включения компоненты:
        # - Центр в пределах поля по вертикали
        # - Не слишком узкая вертикальная полоса
        # - Достаточная площадь

        include = False

        # Основное условие: компонента в центральной части
        if field_y_min <= cy <= field_y_max:
            # Проверяем, что это не узкий вертикальный артефакт
            if aspect_ratio > 0.2 or comp['area'] > (h * w * 0.01):
                include = True

        # Дополнительное условие: большие компоненты включаем всегда
        if comp['area'] > (h * w * 0.05):  # Больше 5% изображения
            include = True

        if include:
            comp_mask = (labels == comp['label']).astype(np.uint8) * 255
            result_mask = cv.bitwise_or(result_mask, comp_mask)

    # Морфологическое закрытие для соединения близких фрагментов
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 5))
    result_mask = cv.morphologyEx(result_mask, cv.MORPH_CLOSE, kernel, iterations=2)

    if dbg:
        dbg.save("30_all_field_components", result_mask)
        dbg.save("31_components_overlay", result_mask, overlay=True)

    return result_mask


def fit_rectangle_to_field(mask, img_rgb, dbg=None):
    """Финальная обработка контура поля с сохранением всей области"""
    h, w = mask.shape

    # Находим контуры
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask, None

    # Берём самый большой контур
    main_contour = max(contours, key=cv.contourArea)

    # Сглаживание контура морфологией
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (11, 11))
    mask_smooth = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    mask_smooth = cv.morphologyEx(mask_smooth, cv.MORPH_OPEN, kernel)

    # Находим контур после сглаживания
    contours_smooth, _ = cv.findContours(mask_smooth, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if contours_smooth:
        main_contour_smooth = max(contours_smooth, key=cv.contourArea)
    else:
        main_contour_smooth = main_contour

    # Упрощение контура Douglas-Peucker с малым epsilon для сохранения формы
    epsilon = 0.002 * cv.arcLength(main_contour_smooth, True)  # 0.2% периметра
    approx = cv.approxPolyDP(main_contour_smooth, epsilon, True)

    # Проверяем, нужна ли прямоугольная коррекция
    # Вычисляем минимальный прямоугольник для оценки наклона
    rect = cv.minAreaRect(approx)
    angle = rect[2]

    # Корректируем угол
    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = angle - 90

    # Если поле почти горизонтально (угол < 10°), используем выпуклую оболочку
    if abs(angle) < 10:
        hull = cv.convexHull(approx)
    else:
        # Если поле сильно наклонено, сохраняем исходную форму
        # но всё равно применяем выпуклую оболочку для устранения вогнутостей
        hull = cv.convexHull(main_contour_smooth)

    # Создаём финальную маску
    final_mask = np.zeros_like(mask)
    cv.fillPoly(final_mask, [hull], 255)

    # Проверка: если финальная маска потеряла больше 20% площади,
    # возвращаемся к исходной маске
    original_area = cv.countNonZero(mask)
    final_area = cv.countNonZero(final_mask)

    if final_area < original_area * 0.8:
        # Используем более консервативный подход
        hull = cv.convexHull(main_contour)
        final_mask = np.zeros_like(mask)
        cv.fillPoly(final_mask, [hull], 255)

    if dbg:
        dbg.save("40_smoothed_mask", mask_smooth)
        dbg.save("41_final_mask", final_mask)
        dbg.save("42_final_overlay", final_mask, overlay=True)

    return final_mask, hull


# ==============================
#   ГЛАВНАЯ ФУНКЦИЯ
# ==============================

def detect_football_field(filename: str) -> None:
    """Детектор футбольного поля на основе ExG с прямоугольной аппроксимацией"""

    # Загрузка изображения
    img_bgr = cv.imread(filename)
    if img_bgr is None:
        print(f"Ошибка: не удалось загрузить {filename}")
        return

    # Конвертация в RGB
    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    h_orig, w_orig, _ = img_rgb.shape

    # Минимальная обрезка чёрных полос (если есть)
    # Детекция чёрных полос
    gray = cv.cvtColor(img_rgb, cv.COLOR_RGB2GRAY)

    # Находим границы не-чёрной области
    non_black = gray > 10
    coords = np.column_stack(np.where(non_black))
    if coords.size > 0:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        # Обрезаем с небольшим запасом
        margin = 5
        y_min = max(0, y_min - margin)
        y_max = min(h_orig, y_max + margin)
        x_min = max(0, x_min - margin)
        x_max = min(w_orig, x_max + margin)

        img_rgb = img_rgb[y_min:y_max, x_min:x_max]
    else:
        x_min, y_min = 0, 0

    h, w, _ = img_rgb.shape

    # Инициализация отладчика
    stem = Path(filename).stem
    dbg = Debugger(stem, img_rgb) if DEBUG else None

    if dbg:
        dbg.save("00_input", img_rgb)

    # ШАГ 1: Вычисление ExG
    exg = compute_exg(img_rgb)

    # ШАГ 2: Пороговая обработка с Otsu
    mask = process_exg_otsu(exg, dbg)

    # ШАГ 3: Очистка маски
    mask = clean_mask(mask, dbg)

    # ШАГ 4: Поиск компонент поля
    mask = find_field_components(mask, dbg)

    # ШАГ 5: Прямоугольная аппроксимация
    final_mask, hull = fit_rectangle_to_field(mask, img_rgb, dbg)

    # Визуализация и сохранение результата
    if hull is not None:
        result = img_rgb.copy()
        cv.polylines(result, [hull], True, (255, 0, 0), thickness=max(2, w // 400))

        if dbg:
            dbg.save("99_final_result", result)

        if SAVE_RESULTS:
            # Восстанавливаем координаты для исходного изображения
            hull_original = hull + np.array([x_min, y_min])
            result_full = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
            cv.polylines(result_full, [hull_original], True, (255, 0, 0),
                         thickness=max(2, w_orig // 400))
            result_bgr = cv.cvtColor(result_full, cv.COLOR_RGB2BGR)
            cv.imwrite(f"processed_{filename}", result_bgr)
            print(f"✓ Обработано: {filename}")
    else:
        print(f"✗ Не удалось найти поле: {filename}")


# ==============================
#   ЗАПУСК
# ==============================

if __name__ == "__main__":
    images = [
        "panorama_1.jpg",
        "panorama_frame_2.jpg",
        "panorama_frame_3.jpg",
        "panorama_frame_50.jpg",
        "panorama_frame_51.jpg",
        "panorama_frame_52.jpg",
    ]

    print("Детекция поля: ExG + Otsu + прямоугольная аппроксимация")
    print("-" * 50)

    for img_file in images:
        try:
            detect_football_field(img_file)
        except Exception as e:
            print(f"✗ Ошибка при обработке {img_file}: {str(e)}")

    print("-" * 50)
    print("Обработка завершена!")