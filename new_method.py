import cv2 as cv
import numpy as np
from pathlib import Path
from sklearn.mixture import GaussianMixture
from scipy import ndimage
from scipy.signal import find_peaks
from typing import Tuple, Optional, List

# ==============================
#   НАСТРОЙКИ
# ==============================

DEBUG = True
SAVE_RESULTS = True


# ==============================
#   ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==============================

class Debugger:
    """Класс для отладочного вывода и визуализации промежуточных результатов."""

    def __init__(self, stem: str, base_rgb: np.ndarray):
        self.dir = Path("debug") / stem
        self.dir.mkdir(parents=True, exist_ok=True)
        self.step = 0
        self.base = base_rgb

    def save(self, name: str, img: np.ndarray, overlay: bool = False) -> None:
        """Сохранить изображение для отладки."""
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

    def save_plot(self, name: str, fig) -> None:
        """Сохранить matplotlib график."""
        if not DEBUG:
            return
        fig.savefig(str(self.dir / f"{self.step:02d}_{name}.png"), dpi=150, bbox_inches='tight')
        self.step += 1


# ==============================
#   STAGE 1: PRE-PROCESSING
# ==============================

def preprocessing(img_rgb: np.ndarray, dbg: Optional[Debugger] = None) -> np.ndarray:
    """
    Морфологическое открытие для удаления мелких артефактов.

    Согласно статье: применяется grayscale opening с круглым структурным элементом
    диаметром D = 0.003 * sqrt(H^2 + W^2).

    Args:
        img_rgb: Входное RGB изображение
        dbg: Объект для отладочного вывода

    Returns:
        Предобработанное изображение
    """
    h, w, _ = img_rgb.shape

    # Вычисление диаметра согласно формуле из статьи
    D = int(0.003 * np.sqrt(h ** 2 + w ** 2))
    D = max(3, D)  # Минимум 3 пикселя
    if D % 2 == 0:
        D += 1  # Должен быть нечётным

    # Круглый структурный элемент
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (D, D))

    # Применение морфологического открытия к каждому каналу
    preprocessed = np.zeros_like(img_rgb)
    for i in range(3):
        preprocessed[:, :, i] = cv.morphologyEx(img_rgb[:, :, i], cv.MORPH_OPEN, kernel)

    if dbg:
        dbg.save("01_preprocessed", preprocessed)

    return preprocessed


# ==============================
#   STAGE 2: PIXEL-LEVEL SEGMENTATION
# ==============================

def compute_green_chromaticity(img_rgb: np.ndarray, dbg: Optional[Debugger] = None) -> np.ndarray:
    """
    Вычисление нормализованной зеленой компоненты g = G / (R + G + B).

    Согласно статье: g инвариантна к изменениям освещения и показывает
    близость цвета к зеленому (чем выше g, тем зеленее цвет).

    Args:
        img_rgb: Входное RGB изображение
        dbg: Объект для отладочного вывода

    Returns:
        Массив значений зеленой хроматичности [0, 1]
    """
    rgb_float = img_rgb.astype(np.float32)

    R = rgb_float[:, :, 0]
    G = rgb_float[:, :, 1]
    B = rgb_float[:, :, 2]

    sum_rgb = R + G + B
    sum_rgb = np.where(sum_rgb == 0, 1e-10, sum_rgb)  # Избегаем деления на ноль

    g = G / sum_rgb

    if dbg:
        g_vis = (g * 255).astype(np.uint8)
        dbg.save("02_green_chromaticity", g_vis)

    return g


def _gaussian_pdf(x: np.ndarray, mean: float, var: float) -> np.ndarray:
    """Плотность одномерного нормального распределения."""
    if var <= 0:
        var = 1e-6
    coeff = 1.0 / np.sqrt(2.0 * np.pi * var)
    return coeff * np.exp(-0.5 * ((x - mean) ** 2) / var)


def estimate_pdf_gmm(g: np.ndarray, n_components: int = 4,
                     dbg: Optional[Debugger] = None) -> Tuple[np.ndarray, np.ndarray, GaussianMixture, np.ndarray]:
    """
    Оценка плотности вероятности pdf(g) через смесь гауссиан (GMM).

    Согласно статье: используется EM-алгоритм с K=4 компонентами
    для аппроксимации распределения зеленой хроматичности.

    Args:
        g: Массив значений зеленой хроматичности
        n_components: Количество гауссиан в смеси (по умолчанию 4)
        dbg: Объект для отладочного вывода

    Returns:
        pdf_values: Значения pdf для диапазона [0, 1]
        g_range: Значения g, для которых вычислена pdf
        gmm: Обученная модель GMM
        component_pdfs: Значения плотности для каждой компоненты на g_range
    """
    g_flat = g.flatten().reshape(-1, 1)

    # Обучение GMM с фиксированным random_state для воспроизводимости
    gmm = GaussianMixture(
        n_components=n_components, random_state=42, max_iter=200, covariance_type="full"
    )
    gmm.fit(g_flat)

    # Вычисление pdf на диапазоне [0, 1]
    g_range = np.linspace(0, 1, 1000)
    component_pdfs = []
    for weight, mean, cov in zip(gmm.weights_, gmm.means_.flatten(), gmm.covariances_.reshape(-1)):
        component_pdfs.append(weight * _gaussian_pdf(g_range, mean, cov))

    component_pdfs = np.stack(component_pdfs, axis=1)
    pdf_values = np.sum(component_pdfs, axis=1)

    if dbg:
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(g_range, pdf_values, 'b-', linewidth=2, label='PDF(g)')
            ax.hist(g_flat, bins=100, range=(0, 1), density=True, alpha=0.3, color='green', label='Histogram')
            ax.set_xlabel('Green Chromaticity (g)', fontsize=12)
            ax.set_ylabel('Probability Density', fontsize=12)
            ax.set_title('PDF estimated by GMM (K=4)', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
            dbg.save_plot("03_pdf_gmm", fig)
            plt.close(fig)
        except ImportError:
            pass

    return pdf_values, g_range, gmm, component_pdfs


def _find_local_minima(pdf_values: np.ndarray) -> np.ndarray:
    """Поиск локальных минимумов в pdf."""
    inverted_pdf = -pdf_values
    prominence = max(np.max(pdf_values) * 0.02, 1e-3)
    minima, _ = find_peaks(inverted_pdf, prominence=prominence)
    return minima


def find_threshold_from_pdf(pdf_values: np.ndarray, g_range: np.ndarray,
                            gmm: GaussianMixture, component_pdfs: np.ndarray,
                            dbg: Optional[Debugger] = None) -> float:
    """
    Определение порога τ согласно алгоритму из статьи.

    Алгоритм следует описанию из статьи: выбирается первый локальный минимум
    после перехода, когда компоненты со средним g > 1/3 начинают доминировать.
    """
    threshold_g = 1.0 / 3.0
    means = gmm.means_.flatten()
    field_components = np.where(means > threshold_g)[0]

    if len(field_components) == 0:
        # Если не обнаружено ни одной компоненты с зелёными значениями,
        # возвращаем безопасный порог
        return 0.4

    responsibilities = component_pdfs / np.maximum(pdf_values[:, np.newaxis], 1e-12)
    field_resp = responsibilities[:, field_components].sum(axis=1)
    non_field_resp = 1.0 - field_resp

    # Ищем первую точку, где вероятность «зелёных» компонентов превышает остальные
    dominance_mask = (field_resp >= non_field_resp) & (g_range >= 0.15)
    candidate_indices = np.where(dominance_mask)[0]

    if len(candidate_indices) == 0:
        # fallback: ищем первый максимум соответствующей компоненты
        candidate_indices = np.where(field_resp > 0.5)[0]

    if len(candidate_indices) == 0:
        return 0.4

    transition_idx = candidate_indices[0]

    # Находим локальные минимумы и выбираем ближайший после перехода
    minima = _find_local_minima(pdf_values)
    minima_after_transition = minima[minima >= max(0, transition_idx - 5)]

    if len(minima_after_transition) > 0:
        tau_idx = minima_after_transition[0]
    else:
        tau_idx = transition_idx

    tau = g_range[tau_idx]
    tau = float(np.clip(tau, 0.2, 0.55))

    if dbg:
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(g_range, pdf_values, 'b-', linewidth=2, label='PDF(g)')
            ax.axvline(x=threshold_g, color='gray', linestyle='--', alpha=0.5, label='g = 1/3')
            minima = _find_local_minima(pdf_values)
            if len(minima) > 0:
                ax.plot(g_range[minima], pdf_values[minima], 'mx', markersize=10, label='Minima')
            ax.axvline(x=tau, color='red', linestyle='-', linewidth=2, label=f'Threshold τ = {tau:.3f}')
            ax.set_xlabel('Green Chromaticity (g)', fontsize=12)
            ax.set_ylabel('Probability Density', fontsize=12)
            ax.set_title('Threshold Selection', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
            dbg.save_plot("04_threshold_selection", fig)
            plt.close(fig)

            # Дополнительная визуализация вероятностей компонентов
            if len(field_components) > 0:
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                ax2.plot(g_range, field_resp, label='Field responsibility')
                ax2.plot(g_range, non_field_resp, label='Non-field responsibility')
                ax2.axvline(x=tau, color='red', linestyle='--', label=f'τ = {tau:.3f}')
                ax2.set_xlabel('Green Chromaticity (g)', fontsize=12)
                ax2.set_ylabel('Responsibility', fontsize=12)
                ax2.set_title('Component responsibilities')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                dbg.save_plot("04b_responsibilities", fig2)
                plt.close(fig2)
        except ImportError:
            pass

    return tau


def segment_by_chromaticity(g: np.ndarray, tau: float,
                            dbg: Optional[Debugger] = None) -> np.ndarray:
    """
    Создание маски M1 на основе порога зеленой хроматичности.

    Args:
        g: Массив значений зеленой хроматичности
        tau: Пороговое значение
        dbg: Объект для отладочного вывода

    Returns:
        Бинарная маска M1
    """
    M1 = (g > tau).astype(np.uint8)

    if dbg:
        dbg.save("05_M1_chromaticity_mask", M1 * 255)
        dbg.save("06_M1_overlay", M1 * 255, overlay=True)

    return M1


# ==============================
#   STAGE 3: CHROMATIC DISTORTION FILTERING
# ==============================

def partition_by_pdf_minima(g: np.ndarray, pdf_values: np.ndarray,
                            g_range: np.ndarray) -> Tuple[List[np.ndarray], List[Tuple[float, float]]]:
    """
    Разделение пикселей на сегменты между соседними минимумами pdf.

    Согласно статье: пиксели группируются в сегменты, разделенные
    локальными минимумами функции pdf(g).

    Args:
        g: Массив значений зеленой хроматичности
        pdf_values: Значения pdf
        g_range: Значения g для pdf

    Returns:
        segments: Список масок для каждого сегмента
        segment_ranges: Список диапазонов (g_min, g_max)
    """
    minima = _find_local_minima(pdf_values)

    # Границы сегментов
    boundaries = [0] + list(g_range[minima]) + [1.0]
    boundaries = sorted(set(boundaries))

    segments = []
    segment_ranges = []

    for i in range(len(boundaries) - 1):
        g_min = boundaries[i]
        g_max = boundaries[i + 1]

        segment_mask = (g >= g_min) & (g < g_max)

        if np.any(segment_mask):
            segments.append(segment_mask)
            segment_ranges.append((g_min, g_max))

    return segments, segment_ranges


def compute_chromatic_distortion(img_rgb: np.ndarray, segment_mask: np.ndarray) -> Tuple[
    np.ndarray, Optional[np.ndarray]]:
    """
    Векторизованное вычисление хроматической дисторсии для пикселей сегмента.

    Согласно статье: cd = ||v_perp|| / ||v_parallel||, где v - RGB вектор пикселя,
    c - средний RGB вектор сегмента, v_parallel - проекция v на c.

    Args:
        img_rgb: Входное RGB изображение
        segment_mask: Маска текущего сегмента

    Returns:
        cd: Массив значений хроматической дисторсии
        mean_vector: Средний RGB вектор сегмента
    """
    h, w, _ = img_rgb.shape
    cd = np.zeros((h, w), dtype=np.float32)

    segment_pixels = img_rgb[segment_mask].astype(np.float32)

    if len(segment_pixels) == 0:
        return cd, None

    # Средний RGB вектор сегмента
    mean_vector = np.mean(segment_pixels, axis=0)

    mean_norm = np.linalg.norm(mean_vector)
    if mean_norm < 1e-10:
        return cd, mean_vector

    c_normalized = mean_vector / mean_norm

    # Векторизованные вычисления
    v_parallel_scalar = np.dot(segment_pixels, c_normalized)
    v_parallel = v_parallel_scalar[:, np.newaxis] * c_normalized[np.newaxis, :]
    v_perp = segment_pixels - v_parallel

    norm_v_perp = np.linalg.norm(v_perp, axis=1)
    norm_v_parallel = np.linalg.norm(v_parallel, axis=1)

    # Вычисление cd с защитой от деления на ноль
    cd_values = np.zeros(len(segment_pixels), dtype=np.float32)
    valid_mask = norm_v_parallel > 1e-10
    cd_values[valid_mask] = norm_v_perp[valid_mask] / norm_v_parallel[valid_mask]

    # Заполнение результатов
    y_coords, x_coords = np.where(segment_mask)
    cd[y_coords, x_coords] = cd_values

    return cd, mean_vector


def is_narrow_peak_near_zero(cd_values: np.ndarray, gamma: float) -> bool:
    """
    Проверка наличия узкого пика около нуля согласно рекомендациям статьи.

    Гистограмма cd для «правильных» сегментов имеет узкий пик, расположенный
    вблизи нуля и левее порога γ.
    """
    if len(cd_values) == 0:
        return False

    hist_range = (0, max(float(cd_values.max()), gamma * 3.0))
    hist, bin_edges = np.histogram(cd_values, bins=64, range=hist_range)
    if np.sum(hist) == 0:
        return False

    max_bin = int(np.argmax(hist))
    mode_center = 0.5 * (bin_edges[max_bin] + bin_edges[max_bin + 1])

    perc70 = np.percentile(cd_values, 70)
    perc90 = np.percentile(cd_values, 90)

    mode_ok = mode_center < gamma
    concentrated = perc70 < gamma * 1.2 and perc90 < gamma * 2.0

    return bool(mode_ok and concentrated)


def filter_by_chromatic_distortion(img_rgb: np.ndarray, M1: np.ndarray, g: np.ndarray,
                                   pdf_values: np.ndarray, g_range: np.ndarray,
                                   gamma: float = 0.18, dbg: Optional[Debugger] = None) -> np.ndarray:
    """
    Фильтрация по хроматической дисторсии для удаления ложных срабатываний.

    Согласно статье:
    1. Разделить пиксели на сегменты между минимумами pdf
    2. Для каждого сегмента проверить форму гистограммы cd
    3. Отбросить сегменты с широким распределением
    4. Применить порог cd < γ к оставшимся сегментам

    Args:
        img_rgb: Входное RGB изображение
        M1: Маска после порога по хроматичности
        g: Массив значений зеленой хроматичности
        pdf_values: Значения pdf(g)
        g_range: Диапазон g для pdf
        gamma: Порог для cd (см. ограничение конуса на рис. 4 и Eq. (5) в статье)
        dbg: Объект для отладочного вывода

    Returns:
        Отфильтрованная маска M2
    """
    h, w = M1.shape
    M2 = np.zeros((h, w), dtype=np.uint8)
    cd_full = np.zeros((h, w), dtype=np.float32)

    # Разделение на сегменты
    segments, segment_ranges = partition_by_pdf_minima(g, pdf_values, g_range)

    if dbg:
        print(f"  Найдено сегментов: {len(segments)}")

    for idx, (segment_mask, (g_min, g_max)) in enumerate(zip(segments, segment_ranges)):
        # Применяем маску M1 к сегменту
        segment_green = segment_mask & (M1 > 0)

        if not np.any(segment_green):
            continue

        num_pixels = np.sum(segment_green)
        if dbg:
            print(f"  Сегмент {idx + 1}/{len(segments)}: g∈[{g_min:.3f}, {g_max:.3f}], "
                  f"пикселей={num_pixels}...", end=" ", flush=True)

        # Вычисление хроматической дисторсии
        cd, mean_vector = compute_chromatic_distortion(img_rgb, segment_green)
        cd_full += cd

        # Анализ распределения cd
        cd_values = cd[segment_green]

        if len(cd_values) == 0:
            if dbg:
                print("пусто")
            continue

        # Проверка формы гистограммы (адаптированные критерии)
        if not is_narrow_peak_near_zero(cd_values, gamma):
            if dbg:
                median_cd = np.median(cd_values)
                pct_below = np.sum(cd_values < gamma) / len(cd_values) * 100
                print(f"✗ отброшен (медиана={median_cd:.3f}, <γ: {pct_below:.1f}%)")
            continue

        # Применение порога cd < gamma
        segment_filtered = segment_green & (cd < gamma)
        M2 = M2 | segment_filtered.astype(np.uint8)

        if dbg:
            median_cd = np.median(cd_values)
            pct_below = np.sum(cd_values < gamma) / len(cd_values) * 100
            pixels_accepted = np.sum(segment_filtered)
            print(f"✓ принят (медиана={median_cd:.3f}, <γ: {pct_below:.1f}%, "
                  f"выбрано={pixels_accepted})")

    if dbg:
        # Визуализация результатов
        cd_vis = np.clip(cd_full * 255 / (gamma * 2), 0, 255).astype(np.uint8)
        dbg.save("07_chromatic_distortion", cd_vis)
        dbg.save("08_M2_cd_filtered", M2 * 255)
        dbg.save("09_M2_overlay", M2 * 255, overlay=True)

        # Гистограмма cd
        try:
            import matplotlib.pyplot as plt
            cd_values_all = cd_full[M1 > 0]
            if len(cd_values_all) > 0:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(cd_values_all, bins=100, density=True, alpha=0.7, color='blue')
                ax.axvline(x=gamma, color='red', linestyle='--', linewidth=2, label=f'γ = {gamma}')
                ax.set_xlabel('Chromatic Distortion (cd)', fontsize=12)
                ax.set_ylabel('Density', fontsize=12)
                ax.set_title('Histogram of Chromatic Distortion', fontsize=14)
                ax.legend()
                ax.grid(True, alpha=0.3)
                dbg.save_plot("10_cd_histogram", fig)
                plt.close(fig)
        except ImportError:
            pass

    return M2


# ==============================
#   STAGE 4: POST-PROCESSING
# ==============================

def postprocess_mask(mask: np.ndarray, dbg: Optional[Debugger] = None) -> np.ndarray:
    """
    Постобработка маски: заполнение дыр и выбор крупнейшего компонента.

    Согласно статье:
    1. Заполнение дыр для устранения игроков, линий и других объектов
    2. Выбор крупнейшего связного компонента как финального поля

    Args:
        mask: Входная бинарная маска
        dbg: Объект для отладочного вывода

    Returns:
        Финальная маска поля
    """
    # Заполнение дыр
    mask_bool = mask.astype(bool)
    filled = ndimage.binary_fill_holes(mask_bool)
    mask_filled = filled.astype(np.uint8)

    if dbg:
        dbg.save("11_filled_holes", mask_filled * 255)

    # Выбор крупнейшего связного компонента
    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(mask_filled, connectivity=8)

    if num_labels <= 1:
        final_mask = mask_filled
    else:
        # Исключаем фон (label=0)
        areas = stats[1:, cv.CC_STAT_AREA]
        largest_label = np.argmax(areas) + 1
        final_mask = (labels == largest_label).astype(np.uint8)

    if dbg:
        dbg.save("12_largest_component", final_mask * 255)
        dbg.save("13_final_overlay", final_mask * 255, overlay=True)

    return final_mask


# ==============================
#   ГЛАВНАЯ ФУНКЦИЯ
# ==============================

def detect_football_field(filename: str, skip_cd_filter: bool = False) -> None:
    """
    Детекция футбольного поля согласно алгоритму из статьи.

    Этапы:
    1. Pre-processing: морфологическое открытие
    2. Pixel-level segmentation: анализ зеленой хроматичности
    3. Chromatic distortion filtering: фильтрация ложных срабатываний
    4. Post-processing: заполнение дыр и выбор крупнейшего компонента

    Args:
        filename: Путь к изображению
    """
    # Загрузка изображения
    img_bgr = cv.imread(filename)
    if img_bgr is None:
        print(f"Ошибка: не удалось загрузить {filename}")
        return

    # Конвертация в RGB
    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    h_orig, w_orig, _ = img_rgb.shape

    # Обрезка черных полос (если есть)
    gray = cv.cvtColor(img_rgb, cv.COLOR_RGB2GRAY)
    non_black = gray > 10
    coords = np.column_stack(np.where(non_black))

    if coords.size > 0:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

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
        print(f"\n{'=' * 60}")
        print(f"Обработка: {filename}")
        print(f"Размер: {w}×{h}")
        print(f"{'=' * 60}")

    # ===== STAGE 1: PRE-PROCESSING =====
    if dbg:
        print("\n[Stage 1] Pre-processing...")
    img_preprocessed = preprocessing(img_rgb, dbg)

    # ===== STAGE 2: PIXEL-LEVEL SEGMENTATION =====
    if dbg:
        print("[Stage 2] Pixel-level segmentation...")

    g = compute_green_chromaticity(img_preprocessed, dbg)
    pdf_values, g_range, gmm, component_pdfs = estimate_pdf_gmm(g, n_components=4, dbg=dbg)
    tau = find_threshold_from_pdf(pdf_values, g_range, gmm, component_pdfs, dbg)

    if dbg:
        print(f"  Порог τ = {tau:.3f}")

    M1 = segment_by_chromaticity(g, tau, dbg)

    if dbg:
        green_ratio = np.sum(M1) / (h * w)
        print(f"  Зелёных пикселей (M1): {green_ratio:.2%}")

    # ===== STAGE 3: CHROMATIC DISTORTION FILTERING =====
    if dbg:
        print("[Stage 3] Chromatic distortion filtering...")

    # Используем адаптивный gamma для изображений с дисторсией
    gamma = 0.18  # См. анализ параметров γ в статье
    if skip_cd_filter:
        # Для изображений с сильной дисторсией пропускаем CD фильтрацию
        M2 = M1
        if dbg:
            print("[Stage 3] Chromatic distortion filtering... SKIPPED (fisheye mode)")
    else:
        M2 = filter_by_chromatic_distortion(img_preprocessed, M1, g, pdf_values, g_range,
                                        gamma=gamma, dbg=dbg)

    if dbg:
        green_ratio_filtered = np.sum(M2) / (h * w)
        print(f"  Зелёных пикселей (M2): {green_ratio_filtered:.2%}")

    # ===== STAGE 4: POST-PROCESSING =====
    if dbg:
        print("[Stage 4] Post-processing...")
    final_mask = postprocess_mask(M2, dbg)

    if dbg:
        final_ratio = np.sum(final_mask) / (h * w)
        print(f"  Финальная площадь поля: {final_ratio:.2%}")

    # Визуализация и сохранение результата
    if np.any(final_mask):
        contours, _ = cv.findContours(final_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        if contours:
            result = img_rgb.copy()
            cv.drawContours(result, contours, -1, (255, 0, 0), thickness=max(2, w // 400))

            if dbg:
                dbg.save("14_final_result", result)

            if SAVE_RESULTS:
                result_full = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)

                # Восстановление координат для исходного изображения
                for contour in contours:
                    contour[:, 0, 0] += x_min
                    contour[:, 0, 1] += y_min

                cv.drawContours(result_full, contours, -1, (255, 0, 0),
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

    print("=" * 60)
    print("ДЕТЕКЦИЯ ПОЛЯ: Алгоритм из статьи")
    print("Green Chromaticity + Chromatic Distortion")
    print("=" * 60)

    for img_file in images:
        try:
            detect_football_field(img_file, skip_cd_filter=False)
        except Exception as e:
            print(f"✗ Ошибка при обработке {img_file}: {str(e)}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 60)
    print("Обработка завершена!")
    print("=" * 60)