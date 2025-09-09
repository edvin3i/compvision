import cv2 as cv
import numpy as np

from matplotlib import pyplot as plt

def fb_field_detector(filename: str) -> None:
    img_bgr = cv.imread(filename)
    assert img_bgr is not None, "File not found."
    img = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    plt.imshow(img); plt.axis('off'); plt.title('Source picture')
    # plt.show()
    hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)

    lower = np.array([28, 69, 37], dtype=np.uint8)
    upper = np.array([95, 255, 255], dtype=np.uint8)

    mask = cv.inRange(hsv, lower, upper)
    plt.imshow(mask, cmap='gray'); plt.axis('off'); plt.title("Mask for green")
    # plt.show()

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (9, 9))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=2) # del holes
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=1) # del noise points
    plt.imshow(mask, cmap='gray'); plt.axis('off'); plt.title("Mask after morphology")
    # plt.show()

    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        field_mask = mask.copy()
    else:
        largest_idx = 1 + np.argmax(stats[1:, cv.CC_STAT_AREA])
        field_mask = np.zeros_like(mask)
        field_mask[labels == largest_idx] = 255
    plt.imshow(field_mask, cmap='gray'); plt.axis('off'); plt.title("Fiels is largest component")
    plt.show()

    # GrabCut
    gc_mask = np.where(field_mask>0, cv.GC_PR_FGD, cv.GC_PR_BGD).astype('uint8')
    h, w = field_mask.shape
    gc_mask[int(h*0.86):, :] = cv.GC_BGD
    gc_mask[:int(h*0.05), :int(w*0.15)] = cv.GC_BGD
    gc_mask[:int(h*0.05), int(w*0.85):] = cv.GC_BGD
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)
    cv.grabCut(img, gc_mask, None, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_MASK)
    mask_gc = np.where((gc_mask == cv.GC_FGD) | (gc_mask == cv.GC_PR_FGD), 255, 0).astype('uint8')
    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(mask_gc, connectivity=8)
    largest_idx = 1 + np.argmax(stats[1:, cv.CC_STAT_AREA]) if num_labels > 1 else 0
    field_mask = np.zeros_like(mask_gc); field_mask[labels == largest_idx] = 255
    plt.imshow(mask_gc, cmap='gray'); plt.axis('off'); plt.title("GrabCut algo")
    # plt.show()

    cnts, _ = cv.findContours(field_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    assert cnts, "Countour did not found"
    cnt = max(cnts, key=cv.contourArea) # biggest countour
    epsilon = 0.003 * cv.arcLength(cnt, True)
    poly = cv.approxPolyDP(cnt, epsilon, True).reshape(-1, 2)
    len(poly)

    overlay = img.copy()
    cv.polylines(overlay, [poly], isClosed=True, color=(255, 0, 0), thickness=3) # draw polyline
    plt.imshow(overlay); plt.axis('off'); plt.title("Polygon")
    # plt.show()


if __name__ == '__main__':
    fb_field_detector("panorama_1.jpg")
    # fb_field_detector("panorama_frame_2.jpg")