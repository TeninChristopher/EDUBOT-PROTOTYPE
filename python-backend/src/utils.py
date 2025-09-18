# src/utils.py
import cv2
import numpy as np
import tensorflow as tf
import os

# --------------------------
# Image preprocessing helpers
# --------------------------
IMG_HEIGHT = 64      # model input height
IMG_MAX_WIDTH = 1024 # padded width used for training/inference

def load_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img

def resize_and_pad(img, target_h=IMG_HEIGHT, max_w=IMG_MAX_WIDTH):
    """Resize to target_h while keeping aspect ratio, then pad to max_w."""
    h, w = img.shape
    scale = target_h / h
    new_w = max(1, int(round(w * scale)))
    resized = cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_LINEAR)
    if new_w < max_w:
        pad = np.ones((target_h, max_w - new_w), dtype=resized.dtype) * 255
        out = np.concatenate([resized, pad], axis=1)
    else:
        out = cv2.resize(resized, (max_w, target_h), interpolation=cv2.INTER_LINEAR)
    out = out.astype(np.float32) / 255.0  # normalize 0..1 (white=1)
    out = np.expand_dims(out, axis=-1)    # H,W,1
    return out

# --------------------------
# Page line segmentation
# --------------------------
def segment_page_lines(page_img, debug_out=None, min_width=40, min_height=10):
    """
    Input: grayscale image (numpy array)
    Output: list of (y,x,h,w) boxes sorted top-to-bottom representing text lines.
    """
    # Normalize + denoise
    img = cv2.fastNlMeansDenoising(page_img, None, h=7)
    # Binary inverse: text=white
    _, thr = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Morphological dilation to connect characters into line blobs
    kernel_w = max(1, page_img.shape[1] // 30)  # adapt to page width
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 3))
    dil = cv2.dilate(thr, kernel, iterations=1)

    # Find contours
    cnts = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    boxes = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w < min_width or h < min_height:
            continue
        boxes.append((y, x, h, w))

    boxes.sort(key=lambda b: b[0])  # sort top-to-bottom

    if debug_out:
        dbg = cv2.cvtColor(page_img, cv2.COLOR_GRAY2BGR)
        for (y, x, h, w) in boxes:
            cv2.rectangle(dbg, (x, y), (x+w, y+h), (0,255,0), 2)
        os.makedirs(os.path.dirname(debug_out), exist_ok=True)
        cv2.imwrite(debug_out, dbg)

    return boxes

# --------------------------
# Paragraph reconstruction helpers
# --------------------------
def reconstruct_paragraphs(boxes):
    """
    Given list of boxes (y,x,h,w) sorted top-to-bottom,
    compute gaps between consecutive boxes and determine paragraph breaks.
    Returns list of lists of boxes for each paragraph.
    """
    if not boxes:
        return []

    # compute vertical gaps
    bottoms = [y + h for (y,x,h,w) in boxes]
    gaps = []
    for i in range(1, len(boxes)):
        prev_bottom = bottoms[i-1]
        curr_top = boxes[i][0]
        gaps.append(curr_top - prev_bottom)

    if not gaps:
        return [boxes]

    median_gap = np.median(gaps)
    # heuristic threshold: gaps much larger than median indicate paragraph break
    gap_thr = max(1.5 * median_gap, 20)

    paragraphs = []
    current = [boxes[0]]
    for i in range(1, len(boxes)):
        gap = boxes[i][0] - bottoms[i-1]
        if gap > gap_thr:
            paragraphs.append(current)
            current = [boxes[i]]
        else:
            current.append(boxes[i])
    paragraphs.append(current)
    return paragraphs

# --------------------------
# CTC decoding util (greedy)
# --------------------------
def ctc_greedy_decode(preds, num_to_char):
    """
    preds: numpy array [T, C] or [B,T,C]
    num_to_char: dict or list mapping int->char (int indices are model vocab indices)
    Returns list of decoded strings
    """
    if preds.ndim == 3:
        batch = preds
    else:
        batch = np.expand_dims(preds, axis=0)

    decoded = []
    for p in batch:
        # pick argmax per time step
        seq = np.argmax(p, axis=1)
        # collapse repeats and remove blanks (assume blank index = 0)
        out = []
        prev = -1
        for s in seq:
            if s != prev and s != 0:
                out.append(s)
            prev = s
        # map to chars (model indices start at 1 for first char)
        chars = []
        for idx in out:
            # idx is in 1..V where 1 corresponds to first char in charset
            chars.append(num_to_char.get(int(idx), ""))
        decoded.append("".join(chars))
    return decoded
