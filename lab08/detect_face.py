import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw
import warnings
from scipy.ndimage import convolve, gaussian_filter

MODEL_PATH = "version-RFB-320.onnx"

warnings.filterwarnings("ignore", category=UserWarning)

sess_options = ort.SessionOptions()
sess_options.log_severity_level = 3  

# 3. Use only available providers to prevent the provider fallback warning
desired_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
available_providers = [p for p in desired_providers if p in ort.get_available_providers()]

# face detection
def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    indexes = np.argsort(scores)
    indexes = indexes[-candidate_size:]
    while len(indexes) > 0:
        current = indexes[-1]
        picked.append(current)
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[:-1]
        rest = boxes[indexes, :]

        x1 = np.maximum(rest[:, 0], current_box[0])
        y1 = np.maximum(rest[:, 1], current_box[1])
        x2 = np.minimum(rest[:, 2], current_box[2])
        y2 = np.minimum(rest[:, 3], current_box[3])
        inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        area_current = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
        area_rest = (rest[:, 2] - rest[:, 0]) * (rest[:, 3] - rest[:, 1])
        iou = inter / (area_current + area_rest - inter)

        indexes = indexes[iou <= iou_threshold]
    return box_scores[picked, :]

def detect_faces(image_path, output_path="output_detected.png", conf_threshold=0.7):
    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size

    resized = image.resize((320, 240))
    x = np.asarray(resized, dtype=np.float32)
    x = (x - 127.0) / 128.0
    x = np.transpose(x, (2, 0, 1))[np.newaxis, :, :, :]

    session = ort.InferenceSession(MODEL_PATH, sess_options, providers=available_providers)
    input_name = session.get_inputs()[0].name
    scores, boxes = session.run(None, {input_name: x})

    scores = scores[0]
    boxes = boxes[0]
    probs = scores[:, 1]
    mask = probs >= conf_threshold
    candidates = np.concatenate([boxes[mask], probs[mask].reshape(-1, 1)], axis=1)
    picked = hard_nms(candidates, iou_threshold=0.3) if len(candidates) else candidates

    draw = ImageDraw.Draw(image)
    for x1, y1, x2, y2, score in picked:
        left, top = x1 * orig_w, y1 * orig_h
        right, bottom = x2 * orig_w, y2 * orig_h
        draw.rectangle([left, top, right, bottom], outline=(0, 255, 0), width=3)
        draw.text((left + 3, top - 16), f"Face: {score:.2f}", fill=(0, 255, 0))
    image.save(output_path)
    print(f"{len(picked)} face(s) detected; saved to {output_path}")
    return picked

# Edge detection
def non_maximum_suppression(magnitude, angle):
    h, w = magnitude.shape
    out = np.zeros((h, w), dtype=np.float32)
    deg = angle * 180.0 / np.pi
    deg[deg < 0] += 180

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            a = deg[i, j]
            if (0 <= a < 22.5) or (157.5 <= a <= 180):
                q, r = magnitude[i, j - 1], magnitude[i, j + 1]
            elif 22.5 <= a < 67.5:
                q, r = magnitude[i + 1, j - 1], magnitude[i - 1, j + 1]
            elif 67.5 <= a < 112.5:
                q, r = magnitude[i - 1, j], magnitude[i + 1, j]
            else:
                q, r = magnitude[i - 1, j - 1], magnitude[i + 1, j + 1]
            if magnitude[i, j] >= q and magnitude[i, j] >= r:
                out[i, j] = magnitude[i, j]
    return out

def hysteresis_thresholding(img, low_thresh, high_thresh):
    strong, weak = 255, 50
    res = np.zeros(img.shape, dtype=np.uint8)
    res[img >= high_thresh] = strong
    res[(img >= low_thresh) & (img < high_thresh)] = weak

    h, w = res.shape
    for i, j in zip(*np.where(res == weak)):
        window = res[max(i - 1, 0):min(i + 2, h), max(j - 1, 0):min(j + 2, w)]
        res[i, j] = strong if (window == strong).any() else 0
    return res

def canny_scipy(image_path, output_path="edges_scipy.png", low=20, high=60):
    gray = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32)
    smoothed = gaussian_filter(gray, sigma=1.4)

    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
    gx = convolve(smoothed, kx)
    gy = convolve(smoothed, ky)

    magnitude = np.hypot(gx, gy)
    magnitude = magnitude / magnitude.max() * 255
    angle = np.arctan2(gy, gx)

    thin = non_maximum_suppression(magnitude, angle)
    edges = hysteresis_thresholding(thin, low, high)
    Image.fromarray(edges.astype(np.uint8)).save(output_path)
    print(f"edge map saved to {output_path}")
    return edges

if __name__ == "__main__":
    detect_faces("face.jpg")
    canny_scipy("face.jpg")