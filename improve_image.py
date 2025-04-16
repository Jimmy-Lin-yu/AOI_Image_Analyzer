import os
import cv2
import numpy as np

# ------------------- 單張影像增強 -------------------
def enhance_image_based_on_scores(img_bgr: np.ndarray, scores: dict) -> np.ndarray:
    """
    img_bgr : 3‑channel BGR 影像 (cv2.imread() 讀進來的格式)
    scores  : 例如 {"sharpness": 5, "exposure": 3, "contrast": 2,
                    "uniformity": 3, "noise": 4}
              數字範圍 1–5，數字越小代表品質越差
    return  : 增強後的 BGR 影像
    """
    # 1️⃣ 影像銳利化（Un‑sharp Mask）
    if scores.get("sharpness", 5) <= 3:
        blur = cv2.GaussianBlur(img_bgr, (0, 0), 3)
        img_bgr = cv2.addWeighted(img_bgr, 1.5, blur, -0.5, 0)

    # 2️⃣ 曝光（亮度）補償 – gamma/亮度調整
    if scores.get("exposure", 5) <= 3:
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        # γ 校正：提高或降低 L 通道亮度
        gamma = 1.2 if l.mean() < 128 else 0.8
        l = np.clip(((l / 255.0) ** gamma) * 255.0, 0, 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    # 3️⃣ 對比度不足 – CLAHE
    if scores.get("contrast", 5) <= 3:
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        img_bgr = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    # 4️⃣ 均勻度不足 – 簡易光照校正（模糊減除法）
    if scores.get("uniformity", 5) <= 3:
        # 取得亮度分量並進行背景估計
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        background = cv2.medianBlur(gray, 51)
        diff = cv2.subtract(gray, background)
        diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
        img_bgr = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)

    # 5️⃣ 雜訊過高 – Non‑Local Means 去雜訊
    if scores.get("noise", 5) <= 3:
        img_bgr = cv2.fastNlMeansDenoisingColored(img_bgr, None,
                                                  h=10, hColor=10,
                                                  templateWindowSize=7,
                                                  searchWindowSize=21)
    return img_bgr

# ------------------- 批次處理 -------------------
def batch_enhance(input_dir="input_image", output_dir="output_image",
                  analyzer=None, score_threshold=3):
    """
    讀取 input_dir 所有影像，依品質評分自動增強後存到 output_dir
    analyzer : 你的 ImageQualityAnalyzer 實例
               (需提供 calculate_* 與 evaluate_quality)
    score_threshold : 低於等於此分數才增強
    """
    os.makedirs(input_dir,  exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    if analyzer is None:
        raise ValueError("請傳入 ImageQualityAnalyzer 物件")

    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
    for fname in os.listdir(input_dir):
        if not fname.lower().endswith(exts):
            continue

        in_path  = os.path.join(input_dir, fname)
        out_path = os.path.join(output_dir, fname)

        img = cv2.imread(in_path)
        if img is None:
            print(f"⚠️ 讀取失敗：{fname}")
            continue

        # 取得各項指標與 Score
        metrics = {
            "sharpness":  analyzer.calculate_sharpness(img),
            "exposure":   analyzer.calculate_exposure(img),
            "contrast":   analyzer.calculate_contrast(img),
            "uniformity": analyzer.calculate_light_uniformity(img),
            "noise":      analyzer.calculate_noise(img)
        }
        scores, _ = analyzer.evaluate_quality(metrics)
        score_dict = {k: s for (k, _), s in zip(metrics.items(), scores)}

        # 若有任何項目分數低於門檻，就進行增強
        if any(v <= score_threshold for v in score_dict.values()):
            img = enhance_image_based_on_scores(img, score_dict)

        cv2.imwrite(out_path, img)
        print(f"✅ 已處理：{fname}")

# ------------------- 使用範例 -------------------
if __name__ == "__main__":
    from image_evaluation import ImageQualityAnalyzer   # 依你的專案路徑調整
    analyzer = ImageQualityAnalyzer()
    batch_enhance(input_dir="input_image",
                  output_dir="output_image",
                  analyzer=analyzer,
                  score_threshold=3)
