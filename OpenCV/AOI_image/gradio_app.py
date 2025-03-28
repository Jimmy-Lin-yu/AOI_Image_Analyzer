import gradio as gr
import cv2
import numpy as np
from app import ImageQualityAnalyzer  # 假設你的class定義在app.py

# 定義處理影像的函數 (Gradio 用)
def analyze_image(uploaded_image):
    """
    1. 先確認是否有上傳圖片
    2. 轉成 OpenCV BGR 格式
    3. 計算各項品質指標
    4. 計算分數與總品質
    5. 回傳可讀性文字
    """
    if uploaded_image is None:
        return "請上傳一張圖片"

    # Gradio 傳入的圖片是 RGB，轉成 BGR 給 OpenCV 用
    image_bgr = cv2.cvtColor(uploaded_image, cv2.COLOR_RGB2BGR)

    # 建立分析器(如果你不需要 folder_path，可不傳參數)
    analyzer = ImageQualityAnalyzer()

    # 計算品質指標
    metrics = {
        "sharpness": analyzer.calculate_sharpness(image_bgr),
        "exposure": analyzer.calculate_exposure(image_bgr),
        "contrast": analyzer.calculate_contrast(image_bgr),
        "uniformity": analyzer.calculate_light_uniformity(image_bgr),
        "noise": analyzer.calculate_noise(image_bgr)
    }

    # 依據閾值計算分數與總品質
    scores, total = ImageQualityAnalyzer.evaluate_quality(metrics)

    # 將結果整理成文字輸出
    result_str = "影像品質分析結果：\n"
    for (k, v), score in zip(metrics.items(), scores):
        result_str += f"  {k.capitalize()}: {v:.2f} → Score {score}\n"
    result_str += f"\nTotal Quality: {total:.1f}%"

    return result_str

# 建立 Gradio 介面
demo = gr.Interface(
    fn=analyze_image,               # 綁定的處理函數
    inputs=gr.Image(type="numpy"),  # 上傳圖片 (回傳 numpy array, RGB)
    outputs="text",                 # 輸出文字
    title="Image Quality Analyzer",
    description="上傳一張圖片，分析各項品質並給出分數"
)

if __name__ == "__main__":
    demo.launch()
