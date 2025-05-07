#!/usr/bin/env python3
# brightness_yolo_gradio.py

import gradio as gr
import cv2
import numpy as np
import tempfile
import os
from yolo_model import YOLOImageProcessor

# # 使用者指定模型路徑（硬編碼）
# MODEL_PATH = "/app/best.pt"

def process_and_calc_brightness(image: np.ndarray):
    """
    1️⃣ 把上傳的 NumPy RGB 圖轉 BGR 存檔到臨時 upload_dir  
    2️⃣ 呼叫 YOLOImageProcessor 處理該資料夾，裁出所有物件到 crop_dir  
    3️⃣ 讀取 crop_dir 裡每張裁切圖，計算灰階平均亮度  
    4️⃣ 回傳 (裁切影像列表, 亮度文字結果)
    """
    if image is None:
        return [], "❗️請先上傳圖片"
    if not MODEL_PATH or not os.path.isfile(MODEL_PATH):
        return [], f"❗️找不到模型檔: {MODEL_PATH}"

    cropped_images = []
    brightness_list = []
    try:
        with tempfile.TemporaryDirectory() as upload_dir, tempfile.TemporaryDirectory() as crop_dir:
            # 1) 存 input.png
            input_path = os.path.join(upload_dir, "input.png")
            bgr = image[:, :, ::-1] if image.ndim == 3 else image
            cv2.imwrite(input_path, bgr)

            # 2) YOLO 裁切
            yolo = YOLOImageProcessor(MODEL_PATH, upload_dir, crop_dir)
            yolo.process_images()

            # 3) 讀 crop_dir、蒐集裁切圖和亮度
            for fn in sorted(os.listdir(crop_dir)):
                fp = os.path.join(crop_dir, fn)
                crop = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
                if crop is None:
                    continue
                cropped_images.append(crop) 
                brightness_list.append(float(np.mean(crop)))
    except Exception as e:
        return [], f"🛑 處理失敗：{e}"

    # 4) 如果沒偵測到任何物件
    if not brightness_list:
        return [], "ℹ️ 未偵測到任何物件"

    # 計算文字結果
    avg = sum(brightness_list) / len(brightness_list)
    lines = [f"{i+1}. {v:.2f}" for i, v in enumerate(brightness_list)]
    result_text = (
        "🔹 各物件平均亮度:\n" + "\n".join(lines)
        + f"\n\n⭐️ 整體平均: {avg:.2f}"
    )
    # 回傳 (影像列表, 文字結果)
    return cropped_images, result_text

css = """
.gradio-container { max-width:600px; margin:auto; padding:20px; }
.gradio-row       { margin-top:20px; }
"""

with gr.Blocks(css=css, title="YOLO + 平均亮度計算") as demo:
    gr.Markdown(
        """
        <h1 style="text-align:center; color:#4A90E2;">
            🔍 YOLO + 平均亮度計算
        </h1>
        <p style="text-align:center;">
            上傳圖片，先做物件切割，再算裁切區域的平均灰階亮度。
        </p>
        """
    )

    with gr.Row():
        image_input      = gr.Image(sources=["upload"], type="numpy", label="📂 上傳影像")

    # 新增：用來顯示所有裁切後的影像
    gallery_output   = gr.Gallery(label="✂️ 裁切結果")

    brightness_output = gr.Textbox(label="🌟 亮度結果", interactive=False, lines=10)

    # 輸出改為 [gallery, text]
    image_input.upload(
        fn=process_and_calc_brightness,
        inputs=image_input,
        outputs=[gallery_output, brightness_output]
    )

    gr.Markdown(
        """
        **說明**  
        1. 點擊左側「Browse」或拖拉圖片  
        2. Gradio 會自動執行 YOLO 裁切並計算裁切後區域的平均亮度  
        3. 下方會顯示所有裁切區塊，並在右側顯示亮度數值
        """
    )
if __name__ == "__main__":
    # 在此修改模型路徑
    MODEL_PATH = "/app/best.pt"
    demo.launch(server_name="0.0.0.0", server_port=8800)
