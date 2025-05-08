import gradio as gr
import cv2
import numpy as np
import tempfile
import os
import sys
from yolo_model import YOLOImageProcessor

class YOLOBrightnessCalculator:
    """
    封裝 YOLO 裁切與平均亮度計算功能的類別。
    使用方法：
        calc = YOLOBrightnessCalculator(model_path)
        crops, text = calc.process(image_np)
    """
    def __init__(self, model_path: str):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"找不到模型檔: {model_path}")
        self.model_path = model_path

    def process(self, image: np.ndarray):
        """
        Args:
            image: numpy.ndarray (RGB 格式) 或 BGR 都可
        Returns:
            cropped_images: list[np.ndarray] 灰階裁切結果列表
            result_text: str 計算後的文字說明
        """
        if image is None:
            return [], "❗️ 請先提供影像"

        cropped_images = []
        brightness_list = []
        try:
            with tempfile.TemporaryDirectory() as upload_dir, tempfile.TemporaryDirectory() as crop_dir:
                # 存原始影像
                in_path = os.path.join(upload_dir, "input.png")
                bgr = image[:, :, ::-1] if image.ndim == 3 else image
                cv2.imwrite(in_path, bgr)

                # YOLO 裁切
                yolo = YOLOImageProcessor(self.model_path, upload_dir, crop_dir)
                yolo.process_images()

                # 計算裁切後每張影像亮度
                for fn in sorted(os.listdir(crop_dir)):
                    fp = os.path.join(crop_dir, fn)
                    crop = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
                    if crop is not None:
                        cropped_images.append(crop)
                        brightness_list.append(float(np.mean(crop)))
        except Exception as e:
            return [], f"🛑 錯誤：{e}"

        if not brightness_list:
            return [], "ℹ️ 未偵測到任何物件"

        # 組合結果文字
        avg = sum(brightness_list) / len(brightness_list)
        lines = [f"{i+1}. {v:.2f}" for i, v in enumerate(brightness_list)]
        text = (
            "🔹 各物件平均亮度:\n" + "\n".join(lines)
            + f"\n\n⭐️ 整體平均: {avg:.2f}"
        )
        return cropped_images, text


def launch_ui(model_path: str, port: int = 8800):
    """
    啟動 Gradio 介面。
    """
    calc = YOLOBrightnessCalculator(model_path)
    css = ".gradio-container{max-width:600px;margin:auto;padding:20px;} .gradio-row{margin-top:20px;}"
    with gr.Blocks(css=css, title="YOLO + 平均亮度") as demo:
        gr.Markdown("""
        ### 🔍 YOLO + 平均亮度計算
        上傳圖片，先執行 YOLO 裁切，再計算各裁切區平均灰階亮度。
        """)
        with gr.Row():
            img_in = gr.Image(sources=["upload"], type="numpy", label="📂 上傳影像")
        gallery = gr.Gallery(label="✂️ 裁切結果")
        out_txt = gr.Textbox(label="🌟 結果", lines=8)
        img_in.upload(calc.process, img_in, [gallery, out_txt])
    demo.launch(server_name="0.0.0.0", server_port=port)


if __name__ == "__main__":
    # CLI 用法：
    #   python brightness_yolo_gradio.py /path/to/image.png /path/to/model.pt
    #CLI 模式：提供 python brightness_yolo_gradio.py input.png model.pt，會在終端印出文字結果，方便從程式直接呼叫
    if len(sys.argv) == 3:
        img_path, model_path = sys.argv[1:]
        if not os.path.isfile(img_path):
            print(f"找不到影像: {img_path}")
            sys.exit(1)
        calc = YOLOBrightnessCalculator(model_path)
        img = cv2.imread(img_path)
        crops, text = calc.process(img)
        print(text)
    #UI 模式：若未傳入兩個參數，會預設讀 MODEL_PATH 並啟動 Gradio 介面。
    else:
        # 啟動 UI
        MODEL_PATH = "/app/best.pt"  # 修改為實際模型
        launch_ui(MODEL_PATH, port=8800)

