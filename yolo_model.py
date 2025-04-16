from ultralytics import YOLO
import cv2
import os
from datetime import datetime
import time

class YOLOImageProcessor:
    def __init__(self, model_path, image_folder, output_folder, log_folder="logs"):
        """
        初始化 YOLOImageProcessor

        :param model_path: YOLO 模型權重路徑
        :param image_folder: 原始圖片所在的資料夾
        :param output_folder: 處理後圖片輸出的資料夾
        :param log_folder: 儲存 log 檔的資料夾，預設為 "logs"
        """
        self.model_path = model_path
        self.image_folder = image_folder
        self.output_folder = output_folder
        self.log_folder = log_folder
        
        os.makedirs(self.log_folder, exist_ok=True)
        os.makedirs(self.output_folder, exist_ok=True)
        self.log_path = os.path.join(self.log_folder, "inference.log")
        
        # 載入 YOLO 模型
        self.model = YOLO(self.model_path)
        print(f"模型目前使用的設備：{self.model.device}")
    
    def _log_inference(self, image_filename, start_time_str, end_time_str, duration):
        """
        寫入 log 訊息
        """
        log_message = (
            f"[{image_filename}]\n"
            f"開始時間: {start_time_str}\n"
            f"結束時間: {end_time_str}\n"
            f"耗時: {duration} 秒\n"
            f"{'-'*40}\n"
        )
        with open(self.log_path, "a") as f:
            f.write(log_message)
    
    def run_inference_with_timer(self, image_filename, func, *args, **kwargs):
        """
        執行推論並計時，並將執行細節寫入 log 檔

        :param image_filename: 當前處理的圖片檔名
        :param func: 執行推論的函數
        :return: 推論結果
        """
        start_time = datetime.now()
        start_time_str = start_time.strftime('%Y-%m-%d %H:%M:%S')
        start = time.time()
        
        result = func(*args, **kwargs)
        
        end = time.time()
        end_time = datetime.now()
        end_time_str = end_time.strftime('%Y-%m-%d %H:%M:%S')
        duration = round(end - start, 3)
        
        print(f"🕒 推論 {image_filename} 耗時：{duration} 秒")
        self._log_inference(image_filename, start_time_str, end_time_str, duration)
        
        return result

    def process_images(self):
        """
        處理 image_folder 中的所有圖片，
        執行 YOLO 推論、擷取第一個偵測到的物件框，並將裁切後的圖片存入 output_folder。
        """
        for image_filename in os.listdir(self.image_folder):
            if not image_filename.lower().endswith((".jpg", ".png", ".jpeg","bmp")):
                continue

            image_path = os.path.join(self.image_folder, image_filename)
            image = cv2.imread(image_path)
            if image is None:
                print(f"⚠️ 無法讀取 {image_filename}，跳過")
                continue
            h_img, w_img = image.shape[:2]

            # 執行推論，並對推論進行計時與記錄
            results = self.run_inference_with_timer(image_filename, self.model, image_path)

            for result in results:
                obb = result.obb

                if obb.xyxy.shape[0] == 0:
                    print(f"⚠️ {image_filename} 沒有偵測結果，跳過處理")
                    with open(self.log_path, "a") as f:
                        f.write(f"[{image_filename}] ❌ 沒有偵測結果，已跳過\n{'-'*40}\n")
                    continue

                # 以第一個偵測框作為示範（若需要處理所有框則可以加入迴圈）
                x1, y1, x2, y2 = obb.xyxy.cpu().numpy()[0].astype(int)

                # 防呆處理邊界值
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w_img, x2), min(h_img, y2)
                if x2 - x1 < 2 or y2 - y1 < 2:
                    print(f"⚠️ {image_filename} 裁切框過小，跳過")
                    continue

                # 裁切圖片並保持原圖畫質
                cropped_image = image[y1:y2, x1:x2]
                output_name = os.path.splitext(image_filename)[0] + "_crop.png"
                output_path = os.path.join(self.output_folder, output_name)
                cv2.imwrite(output_path, cropped_image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                print(f"✅ 已保存處理後圖片到: {output_path}")

        print("📦 所有圖片處理完成！")

# ────────── 執行入口 ──────────
if __name__ == "__main__":
    model_path = "/app/obb_runs/yolov8n-obb2/weights/best.pt"
    image_folder = "/app/dataset/YOLO_seg/Train/images"
    output_folder = "/app/inference_image"
    
    processor = YOLOImageProcessor(model_path, image_folder, output_folder)
    processor.process_images()
