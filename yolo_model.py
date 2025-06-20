from ultralytics import YOLO
import cv2
import os
from datetime import datetime
import time
import numpy as np
class YOLOImageProcessor:
    def __init__(self, image_folder, output_folder, log_folder="logs",device='cuda:0'):
        """
        初始化 YOLOImageProcessor

        :param model_path: YOLO 模型權重路徑
        :param image_folder: 原始圖片所在的資料夾
        :param output_folder: 處理後圖片輸出的資料夾
        :param log_folder: 儲存 log 檔的資料夾，預設為 "logs"
        """
        self.model_path    = "/app/best.pt"
        self.image_folder = image_folder
        self.output_folder = output_folder
        self.log_folder = log_folder
        self.device = device
        
        os.makedirs(self.log_folder, exist_ok=True)
        os.makedirs(self.output_folder, exist_ok=True)
        self.log_path = os.path.join(self.log_folder, "inference.log")
        
        # 載入 YOLO 模型
        self.model = YOLO(self.model_path)
        self.model.to(self.device)   
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

    # def process_images(self):
    #     """
    #     處理 image_folder 中的所有圖片，
    #     執行 YOLO 推論、擷取第一個偵測到的物件框（axis-aligned bbox），
    #     以 4 個點形式記錄，並將裁切後的圖片存入 output_folder。
    #     """
    #     for image_filename in os.listdir(self.image_folder):
    #         if not image_filename.lower().endswith((".jpg", ".png", ".jpeg", "bmp")):
    #             continue

    #         image_path = os.path.join(self.image_folder, image_filename)
    #         image = cv2.imread(image_path)
    #         if image is None:
    #             print(f"⚠️ 無法讀取 {image_filename}，跳過")
    #             continue
    #         h_img, w_img = image.shape[:2]

    #         # 執行推論並記時
    #         results = self.run_inference_with_timer(image_filename, self.model, image_path)

    #         for result in results:
    #             # 1. 取得所有 axis-aligned bbox (N×4)
    #             bboxes = result.boxes.xyxy.cpu().numpy()  # [[x1,y1,x2,y2], ...]
    #             if bboxes.shape[0] == 0:
    #                 print(f"⚠️ {image_filename} 沒有偵測結果，跳過處理")
    #                 with open(self.log_path, "a") as f:
    #                     f.write(f"[{image_filename}] ❌ 沒有偵測結果，已跳過\n{'-'*40}\n")
    #                 continue

    #             # 2. 逐一處理每個 bbox
    #             for idx, (x1, y1, x2, y2) in enumerate(bboxes):
    #                 # 四個角點，順時針或逆時針都可以
    #                 corners = [
    #                     [int(x1), int(y1)],
    #                     [int(x1), int(y2)],
    #                     [int(x2), int(y2)],
    #                     [int(x2), int(y1)],
    #                 ]
    #                 print(f"🗺️ {image_filename} 框 {idx} 的 4 點：{corners}")
    #                 with open(self.log_path, "a") as f:
    #                     f.write(f"[{image_filename}] 框 {idx} 的 4 點：{corners}\n")

    #                 # 3. 防呆：邊界裁切
    #                 x1_i, y1_i = max(0, corners[0][0]), max(0, corners[0][1])
    #                 x2_i, y2_i = min(w_img, corners[2][0]), min(h_img, corners[2][1])
    #                 if x2_i - x1_i < 2 or y2_i - y1_i < 2:
    #                     print(f"⚠️ {image_filename} 框 {idx} 太小，跳過裁切")
    #                     continue

    #                 # 4. 裁切並存檔
    #                 cropped = image[y1_i:y2_i, x1_i:x2_i]
    #                 base, _ = os.path.splitext(image_filename)
    #                 output_name = f"{base}_crop{idx}.png"
    #                 output_path = os.path.join(self.output_folder, output_name)
    #                 cv2.imwrite(output_path, cropped, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    #                 print(f"✅ 已保存 {image_filename} 框 {idx} 裁切圖到: {output_path}")

    #     print("📦 所有圖片處理完成！")

    def process_images(self):
        import numpy as np, cv2, os

        for image_filename in os.listdir(self.image_folder):
            if not image_filename.lower().endswith((".jpg", ".png", ".jpeg", "bmp")):
                continue

            # 讀原圖
            image_path = os.path.join(self.image_folder, image_filename)
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if image is None:
                print(f"⚠️ 無法讀取 {image_filename}，跳過")
                continue
            h_img, w_img = image.shape[:2]

            # 推論
            results = self.run_inference_with_timer(image_filename, self.model, image_path)
            for result in results:
                if result.masks is None or len(result.masks.xy) == 0:
                    continue

                # 每一個 instance
                for idx, poly in enumerate(result.masks.xy):
                    pts = poly.astype(np.int32)   # (N,2)

                    # 1) 在整圖上畫遮罩
                    mask_full = np.zeros((h_img, w_img), dtype=np.uint8)
                    cv2.fillPoly(mask_full, [pts], 255)

                    # 2) 用遮罩保留物件區域，其餘變黑
                    masked = cv2.bitwise_and(image, image, mask=mask_full)

                    # 3) 用 boundingRect 裁切出 ROI
                    x, y, w, h = cv2.boundingRect(pts)
                    if w < 2 or h < 2:
                        continue
                    cropped = masked[y:y+h, x:x+w]

                    # 4) 存檔
                    base, _ = os.path.splitext(image_filename)
                    output_name = f"{base}_poly{idx}_crop.jpg"
                    output_path = os.path.join(self.output_folder, output_name)
                    cv2.imwrite(output_path, cropped)
                    print(f"✅ 已保存多邊形切割 {image_filename} instance {idx} 到: {output_path}")

        print("📦 所有圖片處理完成！")
# ────────── 執行入口 ──────────
if __name__ == "__main__":
    model_path = "/app/obb_runs/yolov8n-obb2/weights/best.pt"
    image_folder = "/app/dataset/YOLO_seg/Train/images"
    output_folder = "/app/inference_image"
    
    processor = YOLOImageProcessor(image_folder, output_folder)
    processor.process_images()
