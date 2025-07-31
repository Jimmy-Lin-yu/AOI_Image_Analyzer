# u2net_cropper.py
import cv2
import numpy as np
from pathlib import Path
import PIL
from PIL import Image
from u2_net_inference import U2NetProcessor


class U2NetCropper:
    """
    使用 U²-Net 進行前景分割並切割影像

    參數：
        model_path (str): U²-Net 權重檔路徑
        min_area (int): 最小連通域面積，用於過濾雜訊
        kernel_size (tuple): 閉運算核大小
    """
    def __init__(self, model_path: str, min_area: int = 1000, kernel_size: tuple = (5, 5)):
        self.processor = U2NetProcessor(model_path)
        self.min_area = min_area
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)

    def crop_images(self, input_dir: str, output_dir: str) -> list[str]:
        """
        將 input_dir 下所有影像分割並存至 output_dir

        回傳：
            saved_paths (list[str]): 成功切割並儲存的影像路徑清單
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = []
        for img_path in input_dir.glob("*.*"):
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue

            # 1) 推論並二值化
            mask_float = self.processor.inference(img_bgr)            # [0,1]
            mask_uint8 = (mask_float * 255).astype(np.uint8)         # [0,255]
            _, bin_mask = cv2.threshold(mask_uint8, 127, 255, cv2.THRESH_BINARY)

            # 2) 閉運算去小孔
            bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, self.kernel, iterations=2)

            # 3) 保留最大連通域
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
            full_mask = np.zeros_like(bin_mask)
            if num_labels > 1:
                areas = stats[1:, cv2.CC_STAT_AREA]
                max_idx = 1 + int(np.argmax(areas))
                full_mask[labels == max_idx] = 255
            else:
                full_mask = bin_mask.copy()

            full_mask = cv2.bitwise_not(full_mask)
            # 4) 用遮罩切出前景 (背景黑)
            cut_bgr = cv2.bitwise_and(img_bgr, img_bgr, mask=full_mask)

            # ★★★ 4.1 取 bounding box，裁掉全黑背景 ★★★
            ys, xs = np.where(full_mask > 0)
            if ys.size and xs.size:
                y0, y1 = ys.min(), ys.max()
                x0, x1 = xs.min(), xs.max()
                # （可選）給 5px padding，避免太貼邊
                pad = 5
                y0 = max(0, y0 - pad); y1 = min(cut_bgr.shape[0]-1, y1 + pad)
                x0 = max(0, x0 - pad); x1 = min(cut_bgr.shape[1]-1, x1 + pad)
                cut_bgr = cut_bgr[y0:y1+1, x0:x1+1]

            # 5) 儲存切割結果
            base_name = img_path.stem
            new_name  = f"{base_name}_crop.jpg"
            print(f"處理 {new_name}，切割結果尺寸: {cut_bgr.shape}")
            out_path = output_dir / new_name
            # 轉換為 RGB 並以 PIL 存檔，避免 imwrite 格式錯誤
            cut_rgb = cv2.cvtColor(cut_bgr, cv2.COLOR_BGR2RGB)
            Image.fromarray(cut_rgb).save(out_path)
            saved_paths.append(str(out_path))

        return saved_paths
