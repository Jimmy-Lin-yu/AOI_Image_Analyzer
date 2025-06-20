import cv2
import numpy as np
import os
import base64
class ImageQualityAnalyzer:
    """
    A class to compute image quality metrics and analyze single images or folders of images.
    """
    # 定義各項品質指標的閾值
    thresholds = {
        "sharpness": (100, 50),
        "exposure": (100, 80),
        "contrast": (50, 30),
        "uniformity": (70, 50),
        # "noise": (60, 40),
        # "defect": (100, 50)
    }


    def __init__(self, folder_path: str ):
        if not os.path.isdir(folder_path):
            raise ValueError(f"Invalid folder path: {folder_path}")
        self.folder_path = folder_path

#-----------------------------------------銳利度(sharpness)----------------------------------------------
    # @staticmethod
    # def calculate_sharpness(image: np.ndarray) -> float:
    #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #     return cv2.Laplacian(gray, cv2.CV_64F).var()
    
    @staticmethod
    def calculate_sharpness(image: np.ndarray,
                             method: str = 'variance_of_laplacian',
                             defect_mask: np.ndarray = None) -> float:
        """
        計算影像銳利度（越大越銳利），支援：
          1. variance_of_laplacian：Laplacian 變異數
          2. sobel_mean_magnitude：Sobel 梯度平均幅值
          3. tenengrad：Sobel 梯度平方和（Tenengrad 能量）
          
        可以傳入 defect_mask（二值遮罩，瑕疵區為 True）來排除結構性黑點。
        """
        # 轉灰階並去小噪
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # helper：根據 mask 排除瑕疵區的函式
        def _mask_out(arr):
            flat = arr.ravel()
            if defect_mask is not None:
                mask_flat = (~defect_mask).ravel()
                flat = flat[mask_flat]
            return flat

        if method == 'variance_of_laplacian':
            lap = cv2.Laplacian(gray, cv2.CV_64F)
            vals = _mask_out(lap)
            # Var[L(x,y)] 排除瑕疵區
            return float(np.var(vals))

        elif method == 'sobel_mean_magnitude':
            gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            mag = np.sqrt(gx*gx + gy*gy)
            vals = _mask_out(mag)
            # E[|∇I|]
            return float(np.mean(vals))

        elif method == 'tenengrad':
            gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            energy = gx*gx + gy*gy
            vals = _mask_out(energy)
            # ∑(Gx² + Gy²)
            return float(np.sum(vals))

        else:
            raise ValueError(f"Unknown sharpness method: {method}")
#-----------------------------------------曝光（exposure）----------------------------------------------
    # @staticmethod
    # def calculate_exposure(image: np.ndarray) -> float:
    #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #     return float(np.mean(gray))

    @staticmethod
    def calculate_exposure(image: np.ndarray,
                            ideal_min: float = 80,
                            ideal_max: float = 170,
                            weight_over: float = 0.5,
                            weight_under: float = 0.5) -> float:
        """
        計算曝光品質分數（0–100）：
          1. 先以「與理想平均值範圍」的偏差給出基礎分數 base_score：
               base_score = max(0, 1 - |mean_gray - mid| / ((max-min)/2)) * 100
          2. 扣除過曝比例與欠曝比例的懲罰：
               score = base_score
                       - weight_over  * (over_ratio  * 100)
                       - weight_under * (under_ratio * 100)
          3. 最後將 score 限制在 [0,100] 區間內。
        
        參數：
          ideal_min, ideal_max：理想灰階平均值範圍
          weight_over：過曝懲罰權重（0–1）
          weight_under：欠曝懲罰權重（0–1）
        """
        # 1. 轉灰階並計算平均亮度
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        mean_gray = float(np.mean(gray))

        # 2. 計算過曝／欠曝比例
        total = gray.size
        over_ratio  = np.count_nonzero(gray > 250) / total
        under_ratio = np.count_nonzero(gray <   5) / total

        # 3. 基礎分數：偏差越小越接近 100 分
        mid       = (ideal_min + ideal_max) / 2.0
        half_range = (ideal_max - ideal_min) / 2.0
        deviation = abs(mean_gray - mid)
        deviation_score = max(0.0, 1.0 - deviation / half_range) * 100.0

        # 4. 扣除過／欠曝懲罰
        penalty = weight_over  * (over_ratio  * 100.0) \
                + weight_under * (under_ratio * 100.0)
        score = deviation_score - penalty

        # 5. 限制在 [0,100]
        return float(max(0.0, min(score, 100.0)))   

    
    
#-----------------------------------------對比度(contrast)----------------------------------------------
    # @staticmethod
    # def calculate_contrast(image: np.ndarray) -> float:
    #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #     return float(np.std(gray))
    
    # @staticmethod
    # def calculate_contrast(image: np.ndarray) -> float:
    #    """全圖灰階標準差 + 動態範圍 (max–min) 的混合對比度指標"""
    #    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    #    std_contrast = float(np.std(gray))
    #    dynamic_range = float(gray.max() - gray.min())
    #    # 你可以根據需求給二者不同權重，這裡以各 50% 為例：
    #    return 0.5 * std_contrast + 0.5 * dynamic_range


    @staticmethod
    def calculate_contrast(image: np.ndarray,
                           blocks: int = 8,
                           α_global: float = 0.5,
                           α_local_mean: float = 0.25,
                           α_local_min: float = 0.25) -> float:
        """
        同時計算：
          - 全圖 RMS 對比度（標準差）
          - 區塊 RMS 對比度的平均值與最小值
        最後以加權平均的方式合併三者：
          C = α_global * global_std
            + α_local_mean * mean(local_stds)
            + α_local_min  * min(local_stds)
        
        參數：
          blocks：切成 blocks×blocks 小區塊
          α_*：各指標的權重，預設各佔 50%、25%、25%
        """
        # 1. 全圖 RMS 對比度
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        global_std = float(np.std(gray))

        # 2. 區塊 RMS 對比度
        h, w = gray.shape
        bh, bw = h // blocks, w // blocks
        local_stds = []
        for i in range(blocks):
            for j in range(blocks):
                y0, x0 = i * bh, j * bw
                y1 = (i + 1) * bh if i < blocks - 1 else h
                x1 = (j + 1) * bw if j < blocks - 1 else w
                block = gray[y0:y1, x0:x1]
                local_stds.append(block.std())

        mean_local = float(np.mean(local_stds))
        min_local  = float(np.min(local_stds))

        # 3. 加權合併
        contrast_score = (
            α_global      * global_std +
            α_local_mean  * mean_local +
            α_local_min   * min_local
        )
        return contrast_score

#-----------------------------------------均勻度(uniformity)----------------------------------------------
    # @staticmethod
    # def calculate_light_uniformity(image: np.ndarray) -> float:
    #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #     return float(np.std(gray))
    
    @staticmethod
    def calculate_light_uniformity(image: np.ndarray, blocks: int = 4) -> float: #可把影像切成 4×4 或 8×8 的小區塊
        """以 blocks×blocks 小區塊平均亮度的標準差，衡量均勻度"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        h, w = gray.shape
        bh, bw = h // blocks, w // blocks

        means = []
        for i in range(blocks):
            for j in range(blocks):
                y0, x0 = i * bh, j * bw
                # 最後一列／行若尺寸不整除，就取到尾
                y1 = (i + 1) * bh if i < blocks - 1 else h
                x1 = (j + 1) * bw if j < blocks - 1 else w
                block = gray[y0:y1, x0:x1]
                means.append(block.mean())

        return float(np.std(means))


#-----------------------------------------雜訊(noise)----------------------------------------------
    # @staticmethod
    # def calculate_noise(image: np.ndarray) -> float:
    #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    #     return float(np.mean(np.abs(gray - blurred)))


#---------------------------------------黑點瑕疵(defect)-----------------------------------------------
    # @staticmethod
    # def _segment_roi(gray: np.ndarray,
    #                  canny_thresh1=50, canny_thresh2=150,
    #                  morph_kernel=(5,5)) -> np.ndarray:
    #     """
    #     由灰階圖建立一個 single‐channel ROI mask (bool array)，
    #     物件內部為 True，背景為 False。
    #     """
    #     blur = cv2.GaussianBlur(gray, (5,5), 0)
    #     edges = cv2.Canny(blur, canny_thresh1, canny_thresh2)
    #     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, morph_kernel)
    #     closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    #     contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     if not contours:
    #         # 找不到就整圖當 ROI
    #         return np.ones_like(gray, dtype=bool)
    #     cnt = max(contours, key=cv2.contourArea)
    #     mask = np.zeros_like(gray, dtype=np.uint8)
    #     cv2.drawContours(mask, [cnt], -1, 255, cv2.FILLED)
    #     return mask.astype(bool)

    # @staticmethod
    # def calculate_defect_score(image: np.ndarray,
    #                            dark_thresh: int = 50,
    #                            max_defect_ratio: float = 0.05,
    #                            border: int = 20) -> float:
    #     """
    #     把外圈和背景切成白，僅在 ROI 裡面偵測黑點瑕疵，
    #     缺陷越多分數越高 (0–100)。
    #     """
    #     # 0. 灰階
    #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #     h, w = gray.shape

    #     # 1. 分割 ROI
    #     obj_mask = ImageQualityAnalyzer._segment_roi(gray)

    #     # 2. 擴一點 ROI，並剔除圖邊
    #     obj_mask[:border, :] = False
    #     obj_mask[-border:, :] = False
    #     obj_mask[:, :border] = False
    #     obj_mask[:, -border:] = False

    #     # 3. 在 ROI 內找黑點 (灰階值 small than threshold)
    #     defects = (gray < dark_thresh) & obj_mask

    #     # 4. 去除貼邊的小 blob（可選）
    #     num, labels, stats, _ = cv2.connectedComponentsWithStats(defects.astype(np.uint8), connectivity=8)
    #     clean = np.zeros_like(defects)
    #     for lab in range(1, num):
    #         x,y,ww,hh,area = stats[lab]
    #         # 只保留不貼邊的小瑕疵
    #         if x>0 and y>0 and x+ww< w and y+hh< h:
    #             clean[labels==lab] = True

    #     # 5. 計算缺陷比例與分數
    #     roi_area = obj_mask.sum() or 1
    #     ratio    = clean.sum() / float(roi_area)
    #     norm     = min(ratio, max_defect_ratio) / max_defect_ratio
    #     score    = norm * 100.0
    #     return float(score)



#---------------------------------------------------------------------------------------------
#  
    @classmethod
    def evaluate_quality(cls, metrics: dict) -> tuple:
        """
        Compute continuous scores based on the metric values using piecewise linear interpolation.
        
        For each metric:
          - If value <= mid threshold: Score = 1 + (value/mid_threshold) * 2     (Score from 1 to 3)
          - If mid threshold < value < high threshold: 
                Score = 3 + ((value - mid_threshold) / (high_threshold - mid_threshold)) * 2  
                (Score from 3 to 5)
          - If value >= high threshold: Score = 5
          
        Overall Quality (%) = (Average Score / 5) * 100
        """
        scores = []
        for key in ["sharpness", "exposure", "contrast", "uniformity"]:
            value = metrics.get(key, 0)
            high, mid = cls.thresholds[key]
            
            if value >= high:
                score = 5.0
            elif value <= mid:
                # Linear interpolation between 1 and 3 when value ranges from 0 to mid
                score = 1.0 + (value / mid) * 2.0
            else:
                # Linear interpolation between 3 and 5 when value ranges from mid to high
                score = 3.0 + ((value - mid) / (high - mid)) * 2.0
            scores.append(score)
        
        total_quality = sum(scores) / (len(scores) * 5.0) * 100.0
        return scores, total_quality


    def process_evaluate(self, folder_path: str = None) -> None:
        """讀取資料夾內所有符合副檔名的圖片，計算各項品質指標，並印出結果"""
        path = folder_path or self.folder_path
        if not path or not os.path.isdir(path):
            raise ValueError(f"Invalid folder path: {path}")
        
        for fname in os.listdir(path):
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                continue
            
            full_path = os.path.join(path, fname)
            image = cv2.imread(full_path)
            if image is None or image.size == 0:
                print(f"Skipping invalid or empty image: {full_path}")
                continue
            
            # 計算各項品質指標
            metrics = {
                "sharpness": self.calculate_sharpness(image),
                "exposure": self.calculate_exposure(image),
                "contrast": self.calculate_contrast(image),
                "uniformity": self.calculate_light_uniformity(image),
                #"noise": self.calculate_noise(image),
                # "defect": self.calculate_defect_score(image)
            }
            
            # 依據閾值計算分數與總品質
            scores, total = self.evaluate_quality(metrics)
            
            # 印出結果
            print(f"Image: {fname}")
            for key, score in zip(metrics.keys(), scores):
                print(f"  {key.capitalize()}: {metrics[key]:.2f} → Score {score}")
            print(f"  Total Quality: {total:.1f}%")
            print("-" * 30)

    def process_auto_evaluate(self) -> list[dict]:

        records = []
        for fname in os.listdir(self.folder_path):
            if not fname.lower().endswith(("_crop.jpg","_crop.png", "_crop.jpeg")):
                continue
            full = os.path.join(self.folder_path, fname)
            image = cv2.imread(full)
            if image is None:
                continue

            metrics = {
                "sharpness":   self.calculate_sharpness(image),
                "exposure":    self.calculate_exposure(image),
                "contrast":    self.calculate_contrast(image),
                "uniformity":  self.calculate_light_uniformity(image),
            }
            _, total = self.evaluate_quality(metrics)

            # encode 裁切後圖
            _, buf = cv2.imencode(".jpg", image)
            b64img = base64.b64encode(buf).decode()

            records.append({
                "filename":      fname,
                **metrics,
                "total_quality": total,
                "cropped_image": b64img
            })
        return records


if __name__ == "__main__":
    # Change this to your images folder path
    analyzer = ImageQualityAnalyzer(folder)
    analyzer.process_evaluate()

