import cv2
import numpy as np
import os

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
        "noise": (60, 40)
    }


    def __init__(self, folder_path: str = None):
        self.folder_path = folder_path

    @staticmethod
    def calculate_sharpness(image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    @staticmethod
    def calculate_exposure(image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray))

    @staticmethod
    def calculate_contrast(image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return float(np.std(gray))

    @staticmethod
    def calculate_light_uniformity(image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return float(np.std(gray))

    @staticmethod
    def calculate_noise(image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        return float(np.mean(np.abs(gray - blurred)))

#--------------------------------------------------------------------------------------
    @classmethod
    def evaluate_quality(cls, metrics: dict) -> tuple:
        """
        根據每項指標的閾值計算分數。
        每項評分規則：
          - 如果值 >= high threshold, score = 5
          - 如果值 >= mid threshold, score = 3
          - 否則 score = 1
        總品質 = (所有分數平均 / 5) * 100 (百分比)
        """
        scores = []
        for key in ["sharpness", "exposure", "contrast", "uniformity", "noise"]:
            value = metrics.get(key, 0)
            high, mid = cls.thresholds[key]
            if value >= high:
                score = 5
            elif value >= mid:
                score = 3
            else:
                score = 1
            scores.append(score)
        total = sum(scores) / len(scores) / 5 * 100
        return scores, total



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
                "noise": self.calculate_noise(image)
            }
            
            # 依據閾值計算分數與總品質
            scores, total = self.evaluate_quality(metrics)
            
            # 印出結果
            print(f"Image: {fname}")
            for key, score in zip(metrics.keys(), scores):
                print(f"  {key.capitalize()}: {metrics[key]:.2f} → Score {score}")
            print(f"  Total Quality: {total:.1f}%")
            print("-" * 30)

if __name__ == "__main__":
    # Change this to your images folder path
    folder = r"D:\Users\JimmyLin\Desktop\AOI\OpenCV\images"
    analyzer = ImageQualityAnalyzer(folder)
    analyzer.process_evaluate()

