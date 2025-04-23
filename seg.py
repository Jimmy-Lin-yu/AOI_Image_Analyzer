import cv2
import numpy as np

# def segment_object(img: np.ndarray,
#                    canny_thresh1=50, canny_thresh2=150,
#                    morph_kernel=(5,5)) -> np.ndarray:
#     """
#     對輸入 BGR 圖做邊緣切割，回傳一張遮罩 (mask) 以及裁切後的 ROI。
#     """
#     # 1. 灰階 + 高斯模糊
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(gray, (5,5), 0)
    
#     # 2. Canny 邊緣檢測
#     edges = cv2.Canny(blur, canny_thresh1, canny_thresh2)
    
#     # 3. 形態學閉運算：連起散碎邊緣
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, morph_kernel)
#     closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    
#     # 4. 找輪廓，挑面積最大的當作物件
#     contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if not contours:
#         raise RuntimeError("No contours found!")
#     # 依面積排序，最大輪廓
#     cnt = max(contours, key=cv2.contourArea)
    
#     # 5. 建遮罩並擷取 bounding box
#     mask = np.zeros_like(gray, dtype=np.uint8)
#     cv2.drawContours(mask, [cnt], -1, 255, cv2.FILLED)
    
#     x,y,w,h = cv2.boundingRect(cnt)
#     roi = img[y:y+h, x:x+w]
    
#     return mask, roi
def segment_object_white_bg(img: np.ndarray,
                            canny_thresh1=50, canny_thresh2=150,
                            morph_kernel=(5,5)) -> np.ndarray:
    """
    對輸入 BGR 圖做邊緣切割，
    回傳一張輸出圖：物件區域保留原圖，其餘區域填滿白色。
    """
    # 1. 灰階 + 高斯模糊
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    
    # 2. Canny 邊緣檢測
    edges = cv2.Canny(blur, canny_thresh1, canny_thresh2)
    
    # 3. 形態學閉運算：連起散碎邊緣
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, morph_kernel)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 4. 找輪廓，挑面積最大的當作物件
    contours, _ = cv2.findContours(closed,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("No contours found!")
    cnt = max(contours, key=cv2.contourArea)
    
    # 5. 建 single‐channel mask
    mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, cv2.FILLED)
    
    # 6. 生成三通道遮罩，並把 mask == 0 區域填白
    result = img.copy()
    result[mask == 0] = (255, 255, 255)
    
    return result
if __name__ == "__main__":
    img = cv2.imread("/app/1.jpg")
    # mask, roi = segment_object(img)
    out = segment_object_white_bg(img)
    # cv2.imwrite("3.jpg", mask)
    # cv2.imwrite("4.jpg", roi)
    cv2.imwrite("5.jpg", out)