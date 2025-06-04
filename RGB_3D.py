import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# -------------------------------
# 讀取資料 (若已有 RGB_data.csv，可取消下面模擬區塊)
# -------------------------------

df = pd.read_csv('ALL.csv')  # CSV 中需包含 R, G, B 三個欄位


# -------------------------------
# 嚴格分類：只有恰好等於 0 才歸為對應平面
# -------------------------------
mask_RG  = (df['R'] > 0) & (df['G'] > 0) & (df['B'] == 0)  # B = 0 → RG plane
mask_GB  = (df['G'] > 0) & (df['B'] > 0) & (df['R'] == 0)  # R = 0 → GB plane
mask_RB  = (df['R'] > 0) & (df['B'] > 0) & (df['G'] == 0)  # G = 0 → RB plane
mask_RGB = (df['R'] > 0) & (df['G'] > 0) & (df['B'] > 0)   # all > 0 → Full RGB
mask_Others = ~(mask_RG | mask_GB | mask_RB | mask_RGB)  # at least two coords = 0

# -------------------------------
# 印出各分類筆數，確認分類是否正確
# -------------------------------
print("RG plane (R>0, G>0, B=0)   :", mask_RG.sum(), "points")
print("GB plane (G>0, B>0, R=0)   :", mask_GB.sum(), "points")
print("RB plane (R>0, B>0, G=0)   :", mask_RB.sum(), "points")
print("Full RGB   (R>0,G>0,B>0)   :", mask_RGB.sum(), "points")
print("Others (two coords = 0)    :", mask_Others.sum(), "points\n")

# -------------------------------
# 繪製 3D 散佈圖：黑底背景、白色文字與網格線
# -------------------------------
plt.close('all')
fig = plt.figure(figsize=(8, 7))
fig.patch.set_facecolor('black')  # 整張畫布背景設為黑色

ax = fig.add_subplot(111, projection='3d')

# 設定 3D 面板 (x-y, y-z, x-z) 背景為純黑
ax.xaxis.pane.set_facecolor((0, 0, 0, 1))
ax.yaxis.pane.set_facecolor((0, 0, 0, 1))
ax.zaxis.pane.set_facecolor((0, 0, 0, 1))
ax.set_facecolor((0, 0, 0, 1))

# 將網格線改為白色 (半透明)，並打開網格
ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0.3)
ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0.3)
ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0.3)
ax.grid(True)

# 坐標刻度文字與標籤改為白色
ax.tick_params(colors='white')
ax.xaxis.label.set_color('white')
ax.yaxis.label.set_color('white')
ax.zaxis.label.set_color('white')
ax.title.set_color('white')

# 坐標軸刻度線也改為白色
for line in ax.xaxis.get_ticklines() + ax.yaxis.get_ticklines() + ax.zaxis.get_ticklines():
    line.set_color('white')

# 坐標軸邊框也改成白色
for spine in ax.spines.values():
    spine.set_color('white')

# 繪製各分類點 (依照嚴格 0 條件)
# 1. RG 平面 (B=0)：黃色
if mask_RG.any():
    ax.scatter(
        df.loc[mask_RG, 'R'],
        df.loc[mask_RG, 'G'],
        df.loc[mask_RG, 'B'],
        color='yellow',
        edgecolors='black',
        s=50,
        label='RG plane (B=0)'
    )

# 2. GB 平面 (R=0)：青色
if mask_GB.any():
    ax.scatter(
        df.loc[mask_GB, 'R'],
        df.loc[mask_GB, 'G'],
        df.loc[mask_GB, 'B'],
        color='cyan',
        edgecolors='black',
        s=50,
        label='GB plane (R=0)'
    )

# 3. RB 平面 (G=0)：紫色
if mask_RB.any():
    ax.scatter(
        df.loc[mask_RB, 'R'],
        df.loc[mask_RB, 'G'],
        df.loc[mask_RB, 'B'],
        color='purple',
        edgecolors='black',
        s=50,
        label='RB plane (G=0)'
    )

# 4. Full RGB (all > 0)：白色
if mask_RGB.any():
    ax.scatter(
        df.loc[mask_RGB, 'R'],
        df.loc[mask_RGB, 'G'],
        df.loc[mask_RGB, 'B'],
        color='white',
        edgecolors='black',
        s=30,
        alpha=0.9,
        label='Full RGB (all > 0)'
    )

# 5. Others (two coords = 0)：灰色
if mask_Others.any():
    ax.scatter(
        df.loc[mask_Others, 'R'],
        df.loc[mask_Others, 'G'],
        df.loc[mask_Others, 'B'],
        color='gray',
        edgecolors='black',
        s=30,
        alpha=0.6,
        label='Others (two coords = 0)'
    )

# 設定坐標標籤與標題 (皆為英文，避免中文字缺字)
ax.set_xlabel('R')
ax.set_ylabel('G')
ax.set_zlabel('B')
ax.set_title('3D Scatter: RG / RB / GB / RGB Classification')

# 固定視角 (仰角 elev=25°, 水平 azim=45°)
ax.view_init(elev=25, azim=45)

# 圖例設置：背景深灰、文字白色
legend = ax.legend(loc='upper left', fontsize=9)
legend.get_frame().set_facecolor((0.2, 0.2, 0.2, 0.8))
for text in legend.get_texts():
    text.set_color('white')

# 最後存檔 (黑底 + 所有文字 & 線條為白色)
output_path = '3D_categorical_colors_strict_blackbg.jpg'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.show()

print(f'已儲存圖檔：{output_path}')
