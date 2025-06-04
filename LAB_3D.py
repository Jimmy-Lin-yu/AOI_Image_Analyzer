import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          
from skimage.color import rgb2lab, lab2rgb

# ===================== 1. 讀檔 =====================
df = pd.read_csv("ALL.csv")          # 必含 R,G,B (0–1023)

# ===================== 2. RGB10 → Lab =====================
rgb_norm = df[["R", "G", "B"]].to_numpy() / 1023.0      # 0-1
lab = rgb2lab(rgb_norm.reshape(-1, 1, 3)).reshape(-1, 3)
df[["L", "a", "b"]] = lab

# ===================== 3. 動態半徑 =====================
# 先求每筆 Lab 向量長度，再取最大值
R_data = np.ceil(np.sqrt((df[["a", "b", "L"]] ** 2).sum(1)).max() / 10) * 10
R_sphere = R_data      # 半徑 (例：170)

# ===================== 4. 產生半球網格 =====================
theta, phi = np.meshgrid(
    np.linspace(0, np.pi / 2, 120),       # 0 → 90°（上半球）
    np.linspace(0, 2 * np.pi, 240)        # 0 → 360°
)

a_s = R_sphere * np.sin(theta) * np.cos(phi)
b_s = R_sphere * np.sin(theta) * np.sin(phi)
L_s = R_sphere * np.cos(theta)

lab_surf = np.stack([L_s, a_s, b_s], axis=-1)
rgb_surf = np.clip(lab2rgb(lab_surf), 0, 1)

# ===================== 5. 繪圖 =====================
plt.close("all")
fig = plt.figure(figsize=(9, 8), facecolor="black")
ax = fig.add_subplot(111, projection="3d")
ax.set_facecolor("black")
ax.set_box_aspect([1, 1, 100 / (2 * R_sphere)])   # 保持三軸比例

# ---- 半透明色球 ----
ax.plot_surface(
    a_s, b_s, L_s,
    rstride=4, cstride=4,
    facecolors=rgb_surf,
    linewidth=0, antialiased=False,
    alpha=0.35
)

# ---- 參考軸 (±a*, ±b*, L*) ----
ax.plot([0,  R_sphere], [0, 0], [0, 0], color="red",   lw=2)    # +a*
ax.plot([0, -R_sphere], [0, 0], [0, 0], color="green", lw=2)    # −a*
ax.plot([0, 0], [0,  R_sphere], [0, 0], color="yellow", lw=2)   # +b*
ax.plot([0, 0], [0, -R_sphere], [0, 0], color="blue",  lw=2)    # −b*
ax.plot([0, 0], [0, 0], [0, 100],  color="white", lw=2)         # +L*

# ---- 點雲 ----
ax.scatter(
    df["a"], df["b"], df["L"],
    c=rgb_norm, s=20, edgecolors="none", alpha=0.9
)

# ---- 軸 & 標籤 ----
ax.set_xlim(-R_sphere, R_sphere)
ax.set_ylim(-R_sphere, R_sphere)
ax.set_zlim(0, 100)

ax.set_xlabel("a*  (−G → +R)", color="white")
ax.set_ylabel("b*  (−B → +Y)", color="white")
ax.set_zlabel("L*  (0→100)",    color="white")
ax.set_title("CIELAB", color="white", pad=15)

ax.tick_params(colors="white")
for name in ("xaxis", "yaxis", "zaxis"):
    getattr(ax, name)._axinfo["grid"]["color"] = (1, 1, 1, 0.25)

ax.view_init(elev=25, azim=35)          # 視角可自行微調

# ---- 輸出 ----
outfile = "CIELAB_color_sphere_fit.jpg"
plt.savefig(outfile, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.show()
print(f"已輸出：{outfile}")