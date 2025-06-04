import numpy as np
import matplotlib.pyplot as plt
from skimage.color import lab2rgb, rgb2lab

# -----------------------------------------------------------------------------
# 1) Define two inverse conversion functions:
#    (A) led10bit_to_lab(R10,G10,B10): convert 10‐bit LED to Lab
#    (B) lab_to_led10bit(L, a, b):      convert Lab to 10‐bit LED
#
#    Here we assume 10‐bit LED (0~1023) directly maps to sRGB [0..1] (with gamma):
#      R_lin = R10/1023, G_lin = G10/1023, B_lin = B10/1023
#    Then use skimage.color.rgb2lab / lab2rgb internally for the conversions.
# -----------------------------------------------------------------------------

def led10bit_to_lab(R10, G10, B10):
    """
    Input:   R10, G10, B10 (integers in 0..1023)
    Output:  (L*, a*, b*) in CIE‐Lab (float array of length 3)
    """
    # 1) Convert 0..1023 to 0..1
    rgb01 = np.array([R10, G10, B10], dtype=float) / 1023.0

    # 2) Use skimage.color.rgb2lab (sRGB → linear → XYZ → Lab)
    lab = rgb2lab(rgb01.reshape((1, 1, 3))).reshape((3,))
    return lab  # returns [L*, a*, b*]


def lab_to_led10bit(L, a, b):
    """
    Input:   L*, a*, b* (float)
    Output:  (R10, G10, B10) as 10‐bit integers 0..1023
    """
    # 1) Use skimage.color.lab2rgb to convert Lab → floating sRGB [0..1]
    rgb01 = lab2rgb(np.array([[[L, a, b]]]))  # shape = (1,1,3)
    rgb01 = rgb01.reshape((3,))

    # 2) Clip to [0..1]
    rgb01 = np.clip(rgb01, 0.0, 1.0)

    # 3) Multiply by 1023, round to nearest integer
    rgb10 = np.round(rgb01 * 1023.0).astype(int)
    return tuple(rgb10)  # (R10, G10, B10)


# ---------------------------------------------------------------------
# 2) Generate 8 representative hues:
#      - Fix L* = 50, C* = 100
#      - Hue angles: 0°, 45°, 90°, …, 315°
#      - Compute a* = C* cos(h), b* = C* sin(h)
#    Then convert each (L, a, b) → 10‐bit LED.
# ---------------------------------------------------------------------

def make_8_hue_representatives(L=50, C=100.0):
    """
    L: float, L* value in Lab (e.g. 50)
    C: float, chroma C* (here use 100 as example; may clip if outside sRGB gamut)
    Returns: list of 8 dicts:
        {
          'hue_deg': <hue angle in degrees>,
          'L': <L*>,
          'a': <a*>,
          'b': <b*>,
          'R10': <10bit R>,
          'G10': <10bit G>,
          'B10': <10bit B>
        }
    """
    reps = []
    for i in range(8):
        h_deg = i * 45.0
        h_rad = np.deg2rad(h_deg)
        a = C * np.cos(h_rad)
        b = C * np.sin(h_rad)
        R10, G10, B10 = lab_to_led10bit(L, a, b)
        reps.append({
            'hue_deg': h_deg,
            'L': L, 'a': a, 'b': b,
            'R10': int(R10), 'G10': int(G10), 'B10': int(B10)
        })
    return reps


# ---------------------------------------------------------------------
# 3) Main: call the function and print out results; then plot the circle
# ---------------------------------------------------------------------
if __name__ == "__main__":
    reps = make_8_hue_representatives(L=50, C=50)

    print("── 8 Hue Representatives (Lab → 10‐bit LED) ──")
    print("  hue°    L*     a*      b*    →   R10   G10   B10")
    print("---------------------------------------------------")
    for r in reps:
        print(
            f"  {r['hue_deg']:>3.0f}°   "
            f"{r['L']:6.2f}  {r['a']:7.2f}  {r['b']:7.2f}   →  "
            f"{r['R10']:4d}  {r['G10']:4d}  {r['B10']:4d}"
        )

    # Also plot the a*‐b* plane with the 8 representatives highlighted
    fig, ax = plt.subplots(figsize=(6, 6), facecolor='black')
    ax.set_facecolor('black')

    # Plot continuous hue circle (L*=50, C*=100) for reference
    thetas = np.linspace(0, 2*np.pi, 360)
    aa = 100 * np.cos(thetas)
    bb = 100 * np.sin(thetas)
    lab_circle = np.stack([50.0*np.ones_like(aa), aa, bb], axis=-1)  # shape = (360,3)
    rgb_circle = lab2rgb(lab_circle.reshape((-1,1,3))).reshape((-1,3))
    rgb_circle = np.clip(rgb_circle, 0, 1)
    ax.scatter(aa, bb, s=12, c=rgb_circle, alpha=0.6, marker='o')

    # Plot the 8 representative points with white outline
    for r in reps:
        ax.scatter(
            r['a'], r['b'],
            s=200,
            c=np.array([[r['R10'], r['G10'], r['B10']]], dtype=float)/1023,
            edgecolors='white', linewidths=1.2
        )
        ax.text(
            r['a']*1.05, r['b']*1.05,
            f"{int(r['hue_deg'])}°",
            color='white', fontsize=10, ha='center', va='center'
        )

    # Axis settings
    ax.axhline(0, color='white', linewidth=0.8)
    ax.axvline(0, color='white', linewidth=0.8)
    ax.set_xlim(-110, 110)
    ax.set_ylim(-110, 110)
    ax.set_xlabel("a* (−G → +R)", color='white')
    ax.set_ylabel("b* (−B → +Y)", color='white')
    ax.set_title("CIELAB Hue Circle (L*=50, C*=50) & 8 Representatives (10‐bit LED)",
                 color='white')
    ax.tick_params(colors='white')
    ax.set_aspect('equal', 'box')
    plt.tight_layout()

    output_path = 'Hue_Circle_8_Representatives2.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.show()

    print(f"Saved image file: {output_path}")
