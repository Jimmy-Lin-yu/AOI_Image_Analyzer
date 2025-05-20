import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score


class BrightnessRegression:
    """Utility class for WRGB → Brightness regression & visualisation."""

    # ------------------------------------------------------------------
    # 1. Constructor & basic fitting utilities
    # ------------------------------------------------------------------
    def __init__(self, csv_path: str | Path):
        self.df = pd.read_csv(csv_path)
        required = {"W", "R", "G", "B", "Br"}
        missing = required.difference(self.df.columns)
        if missing:
            raise ValueError(f"CSV 缺少欄位: {', '.join(missing)}")

        # 保留 WRGB 至少有一通道 > 0 的列
        mask = (self.df[["W", "R", "G", "B"]].sum(axis=1) > 0)
        self.df = self.df.loc[mask].reset_index(drop=True)
        if self.df.empty:
            raise ValueError("過濾後沒有任何 WRGB 有效資料可用")

        self._models: dict[int, Pipeline] = {}
        self._scores: dict[int, float] = {}

    def prepare_and_fit(self, degree: int = 1):
        """Fit global WRGB→Br polynomial model and return (model, R²)."""
        if degree not in (1, 2):
            raise ValueError("degree 目前僅支援 1 或 2")

        X = self.df[["W", "R", "G", "B"]].values
        y = self.df["Br"].values

        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
                ("reg", LinearRegression()),
            ]
        )
        model.fit(X, y)
        r2 = model.score(X, y)
        self._models[degree] = model
        self._scores[degree] = r2
        return model, r2

    # ------------------------------------------------------------------
    # 2. Auto‑fit helper – try deg‑1 then deg‑2 if needed, then plot
    # ------------------------------------------------------------------
    def auto_fit_and_plot(
        self,
        channel: str = "auto",
        figsize_res=(6, 4),
        figsize_uni=(7, 4),
        out_dir: str | Path | None = None,
    ) -> dict[str, str]:
        """Convenience wrapper:
        1. Fit degree‑1; if R² < 0.99 then also fit degree‑2 and keep the better.
        2. Draw residual & univariate‑fit plots with the chosen degree.

        Returns a dict with keys: {'degree', 'r2', 'residual_png', 'univariate_png'}"""
        # 1. try degree‑1
        _, r2_1 = self.prepare_and_fit(1)
        chosen_deg = 1
        chosen_r2 = r2_1
        if r2_1 < 0.99:
            _, r2_2 = self.prepare_and_fit(2)
            if r2_2 > r2_1:
                chosen_deg, chosen_r2 = 2, r2_2

                # 2. make plots
        out_dir = Path(out_dir) if out_dir else Path(tempfile.mkdtemp())
        res_png = self.plot_residuals(degree=chosen_deg, channel=channel,
                                      figsize=figsize_res,
                                      save_path=out_dir / f"residual_deg{chosen_deg}.png")

        uni_path = out_dir / f"univariate_deg{chosen_deg}.png"
        self.plot_univariate_fit(channel=channel, degree=chosen_deg,
                                 figsize=figsize_uni,
                                 save_path=uni_path)
        uni_png = str(uni_path)

        return {
            "degree": chosen_deg,
            "r2": chosen_r2,
            "residual_png": res_png,
            "univariate_png": uni_png,
        }


    # ------------------------------------------------------------------
    # 3. Degree comparison (global model)
    # ------------------------------------------------------------------
    def compare_degrees(self, degrees=(1, 2), cv: int = 5):
        """Cross‑validate degree 1/2 and print mean R²."""
        X = self.df[["W", "R", "G", "B"]].values
        y = self.df["Br"].values
        out = {}
        for d in degrees:
            pipe = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("poly", PolynomialFeatures(d, include_bias=False)),
                    ("reg", LinearRegression()),
                ]
            )
            out[d] = cross_val_score(pipe, X, y, cv=cv, scoring="r2").mean()

        print("─" * 40)
        print("回歸階數比較 (交叉驗證 R²)")
        for d, sc in sorted(out.items(), key=lambda kv: kv[1], reverse=True):
            print(f"  次數 {d}: R² = {sc:.4f}")
        print("─" * 40)
        return out

    # ------------------------------------------------------------------
    # 3. Residual plot (per‑channel)
    # ------------------------------------------------------------------
    def plot_residuals(
        self,
        degree: int = 1,
        channel: str = "auto",
        figsize=(6, 4),
        save_path: str | Path | None = None,
    ) -> str:
        """Residual vs channel intensity (auto‑select channel if needed)."""
        model, _ = self.prepare_and_fit(degree)
        residuals = self.df["Br"].values - model.predict(
            self.df[["W", "R", "G", "B"]].values
        )

        channels = ["W", "R", "G", "B"]
        if channel == "auto" or channel not in channels or not (self.df[channel] > 0).any():
            channel = next((ch for ch in channels if (self.df[ch] > 0).any()), None)
            if channel is None:
                raise ValueError("找不到任何通道有大於0的資料")

        x = self.df[channel].values

        if save_path is None:
            tmp = tempfile.mkdtemp()
            save_path = Path(tmp) / f"residuals_vs_{channel}_deg{degree}.png"
        else:
            save_path = Path(save_path)

        plt.figure(figsize=figsize)
        plt.scatter(x, residuals, s=36, c="#71A6FF", edgecolors="none", alpha=0.7)
        plt.axhline(0, color="gray", ls="--", lw=1)
        plt.xlabel(f"{channel} Intensity")
        plt.ylabel("Residual (True − Pred)")
        plt.title(f"Residual vs {channel} (deg={degree})")
        plt.xlim(0, 1024)
        plt.xticks(np.arange(0, 1025, 200))
        plt.grid(ls="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        return str(save_path)

    # ------------------------------------------------------------------
    # 4. Univariate fit plot (per‑channel)
    # ------------------------------------------------------------------
    def plot_univariate_fit(
        self,
        channel: str = "auto",
        degree: str | int = "auto",
        figsize=(7, 4),
        save_path: str | Path | None = None,
    ) -> tuple[int, float]:
        """Fit y=Br vs single channel and plot. If degree="auto":
        • try degree‑1 → if R²≥0.99 accept; else compare degree‑1/2 and choose best."""
        channels = ["W", "R", "G", "B"]
        if channel == "auto" or channel not in channels or not (self.df[channel] > 0).any():
            channel = next((ch for ch in channels if (self.df[ch] > 0).any()), None)
            if channel is None:
                raise ValueError("找不到任何通道有大於0的資料")

        df_ch = self.df[self.df[channel] > 0]
        x = df_ch[[channel]].values
        y = df_ch["Br"].values

        # decide best_deg & r2_ch
        if degree == "auto":
            # degree‑1 first
            poly1 = PolynomialFeatures(1, include_bias=False)
            r2_1 = LinearRegression().fit(poly1.fit_transform(x), y).score(poly1.transform(x), y)
            if r2_1 >= 0.99:
                best_deg, r2_ch = 1, r2_1
            else:
                poly2 = PolynomialFeatures(2, include_bias=False)
                r2_2 = LinearRegression().fit(poly2.fit_transform(x), y).score(poly2.transform(x), y)
                if r2_2 > r2_1:
                    best_deg, r2_ch = 2, r2_2
                else:
                    best_deg, r2_ch = 1, r2_1
        else:
            best_deg = int(degree)
            poly_tmp = PolynomialFeatures(best_deg, include_bias=False)
            r2_ch = LinearRegression().fit(poly_tmp.fit_transform(x), y).score(poly_tmp.transform(x), y)

        # final fit with chosen degree
        poly = PolynomialFeatures(best_deg, include_bias=False).fit(x)
        lin = LinearRegression().fit(poly.transform(x), y)
        coef = np.concatenate(([lin.intercept_], lin.coef_))

        xs = np.linspace(x.min(), x.max(), 200).reshape(-1, 1)
        ys = lin.predict(poly.transform(xs))

        if save_path is None:
            tmp = tempfile.mkdtemp()
            save_path = Path(tmp) / f"fit_{channel}_deg{best_deg}.png"
        else:
            save_path = Path(save_path)

        plt.figure(figsize=figsize)
        plt.scatter(x, y, s=36, c="#71A6FF", edgecolors="none", alpha=0.8)
        plt.plot(xs, ys, c="#1F77B4", ls="--", lw=2)
        title_map = {"W": "W Light", "R": "R Light", "G": "G Light", "B": "B Light"}
        plt.title(title_map[channel])
        plt.xlabel(f"{channel} Intensity")
        plt.ylabel("Brightness (Br)")
        plt.ylim(0, 250)
        plt.yticks(np.arange(0, 251, 50))
        plt.grid(ls="--", alpha=0.6)

        # annotation
        terms = [f"{coef[i]:.4g}x" if i == 1 else f"{coef[i]:.4g}x^{i}" for i in range(1, best_deg + 1)]
        eqn = " + ".join(terms) + f" + {coef[0]:.4g}"
        plt.text(0.4, 0.25, f"y = {eqn}\nR² = {r2_ch:.4f}", transform=plt.gca().transAxes,
                 fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        return best_deg, r2_ch
