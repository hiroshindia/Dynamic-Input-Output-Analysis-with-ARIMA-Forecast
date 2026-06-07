"""
産業連関分析 結果グラフ化モジュール
=====================================
io_analysis.py の各分析結果を matplotlib でグラフ化する。
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定（環境に応じて変更）
plt.rcParams['font.family'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
LABEL_FONT = 9


# ---------------------------------------------------------------------------
# 1. 輸入石油価格の波及効果（DPD1 vs DPD2）
# ---------------------------------------------------------------------------

def plot_spread_effect(DPD1: np.ndarray, DPD2: np.ndarray,
                       sector_names: list = None,
                       save_path: str = "fig_spread_effect.png"):
    """
    輸入石油価格の波及効果：直接効果 vs 波及効果の比較棒グラフ

    Parameters
    ----------
    DPD1        : 直接効果による価格変化 (28,1)
    DPD2        : 波及効果を含む価格変化 (28,1)
    sector_names: 産業名リスト（省略時は番号）
    save_path   : 保存先ファイルパス
    """
    n = len(DPD1)
    labels = sector_names if sector_names else [f"Sec.{i+1}" for i in range(n)]
    x = np.arange(n)
    w = 0.4

    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    fig.suptitle("I. Spread Effect of Import Oil Price on Domestic Prices",
                 fontsize=13, fontweight='bold', y=0.98)

    # 上段: 直接効果 vs 波及効果（棒グラフ）
    ax = axes[0]
    ax.bar(x - w/2, DPD1.flatten(), w, label="Direct Effect (DPD1)",
           color="#2196F3", alpha=0.85)
    ax.bar(x + w/2, DPD2.flatten(), w, label="Spread Effect (DPD2)",
           color="#FF5722", alpha=0.85)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_title("Price Change by Sector: Direct vs Spread Effect", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=LABEL_FONT)
    ax.set_ylabel("Price Change")
    ax.legend(fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # 下段: 差分（波及効果 - 直接効果）
    ax2 = axes[1]
    diff = DPD2.flatten() - DPD1.flatten()
    colors = ['#4CAF50' if v >= 0 else '#F44336' for v in diff]
    ax2.bar(x, diff, color=colors, alpha=0.85)
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.set_title("Additional Effect via Leontief Inverse (DPD2 - DPD1)", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=LABEL_FONT)
    ax2.set_ylabel("Additional Effect")
    ax2.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存: {save_path}")


# ---------------------------------------------------------------------------
# 2. 石油消費変動の要因分解（積み上げ棒グラフ）
# ---------------------------------------------------------------------------

def plot_petroleum_decomposition(PETRO: np.ndarray,
                                  STRUC: np.ndarray,
                                  FINAL: np.ndarray,
                                  CROSS: np.ndarray,
                                  DXp: np.ndarray = None,
                                  sector_names: list = None,
                                  save_path: str = "fig_petroleum_decomp.png"):
    """
    石油消費変動の要因分解：積み上げ棒グラフ

    Parameters
    ----------
    PETRO  : 投入係数変化の寄与 (31,1)
    STRUC  : 構造変化（逆行列変化）の寄与 (31,1)
    FINAL  : 最終需要変化の寄与 (31,1)
    CROSS  : 交差項 (31,1)
    DXp    : 実際の石油中間需要変化（検証用、省略可）
    """
    n = PETRO.shape[0]
    labels = sector_names if sector_names else [f"Ind.{i+1}" for i in range(n)]
    x = np.arange(n)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("II-1. Fluctuation of Petroleum Consumption: Factor Decomposition",
                 fontsize=13, fontweight='bold')

    # 左: 積み上げ棒グラフ
    ax = axes[0]
    p = PETRO.flatten()
    s = STRUC.flatten()
    f = FINAL.flatten()
    c = CROSS.flatten()

    # 正の寄与と負の寄与を分けて積み上げる
    def pos(arr): return np.where(arr > 0, arr, 0)
    def neg(arr): return np.where(arr < 0, arr, 0)

    bottom_pos = np.zeros(n)
    bottom_neg = np.zeros(n)
    colors = ["#1565C0", "#2E7D32", "#F57F17", "#880E4F"]
    items  = [("Input Coeff. Change\n(PETRO)", p),
              ("Structure Change\n(STRUC)",    s),
              ("Final Demand Change\n(FINAL)",  f),
              ("Cross Term\n(CROSS)",           c)]

    for (lbl, vals), col in zip(items, colors):
        ax.bar(x, pos(vals), bottom=bottom_pos, label=lbl, color=col, alpha=0.85)
        ax.bar(x, neg(vals), bottom=bottom_neg, color=col, alpha=0.85)
        bottom_pos += pos(vals)
        bottom_neg += neg(vals)

    if DXp is not None:
        dxp_flat = DXp.flatten()[:n]  # industry数に合わせてトリム
        ax.plot(x[:len(dxp_flat)], dxp_flat, 'ko-', markersize=4,
                linewidth=1.2, label="Actual DXp", zorder=5)

    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=LABEL_FONT)
    ax.set_ylabel("Change in Petroleum Demand")
    ax.set_title("Stacked Factor Decomposition by Industry", fontsize=11)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    # 右: 各要因の合計パイチャート
    ax2 = axes[1]
    totals = np.array([abs(p.sum()), abs(s.sum()),
                       abs(f.sum()), abs(c.sum())])
    pie_labels = ["Input Coeff.\n(PETRO)", "Structure\n(STRUC)",
                  "Final Demand\n(FINAL)", "Cross\n(CROSS)"]
    pie_colors = colors
    wedges, texts, autotexts = ax2.pie(
        totals, labels=pie_labels, colors=pie_colors,
        autopct='%1.1f%%', startangle=90,
        textprops={'fontsize': 9})
    ax2.set_title("Relative Share of Each Factor\n(absolute values)", fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存: {save_path}")


# ---------------------------------------------------------------------------
# 3. 石油投入係数の変動：代替効果・省エネ効果（散布図＋棒グラフ）
# ---------------------------------------------------------------------------

def plot_input_coefficient(Pij0: np.ndarray, Pij1: np.ndarray,
                            SUBS: np.ndarray, REFE: np.ndarray,
                            CROS: np.ndarray,
                            Rp0: np.ndarray = None, Rp1: np.ndarray = None,
                            Eij0: np.ndarray = None, Eij1: np.ndarray = None,
                            sector_names: list = None,
                            save_path: str = "fig_input_coeff.png"):
    """
    石油投入係数の変動分析グラフ（4パネル）

    Parameters
    ----------
    Pij0, Pij1 : 基準年・比較年の石油投入係数 (1,69)
    SUBS       : 代替効果 (1,69)
    REFE       : 省エネ効果 (1,69)
    CROS       : 交差項 (1,69)
    Rp0, Rp1   : 石油比率（省略可）
    Eij0, Eij1 : エネルギー強度（省略可）
    """
    n = Pij0.shape[1]
    labels = sector_names if sector_names else [f"Ind.{i+1}" for i in range(n)]
    x = np.arange(n)

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("II-2. Fluctuation of Petroleum Input Coefficient",
                 fontsize=13, fontweight='bold')
    gs = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # --- パネル1: 投入係数の変化（基準年 vs 比較年）---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(Pij0.flatten(), Pij1.flatten(),
                s=30, alpha=0.7, color='#1976D2', edgecolors='white', linewidths=0.5)
    lims = [min(Pij0.min(), Pij1.min()) * 0.95,
            max(Pij0.max(), Pij1.max()) * 1.05]
    ax1.plot(lims, lims, 'r--', linewidth=1, label='No change line')
    ax1.set_xlabel("Pij (Base Year)", fontsize=10)
    ax1.set_ylabel("Pij (Period-t)", fontsize=10)
    ax1.set_title("Petroleum Input Coeff: Base vs Period-t", fontsize=10)
    ax1.legend(fontsize=8)
    ax1.grid(linestyle='--', alpha=0.4)

    # --- パネル2: 要因分解（積み上げ棒グラフ）---
    ax2 = fig.add_subplot(gs[0, 1])
    dpij = (Pij1 - Pij0).flatten()
    subs = SUBS.flatten()
    refe = REFE.flatten()
    cros = CROS.flatten()

    def pos(a): return np.where(a > 0, a, 0)
    def neg(a): return np.where(a < 0, a, 0)

    bp, bn = np.zeros(n), np.zeros(n)
    for vals, col, lbl in [
        (subs, '#1565C0', 'Substitution (SUBS)'),
        (refe, '#2E7D32', 'Energy Saving (REFE)'),
        (cros, '#F57F17', 'Cross Term (CROS)'),
    ]:
        ax2.bar(x, pos(vals), bottom=bp, color=col, alpha=0.8, label=lbl)
        ax2.bar(x, neg(vals), bottom=bn, color=col, alpha=0.8)
        bp += pos(vals)
        bn += neg(vals)
    ax2.plot(x, dpij, 'ko-', markersize=3, linewidth=1, label='Actual DPij', zorder=5)
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.set_title("Factor Decomposition of DPij", fontsize=10)
    ax2.set_ylabel("Change in Input Coefficient")
    ax2.legend(fontsize=7, loc='upper right')
    ax2.grid(axis='y', linestyle='--', alpha=0.4)
    ax2.set_xticks([])

    # --- パネル3: 石油比率の変化（Rp）---
    ax3 = fig.add_subplot(gs[1, 0])
    if Rp0 is not None and Rp1 is not None:
        drp = (Rp1 - Rp0).flatten()
        colors3 = ['#1565C0' if v >= 0 else '#C62828' for v in drp]
        ax3.bar(x, drp, color=colors3, alpha=0.8)
        ax3.axhline(0, color='black', linewidth=0.8)
        ax3.set_title("Change in Oil/Energy Ratio (DRp)", fontsize=10)
        ax3.set_ylabel("DRp")
        ax3.grid(axis='y', linestyle='--', alpha=0.4)
        ax3.set_xticks([])
    else:
        ax3.text(0.5, 0.5, "Rp0/Rp1 not provided",
                 ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title("Change in Oil/Energy Ratio (DRp)", fontsize=10)

    # --- パネル4: エネルギー強度の変化（Eij）---
    ax4 = fig.add_subplot(gs[1, 1])
    if Eij0 is not None and Eij1 is not None:
        deij = (Eij1 - Eij0).flatten()
        colors4 = ['#2E7D32' if v >= 0 else '#C62828' for v in deij]
        ax4.bar(x, deij, color=colors4, alpha=0.8)
        ax4.axhline(0, color='black', linewidth=0.8)
        ax4.set_title("Change in Energy Intensity (DEij)", fontsize=10)
        ax4.set_ylabel("DEij")
        ax4.grid(axis='y', linestyle='--', alpha=0.4)
        ax4.set_xticks([])
    else:
        ax4.text(0.5, 0.5, "Eij0/Eij1 not provided",
                 ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title("Change in Energy Intensity (DEij)", fontsize=10)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存: {save_path}")


# ---------------------------------------------------------------------------
# 4. Leontief 逆行列のヒートマップ
# ---------------------------------------------------------------------------

def plot_leontief_heatmap(matrix: np.ndarray,
                           title: str = "Leontief Inverse Matrix",
                           save_path: str = "fig_leontief.png"):
    """
    レオンチェフ逆行列（または投入係数行列）をヒートマップで表示。
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel("Industry (column)", fontsize=10)
    ax.set_ylabel("Industry (row)", fontsize=10)
    n = matrix.shape[0]
    if n <= 31:
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels([str(i+1) for i in range(n)], fontsize=7)
        ax.set_yticklabels([str(i+1) for i in range(n)], fontsize=7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存: {save_path}")


# ---------------------------------------------------------------------------
# まとめて全図を生成
# ---------------------------------------------------------------------------

def generate_all_figures(results_i, results_base, results_ii1, results_ii2,
                          output_dir: str = "."):
    """
    各分析結果から全グラフを一括生成する。

    Parameters
    ----------
    results_i    : spread_effect_import_oil_price() の返り値
    results_base : petroleum_base_year() の返り値
    results_ii1  : petroleum_period_t() の返り値
    results_ii2  : petroleum_input_coefficient() の返り値
    output_dir   : グラフの保存先ディレクトリ
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    print("\n[グラフ生成中...]")

    plot_spread_effect(
        results_i["DPD1"], results_i["DPD2"],
        save_path=os.path.join(output_dir, "fig1_spread_effect.png")
    )

    plot_leontief_heatmap(
        results_base["LINVO"],
        title="Leontief Inverse Matrix (Base Year, LINVO)",
        save_path=os.path.join(output_dir, "fig2_leontief_base.png")
    )

    plot_petroleum_decomposition(
        results_ii1["PETRO"], results_ii1["STRUC"],
        results_ii1["FINAL"], results_ii1["CROSS"],
        DXp=results_ii1["DXp"],
        save_path=os.path.join(output_dir, "fig3_petroleum_decomp.png")
    )

    plot_input_coefficient(
        results_ii2["Pij0"], results_ii2["Pij1"],
        results_ii2["SUBS"], results_ii2["REFE"], results_ii2["CROS"],
        Rp0=results_ii2["Rp0"], Rp1=results_ii2["Rp1"],
        Eij0=results_ii2["Eij0"], Eij1=results_ii2["Eij1"],
        save_path=os.path.join(output_dir, "fig4_input_coeff.png")
    )

    print(f"\n全4図を {output_dir} に保存しました。")


# ---------------------------------------------------------------------------
# 単体テスト実行
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from io_analysis import (spread_effect_import_oil_price,
                              petroleum_base_year,
                              petroleum_period_t,
                              petroleum_input_coefficient)

    print("ダミーデータでグラフ生成テスト...\n")
    ri   = spread_effect_import_oil_price()
    base = petroleum_base_year()
    rii1 = petroleum_period_t(base)
    rii2 = petroleum_input_coefficient()

    generate_all_figures(ri, base, rii1, rii2, output_dir=".")
