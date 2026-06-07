"""
動学的産業連関分析
ARIMA による投入係数予測 → レオンチェフ逆行列再構成 → 波及効果シミュレーション
===================================================================================
早稲田大学政治経済学部 中村研究室 卒業論文（1986-1988年）の再現
原典: RATS プログラム → Python 変換・発展

分析フロー:
  ① 各産業の投入係数 aᵢⱼ(t) の時系列データ（1960-1980年）
  ② ARIMA(p,d,q) で aᵢⱼ の将来値を予測
  ③ 予測係数で投入係数行列 A を再構成
  ④ レオンチェフ逆行列 L = (I - A)⁻¹ を再計算
  ⑤ 石油波及効果・最終需要の将来シミュレーション
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import linalg
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.dpi'] = 130
plt.rcParams['font.size']  = 9


# =============================================================================
# ARIMA スクラッチ実装（statsmodels なし）
# =============================================================================

class ARIMA:
    """
    ARIMA(p, d, q) モデル
    - AR部分: 自己回帰
    - I 部分: 階差（定常化）
    - MA部分: 移動平均誤差（簡易実装）

    1980年代の卒論で使われていたのと同等のモデル。
    当時は各係数ごとにこれを手動で推定していた。
    """

    def __init__(self, p=1, d=1, q=0):
        self.p = p   # AR次数
        self.d = d   # 階差次数
        self.q = q   # MA次数（簡易: q=0 のみ完全サポート）
        self.ar_params = None
        self.intercept = None
        self.sigma2    = None
        self._orig     = None  # 元系列（逆差分用）

    def _difference(self, y, d):
        """d 階差分"""
        for _ in range(d):
            y = np.diff(y)
        return y

    def _fit_ar_ols(self, y):
        """AR(p) を OLS で推定"""
        p = self.p
        if len(y) <= p:
            # データ不足: ナイーブにゼロ係数
            return np.zeros(p), np.mean(y), 1e-6
        n = len(y)
        # 計画行列
        X = np.column_stack([y[p-i-1:n-i-1] for i in range(p)])
        X = np.column_stack([np.ones(n - p), X])
        y_t = y[p:]
        # OLS: β = (X'X)⁻¹ X'y
        try:
            beta = np.linalg.lstsq(X, y_t, rcond=None)[0]
        except Exception:
            beta = np.zeros(p + 1)
        intercept = beta[0]
        ar_params = beta[1:]
        residuals = y_t - X @ beta
        sigma2 = np.var(residuals) if len(residuals) > 0 else 1e-6
        return ar_params, intercept, sigma2

    def fit(self, y):
        """モデルを時系列 y に当てはめる"""
        self._orig = np.array(y, dtype=float)
        y_diff = self._difference(self._orig.copy(), self.d)
        self.ar_params, self.intercept, self.sigma2 = self._fit_ar_ols(y_diff)
        return self

    def forecast(self, steps=5):
        """steps 期先まで予測（点予測）"""
        # 差分系列を延長
        y_diff = self._difference(self._orig.copy(), self.d)
        history = list(y_diff)
        forecasts_diff = []
        for _ in range(steps):
            if self.p > 0 and len(history) >= self.p:
                lag_vals = np.array(history[-self.p:][::-1])
                next_val = self.intercept + np.dot(self.ar_params, lag_vals)
            else:
                next_val = self.intercept
            forecasts_diff.append(next_val)
            history.append(next_val)

        # 逆差分（累積和で元のスケールに戻す）
        result = np.array(forecasts_diff)
        for _ in range(self.d):
            last = self._orig[-1] if _ == 0 else result_prev[-1]
            result = np.cumsum(result) + last
            result_prev = result.copy()

        return result

    def forecast_with_ci(self, steps=5, alpha=0.95):
        """予測値 + 信頼区間"""
        point = self.forecast(steps)
        z = 1.96  # 95% CI
        se = np.sqrt(self.sigma2) * np.sqrt(np.arange(1, steps + 1))
        lower = point - z * se
        upper = point + z * se
        return point, lower, upper


# =============================================================================
# 仮想的な歴史データの生成
# （実データ: 1960,1965,1970,1975,1980年の産業連関表 5時点）
# =============================================================================

def generate_historical_io_data(n_sectors=10, n_years=5, seed=42):
    """
    5時点（1960,65,70,75,80年）の投入係数行列を生成。
    実データに差し替える場合は、shape = (n_years, n_sectors, n_sectors) の
    numpy配列を用意してください。

    産業分類（簡略版10部門）:
        0: 農林水産業
        1: 鉱業（石炭・金属等）
        2: 石油・石炭製品  ← 石油関連
        3: 化学
        4: 鉄鋼
        5: 機械
        6: 電気機械
        7: 輸送機械
        8: サービス
        9: その他製造業
    """
    np.random.seed(seed)
    years = [1960, 1965, 1970, 1975, 1980]
    n = n_sectors

    # 基本的な投入係数行列（1960年ベース）
    A_base = np.random.rand(n, n) * 0.15
    np.fill_diagonal(A_base, 0)  # 自己投入なし
    # 固有値条件（収束保証）
    while np.max(np.abs(np.linalg.eigvals(A_base))) >= 1:
        A_base *= 0.8

    # 時系列トレンド: オイルショックの影響を反映
    # 石油関連産業（行2）の投入係数は1973年以降下降トレンド
    A_all = np.zeros((n_years, n, n))
    for t, year in enumerate(years):
        noise = np.random.randn(n, n) * 0.005
        trend = np.zeros((n, n))

        # 石油投入係数（行2）: 1970年ピーク → 以降省エネで低下
        oil_factor = 1.0
        if year >= 1975:
            oil_factor = 0.85  # 第1次オイルショック後
        if year >= 1980:
            oil_factor = 0.70  # 第2次オイルショック後

        # 石油行の係数にトレンドを付加
        trend[2, :] = (oil_factor - 1.0) * A_base[2, :]

        # 機械・電機は技術進歩で効率化
        if year >= 1970:
            trend[5, :] -= 0.003 * (year - 1970) / 5
            trend[6, :] -= 0.002 * (year - 1970) / 5

        A_t = A_base + trend + noise
        A_t = np.clip(A_t, 0, None)
        # 各列の和が1未満になるよう正規化
        col_sums = A_t.sum(axis=0)
        for j in range(n):
            if col_sums[j] >= 0.95:
                A_t[:, j] *= 0.90 / col_sums[j]
        np.fill_diagonal(A_t, 0)
        A_all[t] = A_t

    sector_names = ['農林水産', '鉱業', '石油製品', '化学',
                    '鉄鋼', '機械', '電気機械', '輸送機械',
                    'サービス', 'その他']
    return A_all, years, sector_names


# =============================================================================
# ARIMA で全投入係数を予測
# =============================================================================

def forecast_all_coefficients(A_all, years, forecast_years,
                               arima_order=(1, 1, 0)):
    """
    各 (i,j) の投入係数時系列に ARIMA を当てはめ、将来値を予測する。

    Parameters
    ----------
    A_all        : shape (T, n, n) の歴史的投入係数行列
    years        : 歴史データの年リスト
    forecast_years: 予測年リスト
    arima_order  : (p, d, q)

    Returns
    -------
    A_forecast : shape (len(forecast_years), n, n) の予測投入係数行列
    models     : 各 (i,j) の ARIMA モデルオブジェクト
    ci_lower, ci_upper : 95% 信頼区間
    """
    T, n, _ = A_all.shape
    steps = len(forecast_years)
    p, d, q = arima_order

    A_forecast = np.zeros((steps, n, n))
    ci_lower   = np.zeros((steps, n, n))
    ci_upper   = np.zeros((steps, n, n))
    models     = {}

    print(f'ARIMA({p},{d},{q}) で {n}×{n} = {n*n} 個の投入係数を予測中...')

    for i in range(n):
        for j in range(n):
            series = A_all[:, i, j]
            model  = ARIMA(p=p, d=d, q=q).fit(series)
            fc, lo, hi = model.forecast_with_ci(steps=steps)

            # 投入係数は非負
            A_forecast[:, i, j] = np.clip(fc, 0, None)
            ci_lower[:, i, j]   = np.clip(lo, 0, None)
            ci_upper[:, i, j]   = np.clip(hi, 0, None)
            models[(i, j)]      = model

    # 各列の和が 1 未満になるよう調整（レオンチェフ条件）
    for t in range(steps):
        col_sums = A_forecast[t].sum(axis=0)
        for j in range(n):
            if col_sums[j] >= 0.95:
                A_forecast[t, :, j] *= 0.90 / col_sums[j]
        np.fill_diagonal(A_forecast[t], 0)

    print(f'予測完了: {forecast_years}')
    return A_forecast, models, ci_lower, ci_upper


# =============================================================================
# レオンチェフ逆行列の計算と波及効果シミュレーション
# =============================================================================

def compute_leontief_series(A_series):
    """各時点の A から L = (I-A)⁻¹ を計算"""
    T, n, _ = A_series.shape
    L_series = np.zeros_like(A_series)
    I = np.eye(n)
    for t in range(T):
        try:
            L_series[t] = np.linalg.inv(I - A_series[t])
        except np.linalg.LinAlgError:
            L_series[t] = I  # フォールバック
    return L_series


def simulate_spread_effect(L_series, years, oil_price_shock=0.10,
                           oil_sector=2):
    """
    石油価格ショック（10%上昇）による波及効果を各年でシミュレーション

    Parameters
    ----------
    L_series        : shape (T, n, n) のレオンチェフ逆行列時系列
    oil_price_shock : 石油価格上昇率
    oil_sector      : 石油産業のインデックス

    Returns
    -------
    effects : shape (T, n) — 各年・各産業への価格波及効果
    """
    T, n, _ = L_series.shape
    effects = np.zeros((T, n))
    shock_vec = np.zeros(n)
    shock_vec[oil_sector] = oil_price_shock

    for t in range(T):
        effects[t] = L_series[t].T @ shock_vec

    return effects


# =============================================================================
# グラフ描画
# =============================================================================

def plot_arima_forecast(A_all, A_forecast, years, forecast_years,
                         ci_lower, ci_upper, sector_names,
                         save_path='fig_arima_forecast.png'):
    """
    重要な投入係数の ARIMA 予測結果をグラフ化（石油関連中心）
    """
    # 注目する (i,j) ペア: 石油産業（行2）の各産業への投入
    n = len(sector_names)
    watch_pairs = [(2, j) for j in range(min(n, 8))]  # 石油行

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(
        'ARIMA Forecast of Input Coefficients\n'
        '（石油製品産業 → 各産業への投入係数の予測）',
        fontsize=12, fontweight='bold'
    )
    all_years  = list(years) + list(forecast_years)
    hist_x     = list(years)
    fore_x     = list(forecast_years)

    for idx, (i, j) in enumerate(watch_pairs):
        ax = axes[idx // 4][idx % 4]
        hist_vals = A_all[:, i, j]
        fore_vals = A_forecast[:, i, j]
        lo_vals   = ci_lower[:, i, j]
        hi_vals   = ci_upper[:, i, j]

        # 歴史データ
        ax.plot(hist_x, hist_vals, 'o-', color='#1565C0',
                lw=2, ms=5, label='Actual (1960-1980)', zorder=3)
        # 予測
        ax.plot(fore_x, fore_vals, 's--', color='#E53935',
                lw=2, ms=5, label='ARIMA Forecast', zorder=3)
        # 信頼区間
        ax.fill_between(fore_x, lo_vals, hi_vals,
                        color='#EF9A9A', alpha=0.4, label='95% CI')
        # 区切り線
        ax.axvline(x=1980, color='gray', ls=':', lw=1)
        ax.set_title(f'a[石油→{sector_names[j]}]', fontsize=9)
        ax.set_xlabel('Year', fontsize=8)
        ax.set_ylabel('Coeff.', fontsize=8)
        ax.legend(fontsize=6, loc='best')
        ax.grid(ls='--', alpha=0.4)
        ax.tick_params(labelsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  保存: {save_path}')


def plot_leontief_evolution(L_hist, L_fore, years, forecast_years,
                             sector_names,
                             save_path='fig_leontief_evolution.png'):
    """
    レオンチェフ逆行列の時間変化（石油行の乗数効果）
    """
    all_years = list(years) + list(forecast_years)
    all_L     = np.concatenate([L_hist, L_fore], axis=0)
    n         = len(sector_names)

    # 石油産業（行2）の列和 = 石油への総依存度
    oil_multiplier = all_L[:, 2, :].sum(axis=1)  # 石油行の行和

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        'Leontief Inverse Matrix: Time Evolution\n'
        '（レオンチェフ逆行列の時系列変化）',
        fontsize=12, fontweight='bold'
    )

    # 左: 石油乗数効果の時系列
    ax = axes[0]
    hist_n = len(years)
    ax.plot(years, oil_multiplier[:hist_n], 'o-',
            color='#1565C0', lw=2.5, ms=6, label='Actual')
    ax.plot(forecast_years, oil_multiplier[hist_n:], 's--',
            color='#E53935', lw=2.5, ms=6, label='ARIMA Forecast')
    ax.axvline(x=1980, color='gray', ls=':', lw=1.2, label='Forecast start')
    ax.axvspan(min(forecast_years), max(forecast_years),
               alpha=0.08, color='red')
    ax.set_title('Oil Sector Multiplier (Row Sum of L[oil,:])')
    ax.set_xlabel('Year'); ax.set_ylabel('Multiplier')
    ax.legend(); ax.grid(ls='--', alpha=0.4)

    # 右: 2時点（1980年 vs 最終予測年）のヒートマップ比較
    ax2 = axes[1]
    diff_L = all_L[-1] - all_L[hist_n - 1]  # 予測最終年 - 1980年
    im = ax2.imshow(diff_L, cmap='RdBu_r', aspect='auto',
                    vmin=-np.abs(diff_L).max(), vmax=np.abs(diff_L).max())
    plt.colorbar(im, ax=ax2, shrink=0.8)
    ax2.set_title(f'Change in L: {forecast_years[-1]} - 1980')
    ax2.set_xlabel('Industry (column)'); ax2.set_ylabel('Industry (row)')
    ax2.set_xticks(range(n))
    ax2.set_xticklabels([s[:3] for s in sector_names], rotation=45, fontsize=7)
    ax2.set_yticks(range(n))
    ax2.set_yticklabels([s[:3] for s in sector_names], fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  保存: {save_path}')


def plot_spread_simulation(effects_hist, effects_fore,
                            years, forecast_years, sector_names,
                            save_path='fig_spread_simulation.png'):
    """
    石油価格ショックの波及効果シミュレーション（歴史 + 予測）
    """
    all_years   = list(years) + list(forecast_years)
    all_effects = np.concatenate([effects_hist, effects_fore], axis=0)
    hist_n      = len(years)
    n           = len(sector_names)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle(
        'Simulation: Spread Effect of 10% Oil Price Shock\n'
        '（石油価格10%上昇の国内価格への波及効果シミュレーション）',
        fontsize=12, fontweight='bold'
    )

    # 上: 時系列折れ線（産業別）
    ax = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, n))
    for j in range(n):
        ax.plot(years, all_effects[:hist_n, j], 'o-',
                color=colors[j], lw=1.5, ms=4, label=sector_names[j])
        ax.plot(forecast_years, all_effects[hist_n:, j], 's--',
                color=colors[j], lw=1.5, ms=4)
    ax.axvline(x=1980, color='gray', ls=':', lw=1.2)
    ax.axvspan(min(forecast_years), max(forecast_years),
               alpha=0.06, color='red', label='Forecast period')
    ax.set_title('Price Effect by Industry (Actual + Forecast)')
    ax.set_xlabel('Year'); ax.set_ylabel('Price Change (%)')
    ax.legend(fontsize=7, loc='upper right', ncol=2)
    ax.grid(ls='--', alpha=0.4)

    # 下: 棒グラフ比較（1980年 vs 最終予測年）
    ax2 = axes[1]
    x   = np.arange(n)
    w   = 0.35
    v80 = all_effects[hist_n - 1]
    vfc = all_effects[-1]
    ax2.bar(x - w/2, v80 * 100, w, label='1980 (Actual)',
            color='#1565C0', alpha=0.85)
    ax2.bar(x + w/2, vfc * 100, w, label=f'{forecast_years[-1]} (Forecast)',
            color='#E53935', alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(sector_names, rotation=30, ha='right', fontsize=8)
    ax2.set_ylabel('Price Effect (%)')
    ax2.set_title(f'Comparison: 1980 vs {forecast_years[-1]} Forecast')
    ax2.legend(); ax2.grid(axis='y', ls='--', alpha=0.4)
    ax2.axhline(0, color='black', lw=0.8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  保存: {save_path}')


def plot_summary_dashboard(A_all, A_forecast, L_hist, L_fore,
                            effects_hist, effects_fore,
                            years, forecast_years, sector_names,
                            save_path='fig_dashboard.png'):
    """
    全分析のサマリーダッシュボード（1枚にまとめ）
    """
    all_years   = list(years) + list(forecast_years)
    all_effects = np.concatenate([effects_hist, effects_fore], axis=0)
    all_L       = np.concatenate([L_hist, L_fore], axis=0)
    hist_n      = len(years)
    n           = len(sector_names)

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        '動学的産業連関分析ダッシュボード\n'
        'Dynamic Input-Output Analysis with ARIMA Forecast\n'
        '早稲田大学 中村研究室 卒業論文（1986-1988）復元',
        fontsize=13, fontweight='bold', y=1.01
    )
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.38)

    colors10 = plt.cm.tab10(np.linspace(0, 1, n))

    # A) 石油投入係数の予測（代表4産業）
    ax_a = fig.add_subplot(gs[0, :2])
    rep_sectors = [0, 3, 5, 8]
    for j in rep_sectors:
        c = colors10[j]
        ax_a.plot(years, A_all[:, 2, j], 'o-', color=c, lw=1.8, ms=4,
                  label=f'→{sector_names[j]}')
        ax_a.plot(forecast_years, A_forecast[:, 2, j], 's--', color=c, lw=1.8, ms=4)
    ax_a.axvline(1980, color='gray', ls=':', lw=1)
    ax_a.axvspan(min(forecast_years), max(forecast_years), alpha=0.07, color='red')
    ax_a.set_title('① 石油産業の投入係数 ARIMA 予測')
    ax_a.set_ylabel('投入係数 aᵢⱼ')
    ax_a.legend(fontsize=7, ncol=2); ax_a.grid(ls='--', alpha=0.4)

    # B) レオンチェフ乗数の推移
    ax_b = fig.add_subplot(gs[0, 2])
    oil_mult = all_L[:, 2, :].sum(axis=1)
    ax_b.plot(years, oil_mult[:hist_n], 'o-', color='#1565C0', lw=2, ms=5)
    ax_b.plot(forecast_years, oil_mult[hist_n:], 's--', color='#E53935', lw=2, ms=5)
    ax_b.axvline(1980, color='gray', ls=':', lw=1)
    ax_b.set_title('② 石油乗数効果の推移')
    ax_b.set_ylabel('乗数 (行和)')
    ax_b.grid(ls='--', alpha=0.4)

    # C) 波及効果シミュレーション（時系列）
    ax_c = fig.add_subplot(gs[1, :2])
    for j in range(n):
        ax_c.plot(years, all_effects[:hist_n, j] * 100, 'o-',
                  color=colors10[j], lw=1.3, ms=3, label=sector_names[j])
        ax_c.plot(forecast_years, all_effects[hist_n:, j] * 100, 's--',
                  color=colors10[j], lw=1.3, ms=3)
    ax_c.axvline(1980, color='gray', ls=':', lw=1)
    ax_c.axvspan(min(forecast_years), max(forecast_years), alpha=0.07, color='red')
    ax_c.set_title('③ 石油価格10%上昇の波及効果シミュレーション')
    ax_c.set_ylabel('価格変化 (%)')
    ax_c.legend(fontsize=6, ncol=2, loc='upper right')
    ax_c.grid(ls='--', alpha=0.4)

    # D) 1980年 vs 予測最終年 比較
    ax_d = fig.add_subplot(gs[1, 2])
    v80 = all_effects[hist_n - 1] * 100
    vfc = all_effects[-1] * 100
    xd  = np.arange(n)
    ax_d.barh(xd - 0.2, v80, 0.35, label='1980', color='#1565C0', alpha=0.85)
    ax_d.barh(xd + 0.2, vfc, 0.35,
              label=f'{forecast_years[-1]} forecast', color='#E53935', alpha=0.85)
    ax_d.set_yticks(xd)
    ax_d.set_yticklabels(sector_names, fontsize=7)
    ax_d.set_xlabel('価格変化 (%)')
    ax_d.set_title(f'④ 1980 vs {forecast_years[-1]}')
    ax_d.legend(fontsize=7); ax_d.grid(axis='x', ls='--', alpha=0.4)

    # E) Leontief 逆行列 ヒートマップ（1980年）
    ax_e = fig.add_subplot(gs[2, 0])
    im_e = ax_e.imshow(all_L[hist_n - 1], cmap='YlOrRd', aspect='auto')
    plt.colorbar(im_e, ax=ax_e, shrink=0.7)
    ax_e.set_title('⑤ Leontief L (1980実績)')
    ax_e.set_xticks(range(n))
    ax_e.set_xticklabels([s[:2] for s in sector_names], fontsize=6, rotation=45)
    ax_e.set_yticks(range(n))
    ax_e.set_yticklabels([s[:2] for s in sector_names], fontsize=6)

    # F) Leontief 逆行列 ヒートマップ（予測最終年）
    ax_f = fig.add_subplot(gs[2, 1])
    im_f = ax_f.imshow(all_L[-1], cmap='YlOrRd', aspect='auto')
    plt.colorbar(im_f, ax=ax_f, shrink=0.7)
    ax_f.set_title(f'⑥ Leontief L ({forecast_years[-1]}予測)')
    ax_f.set_xticks(range(n)); ax_f.set_xticklabels([s[:2] for s in sector_names], fontsize=6, rotation=45)
    ax_f.set_yticks(range(n)); ax_f.set_yticklabels([s[:2] for s in sector_names], fontsize=6)

    # G) 変化量ヒートマップ
    ax_g = fig.add_subplot(gs[2, 2])
    diff = all_L[-1] - all_L[hist_n - 1]
    vmax = np.abs(diff).max()
    im_g = ax_g.imshow(diff, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
    plt.colorbar(im_g, ax=ax_g, shrink=0.7)
    ax_g.set_title(f'⑦ ΔL ({forecast_years[-1]}-1980)')
    ax_g.set_xticks(range(n)); ax_g.set_xticklabels([s[:2] for s in sector_names], fontsize=6, rotation=45)
    ax_g.set_yticks(range(n)); ax_g.set_yticklabels([s[:2] for s in sector_names], fontsize=6)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  保存: {save_path}')


# =============================================================================
# メイン実行
# =============================================================================

if __name__ == '__main__':
    print('=' * 65)
    print('  動学的産業連関分析 with ARIMA')
    print('  早稲田大学 中村研究室 卒業論文（1986-1988年）復元')
    print('=' * 65)

    # --- ① 歴史データ生成（実データに差し替え可能）---
    print('\n[1] 歴史的投入係数行列 生成（1960-1980年、5時点）')
    A_all, years, sector_names = generate_historical_io_data(
        n_sectors=10, n_years=5
    )
    print(f'    形状: {A_all.shape}  ({len(years)} 時点 × 10産業 × 10産業)')
    print(f'    年次: {years}')
    print(f'    産業: {sector_names}')

    # --- ② ARIMA で将来予測 ---
    print('\n[2] ARIMA(1,1,0) で投入係数を予測')
    forecast_years = [1985, 1990, 1995, 2000, 2005]
    A_forecast, models, ci_lower, ci_upper = forecast_all_coefficients(
        A_all, years, forecast_years, arima_order=(1, 1, 0)
    )

    # --- ③ レオンチェフ逆行列の計算 ---
    print('\n[3] レオンチェフ逆行列 L = (I-A)⁻¹ を各年で計算')
    L_hist = compute_leontief_series(A_all)
    L_fore = compute_leontief_series(A_forecast)
    print(f'    歴史: {L_hist.shape}, 予測: {L_fore.shape}')

    # --- ④ 波及効果シミュレーション ---
    print('\n[4] 石油価格10%上昇の波及効果シミュレーション')
    effects_hist = simulate_spread_effect(L_hist, years,
                                           oil_price_shock=0.10, oil_sector=2)
    effects_fore = simulate_spread_effect(L_fore, forecast_years,
                                           oil_price_shock=0.10, oil_sector=2)

    # --- ⑤ 結果サマリー ---
    print('\n[5] 結果サマリー')
    all_effects = np.concatenate([effects_hist, effects_fore], axis=0)
    all_years   = years + forecast_years
    print(f'\n    石油価格10%上昇 → 各産業への価格波及効果 (%):')
    print(f'    {"産業":<10}', end='')
    for yr in [1970, 1980, 1990, 2000]:
        idx = all_years.index(yr)
        print(f'  {yr}年', end='')
    print()
    for j, sname in enumerate(sector_names):
        print(f'    {sname:<10}', end='')
        for yr in [1970, 1980, 1990, 2000]:
            idx = all_years.index(yr)
            print(f'  {all_effects[idx,j]*100:>5.2f}%', end='')
        print()

    # --- ⑥ グラフ生成 ---
    print('\n[6] グラフ生成中...')
    out = '/mnt/user-data/outputs/'

    plot_arima_forecast(A_all, A_forecast, years, forecast_years,
                         ci_lower, ci_upper, sector_names,
                         save_path=out + 'fig_arima_forecast.png')

    plot_leontief_evolution(L_hist, L_fore, years, forecast_years,
                             sector_names,
                             save_path=out + 'fig_leontief_evolution.png')

    plot_spread_simulation(effects_hist, effects_fore, years, forecast_years,
                            sector_names,
                            save_path=out + 'fig_spread_simulation.png')

    plot_summary_dashboard(A_all, A_forecast, L_hist, L_fore,
                            effects_hist, effects_fore,
                            years, forecast_years, sector_names,
                            save_path=out + 'fig_dashboard.png')

    print('\n全分析完了！')
    print('=' * 65)
