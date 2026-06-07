"""
データ読み込みユーティリティ
=============================
RATSプログラムで使われていた各種データファイルを
NumPy配列として読み込むためのヘルパー関数群。

対応フォーマット:
  - スペース区切りテキスト（最も一般的）
  - CSV（カンマ区切り）
  - 固定長フォーマット
  - Fortran形式（科学表記 1.23D+04 など）
"""

import numpy as np
import os
import re


# ---------------------------------------------------------------------------
# 内部ユーティリティ
# ---------------------------------------------------------------------------

def _fortran_to_float(s: str) -> float:
    """Fortran の指数表記 (D/E) を Python の float に変換する"""
    return float(s.replace('D', 'E').replace('d', 'e'))


def _parse_line(line: str) -> list:
    """1行をトークンに分割し、Fortran 表記も含めて float のリストを返す"""
    tokens = line.split()
    return [_fortran_to_float(t) for t in tokens if t]


# ---------------------------------------------------------------------------
# 汎用読み込み関数
# ---------------------------------------------------------------------------

def load_matrix(filepath: str, shape: tuple,
                delimiter: str = None,
                fortran: bool = True,
                skip_rows: int = 0) -> np.ndarray:
    """
    テキストファイルを読み込み、指定した shape の ndarray を返す。

    Parameters
    ----------
    filepath  : ファイルパス
    shape     : (行数, 列数) のタプル
    delimiter : None=空白区切り, ','=CSV など
    fortran   : True の場合 Fortran 指数表記 (1.23D+04) を処理する
    skip_rows : 先頭スキップ行数

    Returns
    -------
    np.ndarray, shape=shape
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"ファイルが見つかりません: {filepath}")

    values = []
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        for _ in range(skip_rows):
            next(f)
        for line in f:
            line = line.strip()
            if not line or line.startswith(('*', '#', '!')):
                continue
            if fortran:
                values.extend(_parse_line(line))
            else:
                if delimiter:
                    values.extend([float(v) for v in line.split(delimiter) if v.strip()])
                else:
                    values.extend([float(v) for v in line.split() if v.strip()])

    total = shape[0] * shape[1]
    if len(values) < total:
        raise ValueError(
            f"データ不足: {filepath} から {len(values)} 個の値を読みましたが、"
            f"{total} 個 ({shape[0]}×{shape[1]}) が必要です。"
        )
    return np.array(values[:total], dtype=float).reshape(shape)


def load_vector(filepath: str, size: int,
                delimiter: str = None,
                fortran: bool = True,
                skip_rows: int = 0) -> np.ndarray:
    """
    テキストファイルを読み込み、shape=(size,1) の列ベクトルを返す。
    """
    mat = load_matrix(filepath, (size, 1), delimiter, fortran, skip_rows)
    return mat


def load_unit_vector(size: int) -> np.ndarray:
    """
    全要素 1 の列ベクトルを返す（RATS の READ UNITV に相当）。
    """
    return np.ones((size, 1))


# ---------------------------------------------------------------------------
# ファイル別ラッパー（プログラム固有）
# ---------------------------------------------------------------------------

class DataFiles:
    """
    各データファイルのパスを一元管理するクラス。
    インスタンス化してからデータを読み込む。

    使い方:
        df = DataFiles(base_dir="C:/data")
        A, Q, E, F, M, GDO = df.load_gulf()
        XIJO = df.load_mto60()
    """

    def __init__(self, base_dir: str = "."):
        self.base_dir = base_dir

    def _path(self, filename: str) -> str:
        return os.path.join(self.base_dir, filename)

    # ---- I. 輸入石油価格の波及効果 ----------------------------------------

    def load_gulf(self):
        """
        GULF.DAT を読み込む。
        RATS: OPEN DATA C:GULF.DAT / READ A Q E F M GDO

        Returns: A(28,28), Q(28,28), E(28,1), F(28,1), M(28,1), GDO(28,1)
        """
        path = self._path("GULF.DAT")
        # GULF.DAT は A, Q, E, F, M, GDO が連続して格納されている想定
        # ファイル構造が異なる場合は shape を調整してください
        A   = load_matrix(path, (28, 28))
        # 注意: 実際のファイルが別々の場合は個別ファイルに分けてください
        # 以下はダミー（実データ差し替え用プレースホルダー）
        Q   = np.linalg.inv(np.eye(28) - A * 0.5)
        E   = load_vector(self._path("E.DAT"),   28) if os.path.exists(self._path("E.DAT"))   else np.zeros((28,1))
        F   = load_vector(self._path("F.DAT"),   28) if os.path.exists(self._path("F.DAT"))   else np.zeros((28,1))
        M   = load_vector(self._path("M.DAT"),   28) if os.path.exists(self._path("M.DAT"))   else np.zeros((28,1))
        GDO = load_vector(self._path("GDO.DAT"), 28) if os.path.exists(self._path("GDO.DAT")) else np.zeros((28,1))
        return A, Q, E, F, M, GDO

    # ---- II-1. 石油消費の変動 ----------------------------------------------

    def load_mto60(self) -> np.ndarray:
        """
        MTO60.DAT（基準年 産業連関表）を読み込む。
        RATS: OPEN DATA C:MTO60.DAT / READ XIJO
        Returns: XIJO(46, 44)
        """
        return load_matrix(self._path("MTO60.DAT"), (46, 44))

    def load_mto65(self) -> np.ndarray:
        """
        MTO65.DAT（比較年 産業連関表）を読み込む。
        RATS: OPEN DATA C:MTO65.DAT / READ XIJ1
        Returns: XIJ1(46, 44)
        """
        return load_matrix(self._path("MTO65.DAT"), (46, 44))

    # ---- II-2. 石油投入係数の変動 ------------------------------------------

    def load_rio69_n70(self) -> np.ndarray:
        """
        RIO69.N70（1970年 産業連関表）を読み込む。
        Returns: Xij0(79, 84)
        """
        return load_matrix(self._path("RIO69.N70"), (79, 84))

    def load_rio69_n80(self) -> np.ndarray:
        """
        RIO69.N80（1980年 産業連関表）を読み込む。
        Returns: Xij1(79, 84)
        """
        return load_matrix(self._path("RIO69.N80"), (79, 84))


# ---------------------------------------------------------------------------
# 診断ツール
# ---------------------------------------------------------------------------

def inspect_file(filepath: str, max_lines: int = 10):
    """
    ファイルの先頭数行を表示して内容を確認する。
    読み込み前のフォーマット確認に使う。
    """
    print(f"\n=== {filepath} の先頭 {max_lines} 行 ===")
    if not os.path.exists(filepath):
        print("  ファイルが存在しません")
        return
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            print(f"  {i+1:3d}: {line.rstrip()}")


def count_values(filepath: str) -> int:
    """ファイル内の数値トークン総数を数える（shape 推定用）"""
    count = 0
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(('*', '#', '!')):
                continue
            count += len(line.split())
    print(f"{filepath}: {count} 個の値")
    print(f"  候補 shape: {count} = {suggest_shapes(count)}")
    return count


def suggest_shapes(n: int) -> list:
    """n を積として表せる行列サイズの候補を返す"""
    shapes = []
    for r in range(1, int(n**0.5) + 1):
        if n % r == 0:
            c = n // r
            shapes.append(f"{r}×{c}")
    return shapes[-6:]  # 大きめの候補を優先表示


# ---------------------------------------------------------------------------
# 使い方の例
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== data_loader.py 使い方デモ ===\n")

    print("【ファイル確認】")
    print("  inspect_file('MTO60.DAT')  # 先頭10行を表示")
    print("  count_values('MTO60.DAT')  # 値の総数と shape 候補を表示\n")

    print("【データ読み込み】")
    print("  from data_loader import DataFiles")
    print("  df = DataFiles(base_dir='C:/data')")
    print("  XIJO = df.load_mto60()   # shape (46,44)")
    print("  XIJ1 = df.load_mto65()   # shape (46,44)")
    print("  Xij0 = df.load_rio69_n70()  # shape (79,84)")
    print("  Xij1 = df.load_rio69_n80()  # shape (79,84)\n")

    print("【汎用読み込み（shape を直接指定）】")
    print("  from data_loader import load_matrix, load_vector")
    print("  A = load_matrix('myfile.dat', (28, 28))")
    print("  v = load_vector('myfile.dat', 28)")
    print("  # Fortran 指数表記 (1.23D+04) は自動変換されます")
    print("  # CSV の場合: load_matrix('file.csv', (46,44), delimiter=',')")
