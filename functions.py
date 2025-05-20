
import numpy as np

# シグモイド関数: 出力を0〜1の範囲にマッピングする活性化関数
# 入力 x が大きいほど出力は1に近づき、小さいほど0に近づく
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ReLU関数: 入力が0以下のとき0、0より大きいときはそのまま出力する活性化関数
# neural network の隠れ層でよく使われる
def relu(x):
    # return 0 if x < 0 else x
    return np.maximum(0, x)
    
# ソフトマックス関数: 出力を全要素の指数関数比に変換し、
# 要素の総和が1になるように正規化する
# 分類問題の出力層で確率として解釈するために使用
def softmax(x):
    # exp_x = np.exp(x)    # 各要素を指数関数で変換（分子）
    # sum_exp_x = np.sum(exp_x)    # 全要素の和（分母）
    # return exp_x / sum_exp_x    # 正規化して確率分布を得る
    x = x - np.max(x, axis=-1, keepdims=True)   # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)


# 交差エントロピー誤差関数: softmax 出力 y と
# 正解ラベル t との間の損失（平均負の対数尤度）を計算
def cross_entropy_error(y, t):
    # 1次元データの場合は2次元に拡張（バッチ対応）
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 教師ラベルが one-hot-vector なら、正解ラベルのインデックスに変換
    if t.size == y.size:
        t = t.argmax(axis=1)

    # バッチサイズ分の平均負の対数尤度を返す
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

# softmax と交差エントロピーを組み合わせた損失関数（推論と損失計算をまとめて呼び出し可能）
def softmax_loss(X, t):
    # まず X に softmax を適用して確率 y を得る
    y = softmax(X)
    # 得られた確率 y とラベル t で交差エントロピー誤差を計算
    return cross_entropy_error(y, t)