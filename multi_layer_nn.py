import numpy as np
from layers import Affine, SoftmaxWithLoss, Relu
from collections import OrderedDict

#[入力層]784次元 → [隠れ層]50ユニット → [出力層]10ユニット（分類クラス数）
class MultiLayerNN:
    def __init__(self, input_size = 784, hidden_sizes = [50], output_size = 10):
        # 784次元の入力を50個の隠れ層ニューロンに伝える(784x50)の重み行列
        # self.params['W1'] = np.random.randn(input_size, hidden_sizes) * 0.01 
        # #各隠れ層ニューロンに加えるバイアス項（50要素の1次元配列）
        # self.params['b1'] = np.zeros(hidden_sizes) 
        # #隠れ層から出力層（10クラス）へ接続する(50x10)の重み行列
        # self.params['W2'] = np.random.randn(hidden_sizes, output_size) * 0.01
        # #各出力ニューロンに加えるバイアス項(10要素の1次元配列)
        # self.params['b2'] = np.zeros(output_size) 
        
        # レイヤーの層数を計算
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.params = {} #各層における重みベクトルとバイアスベクトルを保持するディクショナリ
        # 例:
        # key = 'W1', val = 第1層の重み行列（NumPy配列, shape: 入力×出力）
        # key = 'b1', val = 第1層のバイアスベクトル（NumPy配列, shape: 出力）
        # key = 'W2', val = 第2層の重み行列, ...
        # key = 'b2', val = 第2層のバイアスベクトル, ...
        self.num_layers = len(layer_sizes) - 1
        self.activation = Relu
        
        for i in range(self.num_layers):
            in_size = layer_sizes[i] #第i層のニューロン数 ex. 入力層(784)->中間層(50)ならlayer_sizes[0]=784
            out_size = layer_sizes[i + 1] #i+1層(次の層)のニューロン数 ex. 入力層(784)->中間層(50)ならlayer_sizes[0]=784, layer_sizes[1]=50
            self.params[f'W{i + 1}'] = np.random.randn(in_size, out_size) * 0.01 #ex. 入力層(784)->中間層(50)なら(784x50)の重み行列
            self.params[f'b{i + 1}'] = np.zeros(out_size) #第i+1層のバイアス項 ex. 入力層(784)->中間層(50)なら要素数50の配列
        
        # レイヤーを順伝播＆逆伝播用に OrderedDict に格納
        self.layers = OrderedDict()
        # 隠れ層部分：Affine レイヤー → ReLU レイヤー を交互に追加
        for i in range(self.num_layers - 1):
            # Affine レイヤー: 重み W{i+1}, バイアス b{i+1} を使用
            self.layers[f'Affine{i + 1}'] = Affine(self.params[f'W{i+1}'], self.params[f'b{i+1}'])
            # 活性化関数 ReLU レイヤー
            self.layers[f'ReLU{i+1}'] = Relu()
        # 出力層の直前まで作成したら、最後の Affine レイヤーを追加
        last = self.num_layers
        # 最終 Affine レイヤー: 重み W{last}, バイアス b{last} を使用 
        self.layers[f'Affine{last}'] = Affine(self.params[f'W{last}'], self.params[f'b{last}'])
        
        # 損失計算用の SoftmaxWithLoss レイヤーを最後に設定
        self.last_layer = SoftmaxWithLoss()
    
    # 順伝播処理を行うメソッド
    # 入力 x を順番に各レイヤーに渡して出力を得る
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)  # 各レイヤーで順伝播
        return x  # 出力層のスコア（softmax適用前）
    
    # 損失を計算するメソッド（Softmax + 交差エントロピー誤差）
    # x: 入力データ、t: 正解ラベル（one-hot またはラベル整数）
    def loss(self, x, t):
        y = self.predict(x)  # ネットワークの出力を得る
        return self.last_layer.forward(y, t)  # 損失を計算（Softmax + Loss）
    
    # 勾配計算（逆伝播を用いて各パラメータの勾配を求める）
    def gradient(self, x, t):
        # 1. 順伝播して損失を計算（Softmax + CrossEntropy）
        self.loss(x, t)

        # 2. 出力層から逆伝播を開始（dout=1は損失の微分の初期値）
        dout = self.last_layer.backward(1)

        # 3. 隠れ層を逆順にして順に逆伝播（Affine→ReLUの順でbackward）
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 4. 各Affine層から勾配を取り出して辞書にまとめる
        grads = {}
        for i in range(1, self.num_layers + 1):
            affine = self.layers[f'Affine{i}']
            grads[f'W{i}'] = affine.dW  # 第i層における重みの勾配
            grads[f'b{i}'] = affine.db  # 第i層におけるバイアスの勾配

        return grads
    
    def accuracy(self, x, t):  # 予測精度（正解率）を計算するメソッド
        y = self.predict(x) # 予測結果を計算
        y_pred = np.argmax(y, axis=1) # 予測結果から最大確率のインデックスを取得（予測ラベル）
        if t.ndim != 1: # tが one-hot 表現の場合のみ次元を変換
            t_label = np.argmax(t, axis=1)
        else:
            t_label = t
        return np.sum(y_pred == t_label) / float(x.shape[0])  # 正解ラベルとの一致率を計算して返す