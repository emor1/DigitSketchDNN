"""
誤差逆伝播法を使用したニューラルネットワーク

0から作るディープラーニング5章を参考に実装

The MIT License (MIT)
Copyright (c) 2016 Koki Saitoh
https://github.com/oreilly-japan/deep-learning-from-scratch/blob/master/ch05/two_layer_net.py
"""

import numpy as np
import json
from layers.layers import *
from functions.simple_net import numerical_gradient
from collections import OrderedDict

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01, load=False):
        self.params = {}

        if load :
            with open('data/param.json') as f:
                load_data = json.load(f)
            for key in ('W1', 'b1', 'W2', 'b2'):
                param = load_data[key]
                self.params[key] = np.array(param)
            print("loaded")
        else:
        # 重みとバイアスのパラメータ初期化
            self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
            self.params['b1'] = np.zeros(hidden_size)
            self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
            self.params['b2'] = np.zeros(output_size)
            print("init")

        # レイヤーの作成
        self.layers = OrderedDict() #順番付きの辞書にする
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])       #1層目のAffine
        self.layers['Relu1'] = Relu()       # 1層目のRelu
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        self.lastLayer = SoftMaxWithLoss()

    def save_params(self, json_file):
        network_dict = {}
        for key in ('W1', 'b1', 'W2', 'b2'):
            param = self.params[key]
            print(param[0])
            network_dict[key] = param.tolist()

        with open(json_file, 'w') as f:
            json.dump(network_dict, f, indent=2)

        print("model saved")

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    # x: 入力データ、t；教師データ
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1: t=np.argmax(t, axis=1)

        accuracy = np.sum(y == t) /float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        loss_W = lambda W : self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

    # 誤差逆伝播
    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 設定
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db

        return grads

# 勾配確認
if __name__ == "__main__":
    from dataset.mnist import load_mnist
    
    # 読み込み
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

    x_batch = x_train[:3]
    t_batch = t_train[:3]

    print("Start")
    grad_numerical = network.numerical_gradient(x_batch, t_batch)
    grad_backprop = network.gradient(x_batch, t_batch)

    print("Done")
    for key in grad_numerical.keys():
        diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
        print(key + " : " + str(diff))