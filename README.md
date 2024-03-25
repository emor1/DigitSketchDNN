# DNN Digit Predictor

![DNN](img/sketch_dnn.gif)

DNN Digit Predictorは[「ゼロから作るDeep Learning」](https://www.amazon.co.jp/dp/4873117585?ref_=cm_sw_r_cp_ud_dp_D4WTQD6YZC7XPRNG5K9V)で学んだディープニューラルネットワークを用いて、キャンバスに描かれた数字を推定するPygameのアプリケーションです。

画面左の黒いキャンバスにマウスで数字を描画すると、画面左に描かれた数字を認識した結果が表示されます。

## Dependency

* Language
  * Python 3.10
* Libraries
  * Numpy 1.26.4
  * Pygame 2.5.2

## Getting Started

`$ pip install numpy pygame`

`$ python main.py`

## Directory

├── README.md  
├── data  
│   └── param.json  
├── functions  
│   ├── function.py  
│   └── simple_net.py  
├── img  
│   └── sketch_dnn.gif  
├── layers  
│   ├── layers.py  
│   └── two_layer_net_backprop.py  
└── main.py

## References

* [「ゼロから作るDeep Learning」](https://www.amazon.co.jp/dp/4873117585?ref_=cm_sw_r_cp_ud_dp_D4WTQD6YZC7XPRNG5K9V)
