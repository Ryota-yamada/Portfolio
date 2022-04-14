# Portfolio
実装したツールの例です。  
アップロードしたツールのうち、長期インターン外で実装したツール・業務内で実装したツールをそれぞれ1つずつご紹介させていただきます。長期インターンの業務で作成したツールについてはお客様の秘密保持目的で実際の画像はお見せできません。ご了承ください。
## インターン外で作成したツール例
### SEM画像自動計測ツール
SEM画像内の円柱の径や四角柱の幅を自動で計測するツールを研究室に向けて実装しました。  
手動では1時間半ほどかかる計測作業が3秒程で終わるようになり、現在研究室では必須のツールとなっています。  
### 仕様  
①スケールバー部分をトリミング  
②大津の2値化でSEM画像を2値化  
③P1の面積の60% ～ CP26の面積の140%の範囲外のオブジェクトを2値化画像から除去（画像内に柱のみが映るようになる）  
④輪郭検出  
⑤検出された輪郭のうち、面積の大きさが中央±5個の輪郭（柱）を取り出し、面積の平均値を算出  
⑥直径を算出  
### 環境
前提としてpython環境に「cv2, natsort, yaml, pandas」モジュールが入っているか確かめて下さい。  
cv2のインストール
```
pip install opencv-python
```
natsortのインストール
```
pip install natsort
```
yamlのインストール
```
pip install pyyaml
```
pandasのインストール
```
pip install pandas
```

### 使用方法 
① 各自「get_width_config.yaml」の各設定項目を設定してください。メモ帳などで開けます。変更する部分としては, extension, SEM_5, shape, nm_pixelです。
```
# SEM画像フォルダ　各SEM画像は「CP1.bmp」などメタ原子径が分かるようにする
Image_Dirpath: ./Image/ 
# 画像ファイルの拡張子
extension: tif
# 5号館SEMで撮ったか
SEM_5: True
# メタ原子の形 円: circle  正方形: square
shape: circle
# ImageJのsetscaleで 1nmあたり - pixelを出してください
nm_pixel: 0.1535 
# 出力フォルダ
Out_Dirpath: ./Result/
```
② windows terminalで実行します。本ツールのディレクトリまで移動し、.pyファイル、.yamlファイルの順に入力　後は実行するだけです。
![windowsterminal](https://user-images.githubusercontent.com/75115602/150537147-27237994-d763-43f8-853b-da3332fadd04.png)

③ 結果はResultディレクトリに格納されます。  
・2値化画像(白黒画像)---確認用  
黒くなっている部分は仕様の想定内です。柱同士がつながったりしている部分は自動的に画像から除去しています。（仕様の③）  
![mask_CP20](https://user-images.githubusercontent.com/75115602/163444238-48d1ab2f-ac54-4b65-8ce4-c5dc91f64125.png)
  
・柱幅の計測に用いたメタ原子が色付けされた画像
  
![byouga](https://user-images.githubusercontent.com/75115602/163445587-23234f12-3239-4094-858b-a67db2435106.png)

  
 大きな径の柱も問題なく計測できます。 
 
![byouga_CP20](https://user-images.githubusercontent.com/75115602/163446941-1c99bf7f-92d0-42d3-914c-55fe7242a241.png)

  
・計測結果のcsvファイル　例では先生のSEM画像の計測結果です。高精度に計測できています。  
  
<img src="https://user-images.githubusercontent.com/75115602/163449285-473cbc5b-02ba-4b05-a0a8-9d02323b845b.png" width= "500px">


## インターンの業務で作成したツール例  
以下ではお客様の製品などの秘密保持のため、実際の画像は用いずに仕様などを記述しております。ご了承下さい。  
  
### サブピクセルレベルの位置補正ツール（キーワード：パラボラフィッティング、テンプレートマッチング）  
製品画像を異常検出AIで正確に検査するには、どの画像間でもターゲットの位置が同じであることが求められます。  
そこで、パラボラフィッティングの考えを利用し、ピクセル単位よりもさらに細かい「サブピクセルレベル」でターゲットの位置補正を行うツールを実装しました。  
  
- **仕様**  
  
【基準画像と補正を行いたい画像にテンプレート画像をマッチングさせる。】  
  
<img src="https://user-images.githubusercontent.com/75115602/157582120-27133d26-c902-43a7-aa10-356740707044.png" width="380px">  
  
【最もスコアが高い位置(一番一致する位置)をサブピクセルレベルで推定し、それぞれ取得(x_subpixel, y_subpixel)】  
  
<img src="https://user-images.githubusercontent.com/75115602/157582544-cb26b193-1ae8-4d7d-a9da-a66d5578fdd0.png" width="400px">
  
【(x_subpixel, y_subpixel)から基準画像と位置補正したい画像の移動量を計算。その分を移動させて補正完了】  
