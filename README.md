# Portfolio
実装したツールの例です。  
アップロードしたツールのうち、長期インターン外で実装したツール・業務内で実装したツールをそれぞれ1つずつご紹介させていただきます。長期インターンの業務で作成したツールについてはお客様の秘密保持目的で実際の画像はお見せできません。ご了承ください。
## インターン外で作成したツール例
### SEM画像自動計測ツール
SEM画像内の円柱の径や四角柱の幅を自動で計測するツールを研究室に向けて実装しました。  
手動では1時間半ほどかかる計測作業が3秒程で終わるようになり、現在研究室では必須のツールとなっています。  
- **仕様**  
  
  スケールバー部分をトリミング → 大津の2値化でSEM画像を2値化 → メタ原子のエッジ検出を行い, 内部の面積を算出 → 平均の面積を算出 → 直径算出  
    
- **機能**  
  
【実行パラメータをyamlファイルで設定できる】  
config.yaml  
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
  
【コマンド実行】  
  
<img src="https://user-images.githubusercontent.com/75115602/150537147-27237994-d763-43f8-853b-da3332fadd04.png" width="380px">
  
【SEM画像内の構造の輪郭を複数検出】  
![fitting](https://user-images.githubusercontent.com/75115602/150671276-47b905ee-35e4-4012-bc6c-c1eb7efb6c5e.png)  
  
【全ての画像について直径の計測結果をまとめ、出力】  
<img src="https://user-images.githubusercontent.com/75115602/150671293-cd4598e0-e72d-4ed0-a428-3b47bf062e8c.png" width="350px">  
  
【正方形の幅も計測できる】  
config.yamlのパラメータを変更すると正方形の幅も自動計測できる。  
  
<img src="https://user-images.githubusercontent.com/75115602/150893197-89a7e244-3d63-48fc-bfe2-2fbfd85c3d30.jpg" width="300px">  
  
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
