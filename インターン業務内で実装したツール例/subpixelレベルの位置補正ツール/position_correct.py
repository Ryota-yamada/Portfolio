from subpixel_peak_detection import imread, imwrite, detect_subpixel_peak
import numpy as np
import cv2
import glob
import os
import time


# テンプレートマッチング用のマスキング画像生成して、マスキング画像と元画像返す
def Masking(filepath):
    img = imread(filename=filepath, flags=0, dtype= np.uint8)
    img_h, img_w = img.shape
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.rectangle(mask, (70, 2730),(920, 3190), color=255, thickness=-1)
    img_masking = cv2.bitwise_and(img, mask)
    return img_masking, img

# 基準座標との差分を平行移動
def Move(img, X_st, Y_st, X, Y):
    img_w, img_h = img.shape[::-1]
    X_move = X_st - X
    Y_move = Y_st - Y
    img_correct =  cv2.warpAffine(img, np.array([[1,0,X_move],[0,1,Y_move]],dtype=np.float32), (img_w, img_h))
    return img_correct
    

def Main():
    filepathlist = glob.glob("../../detaset/OK/*.bmp")
    tmp_path = "../../detaset/template/template.bmp"
    os.makedirs("../../detaset_hosei",exist_ok=True)
    out_path = "../../detaset_hosei/"
    # 位置補正の基準となるX, Y座標取得(テンプレート画像の親画像をマッチングに使う。template画像と同じフォルダに格納。)
    img_standard = imread("../../detaset/template/standard.bmp",0)
    X_st, Y_st = detect_subpixel_peak(img = img_standard, tmp_path = tmp_path)
    for filepath in filepathlist:
        img_masking, img = Masking(filepath)
        X, Y = detect_subpixel_peak(img = img_masking, tmp_path = tmp_path)
        img_correct = Move(img = img, X_st = X_st, Y_st = Y_st, X = X, Y = Y)
        filename = str(os.path.basename(filepath))
        imwrite(filename=out_path + filename, img = img_correct , params=None)

time_start = time.perf_counter()
Main()
time_end = time.perf_counter()    
time_pass = time_end - time_start   
print(f"実行時間：{time_pass}")