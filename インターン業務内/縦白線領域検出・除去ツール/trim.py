import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import os
import sys
import time

def imread(filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8):
    try:
        n = np.fromfile(filename, dtype)
        img = cv2.imdecode(n, flags)
        return img
    except Exception as e:
        print(e)
        return None

def imwrite(filename, img, params=None):
    try:
        ext = os.path.splitext(filename)[1]
        result, n = cv2.imencode(ext, img, params)
        
        if result:
            with open(filename, mode='w+b') as f:
                n.tofile(f)
            return True
        else:
            return False
    except Exception as e:
        print(e)
        return False

def trim(filepath,w):
    filename = os.path.basename(filepath)
    result, ex = os.path.splitext(filename)
    os.makedirs("./Image_detect", exist_ok=True)
    os.makedirs("./Image_trim", exist_ok=True)
    # ----1段階目トリミング----
    img = imread(filepath)
    img_gray = imread(filepath, flags=0)
    img_trim1 = img_gray[830:2700,46:6700]
    img_trim1_byoga = img[830:2700,46:6700]
    # ----2段階目トリミング----
    th,img_bin = cv2.threshold(img_trim1, 44.2, 255, cv2.THRESH_BINARY)
    # 直線の検出
    lines = cv2.HoughLinesP(img_bin, rho=1, theta=np.pi/360, threshold=100, minLineLength=120, maxLineGap=5)
    # y座標の範囲が120pixel以上のものを取り出す。→　取り出されたものは縦線なはず。
    line_vert_list = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(y2 - y1) >= 120:
            line_vert_list.append(line)
    # 取り出した縦線を描画
    for line_vert in line_vert_list:
        x1, y1, x2, y2 = line_vert[0]
        cv2.line(img_trim1_byoga, (x1, y1), (x2, y2), color=(0,255,0))
    # line_vert_np → [[[x1, y1, x2, y2], [.....], [.....]]]
    line_vert_np = np.array(line_vert_list)
    X_list = line_vert_np[:, 0, 0]
    X_list_1 = X_list[(1300 < X_list)&(X_list < 1800)] 
    X_list_2 = X_list[(2900 < X_list)&(X_list < 3400)] 
    X_list_3 = X_list[(4500 < X_list)&(X_list < 5000)] 
    X_list_4 = X_list[(6000 < X_list)&(X_list < 6500)] 
    if any([len(X_list_1) == 0, len(X_list_2) == 0, len(X_list_3) == 0, len(X_list_4) == 0]):
        print(f"{filename}の縦線検出に失敗しました。")
        sys.exit()
    # それぞれの領域内で検出された縦線のX座標の平均値算出
    X_1_mean = int(X_list_1.mean())
    X_2_mean = int(X_list_2.mean())
    X_3_mean = int(X_list_3.mean())
    X_4_mean = int(X_list_4.mean())
    # 平均X座標-25 ～ 平均X座標+25の範囲で列ごと削除
    X_1_del = list(np.arange(X_1_mean-25, X_1_mean+26))
    X_2_del = list(np.arange(X_2_mean-25, X_2_mean+26))
    X_3_del = list(np.arange(X_3_mean-25, X_3_mean+26))
    X_4_del = list(np.arange(X_4_mean-25, X_4_mean+26))
    del_list = X_1_del + X_2_del + X_3_del + X_4_del
    img_trim2 = np.delete(img_trim1, del_list, 1)
    # 検出結果を描画した画像を保存
    imwrite(filename = os.path.join("./Image_detect", filename), img=img_trim1_byoga)
    #----3段階トリミング----
    img_h, img_w = img_trim2.shape
    Start = 0
    img_num = int(img_w/(w/2)) -1
    for num in range(img_num):
        num = num + 1
        img_trim3 = img_trim2[:, int(Start):int(Start + w)]
        Start = Start + (w/2)
        imwrite(filename = os.path.join("./Image_trim", result + f"_{str(num).zfill(3)}" + ex), img = img_trim3)

def Main():
    filepathlist = glob.glob("../../detaset_hosei/OK/*.bmp")
    print(filepathlist)
    for filepath in filepathlist:
        trim(filepath,w=2150)

if __name__ == "__main__":
    time_start = time.perf_counter()
    Main()
    time_end = time.perf_counter()    
    time_pass = time_end - time_start   
    print(f"実行時間：{time_pass}")
    

        