import os
import glob
import cv2
import math
import numpy as np

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
    
def detect_subpixel_peak(img, tmp_path):
    # テンプレート画像
    tmp_img = imread(tmp_path, 0)
    tmp_w, tmp_h = tmp_img.shape[::-1]
    
    # 対象画像読み込み
    src = img
    im_w, im_h = src.shape[::-1]
    
    # テンプレートマッチング
    res = cv2.matchTemplate(src, tmp_img, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    # 最大スコアとその位置
    xy = max_val
    x_max, y_max = max_loc
    
    # Parabolic fittingによるSubPixel推定
    if (x_max != 0 and x_max != im_w-1):
        # 前後座標のスコア
        xp = res[y_max][x_max-1]
        xn = res[y_max][x_max+1]
        
        # x方向のSubpixel推定値
        subpix_X = x_max + (xp-xn)/(2.0*(xp-2*xy+xn))
        
    else:
        subpix_X = x_max
    
    if (y_max != 0 and y_max != im_h-1):
        # 前後座標のスコア
        yp = res[y_max-1][x_max]
        yn = res[y_max+1][x_max]
        
        # x方向のSubpixel推定値
        subpix_Y = y_max + (yp-yn)/(2.0*(yp-2*xy+yn))
        
    else:
        subpix_Y = y_max
    
    return subpix_X, subpix_Y
    
    
if __name__ == '__main__':
    # 基準画像のファイルパス
    path_ref = './Shape/'
    files = glob.glob(path_ref+'/*.bmp')
    
    subpix_X, subpix_Y = detect_subpixel_peak(files[1], './template.png')
        
    print ("subpix_x", subpix_X, 'subpix_y', subpix_Y)