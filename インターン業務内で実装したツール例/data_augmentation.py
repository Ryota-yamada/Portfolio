import argparse
import sys
import os
import time
from tqdm import tqdm
import random
import numpy as np
import pandas as pd
import cv2
import datetime

'''
# 使用例
「平行移動の変位を与えた画像を100枚生成したいとき」
python data_Augmentation.py -i ./dataset/sample.png -o ./dest -m ./dataset/DetectArea.png -n 100 --shift
-i: 入力画像パス
-o: 出力フォルダ
-m: マスク画像
-n: 出力枚数
--shift: 平行移動
--rotation: 回転
--scale: スケール
--distortion: 台形変形
「平行移動で20pixel分の変位を与えた画像を100枚生成したいとき」
python data_Augmentation.py -i ./dataset/sample.png -o ./dest -m ./dataset/DetectArea.png -n 100 -l 20 --shift
-l (--limitation)について
(例) 「-l 20」のとき
shit: 20pixel以内で平行移動
rotation: ±20°の範囲で回転
scale: 膨張縮小の中心 (ターゲットの重心) からの最遠点の移動量が±20pixel以内となるような倍率で変形。
distortion: ±20pixel分の台形変形量
# 出力csvについて
filename: 出力画像ファイル名
shift: 平行移動量
rotation: 回転角度
scale: 膨張縮小中心から最遠点の移動量
distortion: 台形変形量
processing_time: 一枚あたりの処理時間
'''


parser = argparse.ArgumentParser()
parser.add_argument("-i","--input", type = str, help="input image path", required=True)
parser.add_argument("-o","--output", type = str, help="output folder path", required=True)
parser.add_argument("-m","--mask", type = str, help="mask image path", required=True)
parser.add_argument("-n", type = int, default = 50, help="output num")
parser.add_argument("-l", "--limitation", type = int, default=None, help = "change limit")
parser.add_argument("--shift", action="store_true", help="image translation")
parser.add_argument("--rotation", action="store_true", help="image rotation")
parser.add_argument("--scale", action="store_true", help="enlarge or reduce")
parser.add_argument("--distortion", action="store_true", help="homograph")

args = parser.parse_args()

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
    
def getMaskPosition(mask):
    contours, hierarchy= cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours ,key=lambda x: cv2.contourArea(x), reverse=True)
    Xmax_M = contours[0][:,0,0].max()
    Xmin_M = contours[0][:,0,0].min()
    Ymax_M = contours[0][:,0,1].max()
    Ymin_M = contours[0][:,0,1].min()
    return Xmax_M, Xmin_M, Ymax_M, Ymin_M, contours

# 回転したマスク画像から画像境界にターゲットが接していないかを判断する関数。接していないときTrue返す。
def check_boundary(mask_rotation, w, h):
    left_sum = np.sum(mask_rotation[:, 0])
    right_sum = np.sum(mask_rotation[:, w-1])
    top_sum = np.sum(mask_rotation[0, :])
    bottom_sum = np.sum(mask_rotation[h-1, :])
    if all([left_sum == 0, right_sum == 0, top_sum == 0, bottom_sum == 0]):
        return True
    else:
        return False

# --limitation指定時、--shiftにてX方向、Y方向の移動量をランダムに選択し返す関数
def getdisplacement(Xshift_lim, Yshift_lim, shift_max):
    
    if args.limitation < shift_max:
        if args.limitation < Xshift_lim:
            X_shift = random.choice(np.arange(0, args.limitation + 1, 1)) # X方向移動量決定（絶対値）
        else:
            X_shift = random.choice(np.arange(0, Xshift_lim + 1, 1))
        Y_shift = random.choice(np.arange(0, int(np.sqrt(args.limitation**2 - X_shift**2)) + 1, 1)) # Y方向移動量決定（絶対値）
    
    else:
        X_shift = random.choice(np.arange(0, Xshift_lim + 1, 1))
        Y_shift = random.choice(np.arange(0, Yshift_lim + 1, 1))
    
    return X_shift, Y_shift


def padding(img, mask, padding_w_list):
    padding_top, padding_bottom, padding_left, padding_right = padding_w_list
    img_padding = cv2.copyMakeBorder(img, padding_top, padding_bottom, padding_left, padding_right, cv2.BORDER_REPLICATE)
    mask_padding = cv2.copyMakeBorder(mask, padding_top, padding_bottom, padding_left, padding_right, cv2.BORDER_REPLICATE) 
        
    return img_padding, mask_padding


def shift(img, mask):
    
    h, w = mask.shape
    
    Xmax_M, Xmin_M, Ymax_M, Ymin_M, contours = getMaskPosition(mask)
    #ターゲットが画像境界に接することのない最大移動量。(lim:limit, pl:プラス方向, mi:マイナス方向)
    Xshift_lim_pl = w-1 - Xmax_M
    Xshift_lim_mi = -1*Xmin_M
    Yshift_lim_pl = h-1 - Ymax_M
    Yshift_lim_mi = -1*Ymin_M

    # --limitationが指定されている場合。
    if args.limitation != None:
        
        # 指定した距離の移動は右上、右下、左上、左下のどの方向で可能か
        direction_list = ["TopRight", "BottomRight", "TopLeft", "BottomLeft"]
        
        # どの方向に移動させるかランダムに決定
        direction = random.choice(direction_list)
        
        if direction == "TopRight":
            Xshift_lim = abs(Xshift_lim_pl)
            Y_shift_lim = abs(Yshift_lim_mi)
            shift_max = np.sqrt(Xshift_lim_pl**2 + Yshift_lim_mi**2)
            X_shift, Y_shift = getdisplacement(Xshift_lim, Y_shift_lim, shift_max)
            Y_shift = -1*Y_shift
            
        elif direction == "BottomRight":
            Xshift_lim = abs(Xshift_lim_pl)
            Y_shift_lim = abs(Yshift_lim_pl)
            shift_max = np.sqrt(Xshift_lim_pl**2 + Yshift_lim_pl**2)
            X_shift, Y_shift = getdisplacement(Xshift_lim, Y_shift_lim, shift_max)
            
        elif direction == "TopLeft":
            Xshift_lim = abs(Xshift_lim_mi)
            Y_shift_lim = abs(Yshift_lim_mi) 
            shift_max = np.sqrt(Xshift_lim_mi**2 + Yshift_lim_mi**2)
            X_shift, Y_shift = getdisplacement(Xshift_lim, Y_shift_lim, shift_max)
            X_shift = -1*X_shift
            Y_shift = -1*Y_shift
            
        elif direction == "BottomLeft":
            Xshift_lim = abs(Xshift_lim_mi)
            Y_shift_lim = abs(Yshift_lim_pl)
            shift_max = np.sqrt(Xshift_lim_mi**2 + Yshift_lim_pl**2)
            X_shift, Y_shift = getdisplacement(Xshift_lim, Y_shift_lim, shift_max)
            X_shift = -1*X_shift
            
    # --limitationが指定なしの場合。
    else:
        if (Xshift_lim_pl == 0) and (Xshift_lim_mi != 0):
            randomlist_Xshift = list(range(-1, Xshift_lim_mi  , -1))
        elif (Xshift_lim_pl != 0) and (Xshift_lim_mi == 0):
            randomlist_Xshift = list(range(0, Xshift_lim_pl , 1))
        elif (Xshift_lim_pl == 0) and (Xshift_lim_mi == 0):
            randomlist_Xshift = [0]
        else:
            randomlist_Xshift = list(range(0, Xshift_lim_pl , 1)) + list(range(-1, Xshift_lim_mi , -1))
        
        if (Yshift_lim_pl == 0) and (Yshift_lim_mi != 0):
            randomlist_Yshift = list(range(-1, Yshift_lim_mi  , -1))
        elif (Yshift_lim_pl != 0) and (Yshift_lim_mi == 0):
            randomlist_Yshift = list(range(0, Yshift_lim_pl , 1))
        elif (Yshift_lim_pl == 0) and (Yshift_lim_mi == 0):
            randomlist_Yshift = [0]
        else:
            randomlist_Yshift = list(range(0, Yshift_lim_pl , 1)) + list(range(-1, Yshift_lim_mi , -1))
                
        # 移動量(ランダムに取り出し)
        X_shift = random.choice(randomlist_Xshift)
        Y_shift = random.choice(randomlist_Yshift)
    
    # X_shiftとY_shiftのうち、大きい方の分だけ、元画像を一周パディング
    padding_w = max([abs(X_shift), abs(Y_shift)])
    padding_w_list = [padding_w, padding_w, padding_w, padding_w]
    img_padding, mask_padding = padding(img, mask, padding_w_list)
    h_padding, w_padding = mask_padding.shape
    
    # 平行移動の変換行列を作成
    afin_matrix = np.float32([[1,0,X_shift],[0,1,Y_shift]])
    
    # パディングした画像にアファイン変換適用
    img_padding_shift = cv2.warpAffine(img_padding, afin_matrix, (w_padding, h_padding))
    mask_padding_shift = cv2.warpAffine(mask_padding, afin_matrix, (w_padding, h_padding))
    
    # パディング分をトリミングして元画像サイズへ
    img_shift = img_padding_shift[padding_w  : padding_w + h, padding_w  : padding_w + w]
    mask_shift = mask_padding_shift[padding_w  : padding_w + h, padding_w  : padding_w + w]
    
    # もし元画像とサイズが変わっていたら、実行停止
    if mask.shape != mask_shift.shape:
        print("error: Unintentional image resizing was detected.")
        sys.exit()
        
    return img_shift, mask_shift, np.sqrt(X_shift**2 + Y_shift**2)
    
def rotation(img, mask):
    
    h, w = mask.shape
    Xmax_M, Xmin_M, Ymax_M, Ymin_M, contours = getMaskPosition(mask)
    mu = cv2.moments(contours[0])
    Px, Py = int(mu["m10"]/mu["m00"]) , int(mu["m01"]/mu["m00"]) # 回転中心座標
    
    if args.limitation == None:
        angle_max = 180
    else:
        angle_max = args.limitation
        
    # はみ出さずに回転できる角度があるか調査
    for angle_test in np.arange(-1*angle_max, angle_max + 1, 1):
        rot_matlix_test = cv2.getRotationMatrix2D((Px, Py), angle_test, 1)
        mask_rotation_test = cv2.warpAffine(mask, rot_matlix_test, (w, h))
        # 4辺境界それぞれの合計輝度が0であれば回転可能。
        rotatable = check_boundary(mask_rotation_test, w, h)
        if rotatable:
            break
    
    if rotatable:
        # ランダムで角度を設定
        selecting = True
        while selecting:
            angle = random.choice(np.arange(-1*angle_max, angle_max + 1, 1))
            rot_matlix_test = cv2.getRotationMatrix2D((Px, Py), angle, 1)
            mask_rotation_test = cv2.warpAffine(mask, rot_matlix_test, (w, h))
            # 4辺境界それぞれの合計輝度が0であれば、回転可能。
            decision_angle = check_boundary(mask_rotation_test, w, h) # 角度が決定したか。
            if decision_angle:
                selecting = False # 角度が決定したなら、ランダム取り出し終了。
        
        # パディング量算出
        l_max = max([Px, Py, (w-1 - Px), (h-1 - Py)])
        radius = int(np.round(np.sqrt(2)*l_max, 0))
        padding_top = radius - Py
        padding_bottom = radius - (h-1 - Py)
        padding_left = radius - Px
        padding_right = radius - (w-1 - Px)
        # パディング
        padding_w_list = [padding_top, padding_bottom, padding_left, padding_right]
        img_padding, mask_padding = padding(img, mask, padding_w_list)
        h_p, w_p = mask_padding.shape
        _, _, _, _, contours_p = getMaskPosition(mask_padding)
        mu_p = cv2.moments(contours_p[0])
        Px_p, Py_p = int(mu_p["m10"]/mu_p["m00"]) , int(mu_p["m01"]/mu_p["m00"])
        # 抽選した角度で回転 
        rot_matlix = cv2.getRotationMatrix2D((Px_p, Py_p), angle, 1)
        # アフィン変換適用
        img_rotation = cv2.warpAffine(img_padding, rot_matlix, (w_p, h_p))
        mask_rotation = cv2.warpAffine(mask_padding, rot_matlix, (w_p, h_p))
        # 重心を基準にトリミング（元画像と同じ位置関係になるはず）
        img_rotation = img_rotation[Py_p - Py:Py_p + (h-1 - Py) + 1, Px_p - Px:Px_p + (w-1 - Px) + 1]
        mask_rotation = mask_rotation[Py_p - Py:Py_p + (h-1 - Py) + 1, Px_p - Px:Px_p + (w-1 - Px) + 1]
        # もし元画像とサイズが変わっていたら、実行停止
        if mask.shape != mask_rotation.shape:
            print("size chenge error")
            sys.exit()
    else:
        img_rotation, mask_rotation = img, mask 
    
    return img_rotation, mask_rotation, angle

def scale(img, mask):
    
    h, w = mask.shape
    Xmax_M, Xmin_M, Ymax_M, Ymin_M, contours = getMaskPosition(mask)
    mu = cv2.moments(contours[0])
    Px, Py = int(mu["m10"]/mu["m00"]) , int(mu["m01"]/mu["m00"]) # マスク外側輪郭の重心座標
    
    # 重心から, マスク領域でx,yが最大最小の4点までのピクセル数(始点は無視)
    X_len_max_c = Xmax_M - Px 
    X_len_min_c = Px - Xmin_M 
    Y_len_max_c = Ymax_M - Py 
    Y_len_min_c = Py - Ymin_M
    
    # Xmax_M, Xmin_M, Ymax_M, Ymin_Mが画像境界に接しだすのはそれぞれ倍率何倍のときか（重心中心)
    X_max_scale = (w-1 - Px)/X_len_max_c
    X_min_scale = (Px)/X_len_min_c
    Y_max_scale = (h-1 - Py)/Y_len_max_c
    Y_min_scale = (Py)/Y_len_min_c
    
    # 上記の4倍率の中で最小なものを取り出す（4点のうち、1点でも境界に接してしまう倍率）
    scale_can = min([X_max_scale, X_min_scale, Y_max_scale, Y_min_scale])
    
    # 画像中の重心から最遠点までの距離算出
    length_far = max([np.sqrt(Px**2+Py**2), np.sqrt((w-1-Px)**2+Py**2), np.sqrt((w-1-Px)**2+(h-1-Py)**2), np.sqrt(Px**2+(h-1-Py)**2)])
    
    if args.limitation != None:
        scale_far = 1 + args.limitation/length_far # 画像中の最遠点にて指定したピクセル分移動するときの倍率
        # 画像中の最遠点にて指定したピクセル分移動するときの倍率(scale_far)が倍率の上限未満のとき、最大倍率はscale_farとする。
        if scale_far < scale_can:
            scale_max = scale_far
        else:
            scale_max = scale_can
        if scale_max < 2:
            scale_min = 2 - scale_max
        else:
            scale_min = 0.1
    else:
        scale_max = scale_can
        if scale_max < 2:
            scale_min = 2 - scale_max
        else:
            scale_min = 0.1
            
    randomlist_scale = np.arange(np.round(scale_min, 3)*1000 + 1, np.round(scale_max, 3)*1000 , 1)/1000
    scale = random.choice(randomlist_scale)
    
    # 倍率1未満の時パディング処理必要
    if (scale < 1):  
        # 推定される余白のピクセル数
        bl_top = int(np.round(((1 - scale)*(Py)), 0)) #　上の余白ピクセル数
        bl_bottom = int(np.round(((1 - scale)*(h - 1  - Py)), 0)) #　下の余白ピクセル数
        bl_left = int(np.round(((1 - scale)*(Px)), 0)) #  左の余白ピクセル数
        bl_right = int(np.round(((1 - scale)*(w - 1 - Px)), 0)) #  右の余白ピクセル数
    
        # パディング要素を用意（意図しないピクセル数のずれ（おそらく少数点の丸めの影響？）を防止するため、余分に1ピクセル多くトリミング＆パディング）
        padding_w_list = [bl_top + 1, bl_bottom + 1, bl_left + 1, bl_right + 1]
        
        # 元画像をスケーリング
        scale_matlix = cv2.getRotationMatrix2D((Px, Py), 0, scale)
        img_scale = cv2.warpAffine(img, scale_matlix, (w, h))
        mask_scale = cv2.warpAffine(mask, scale_matlix, (w, h))
        
        # 余白部分をトリミング
        img_scale = img_scale[bl_top + 1:h-1 - bl_bottom - 1 + 1, bl_left + 1:w-1 - bl_right - 1 + 1]
        mask_scale = mask_scale[bl_top + 1:h-1 - bl_bottom - 1 + 1, bl_left + 1:w-1 - bl_right - 1 + 1]
        
        # 生じた余白にパディング
        img_scale, mask_scale = padding(img_scale, mask_scale, padding_w_list)
        # もし元画像とサイズが変わっていたら、実行停止
        if mask.shape != mask_scale.shape:
            print("error: Unintentional image resizing was detected.")
            sys.exit()
    else:   
        scale_matlix = cv2.getRotationMatrix2D((Px, Py), 0, scale)
        img_scale = cv2.warpAffine(img, scale_matlix, (w, h))
        mask_scale = cv2.warpAffine(mask, scale_matlix, (w, h))
    
    return img_scale, mask_scale, (scale-1)*length_far

def distortion(img, mask):
    h, w = mask.shape
    Xmax_M, Xmin_M, Ymax_M, Ymin_M, contours = getMaskPosition(mask)
    # マスクの外接短形4点を取得:左上の点から半時計回りに(X1, Y1), (X2, Y2), (X3, Y3), (X4, Y4)
    X1_ori, Y1_ori, w_m, h_m = cv2.boundingRect(contours[0]) 
    X2_ori = X1_ori
    X3_ori = X1_ori + w_m
    X4_ori = X3_ori
    Y2_ori = Y1_ori + h_m
    Y3_ori = Y2_ori
    Y4_ori = Y1_ori
    
    # 外接短形の中心点
    X_c = int(np.round((X1_ori+X4_ori)/2, 0))
    Y_c = int(np.round((Y1_ori+Y2_ori)/2, 0))
    
    source_points = np.array([[X1_ori, Y1_ori], [X2_ori, Y2_ori], [X3_ori, Y3_ori], [X4_ori, Y4_ori]], dtype=np.float32)
    
    # 台形変換の仕様
    pattern_list = ["pattern1", "pattern2", "pattern3", "pattern4"]
    type_distortion = random.choice(pattern_list) 
    if (type_distortion == "pattern1")or(type_distortion == "pattern2"):
        if X_c < int(np.round((w-1)/2, 0)):
            d_border_max = X2_ori
        else:
            d_border_max = w-1 - X3_ori
        if (w_m/2) > d_border_max:
            d_max = d_border_max
        else:
            d_max = int(np.round(w_m/2, 0))
        d = random.choice(range(1, d_max, 1))
        if type_distortion == "pattern1":
            X1, Y1 = X1_ori + d, Y1_ori
            X2, Y2 = X2_ori - d, Y2_ori
            X3, Y3 = X3_ori + d, Y3_ori
            X4, Y4 = X4_ori - d, Y4_ori
        elif type_distortion == "pattern2":
            X1, Y1 = X1_ori - d, Y1_ori
            X2, Y2 = X2_ori + d, Y2_ori
            X3, Y3 = X3_ori - d, Y3_ori
            X4, Y4 = X4_ori + d, Y4_ori

    if (type_distortion == "pattern3")or(type_distortion == "pattern4"):
        if Y_c < int(np.round((h-1)/2, 0)):
            d_border_max = Y1_ori
        else:
            d_border_max = h-1 - Y2_ori
        if (h_m/2) > d_border_max:
            d_max = d_border_max
        else:
            d_max = int(np.round(h_m/2, 0))
        d = random.choice(range(1, d_max, 1))
        if type_distortion == "pattern3":
            X1, Y1 = X1_ori, Y1_ori + d
            X2, Y2 = X2_ori, Y2_ori - d
            X3, Y3 = X3_ori, Y3_ori + d
            X4, Y4 = X4_ori, Y4_ori - d
        elif type_distortion == "pattern4":
            X1, Y1 = X1_ori, Y1_ori -d
            X2, Y2 = X2_ori, Y2_ori + d
            X3, Y3 = X3_ori, Y3_ori - d
            X4, Y4 = X4_ori, Y4_ori + d 

    # 変換後の4点
    target_points = np.array([[X1, Y1], [X2, Y2], [X3, Y3], [X4, Y4]], dtype=np.float32)
    #　変換行列計算
    mat = cv2.getPerspectiveTransform(source_points, target_points)
    # 射影変換
    perspective_img = cv2.warpPerspective(img, mat, (w, h))
    perspective_mask = cv2.warpPerspective(mask, mat, (w, h))
    
    return perspective_img, perspective_mask


def main():
    # コマンド引数の適切性確認：オーギュメンテーション手法が何も指定されていないとき
    if not any([args.shift, args.rotation, args.scale, args.distortion]):
        print("No augmentation method specified.(ex. -i, -o, -m)")
        sys.exit()
    
    FileName = os.path.basename(args.input)
    Name, ext = os.path.splitext(FileName)
    # 出力フォルダ作成
    os.makedirs(args.output, exist_ok=True)
    
    FileName_out_list = []
    shift_list = []
    rotation_list = []
    scale_list = []
    distortion_list = []
    processingtime_list = []
    
    for n in tqdm(range(args.n)):
        img = imread(args.input)
        mask = imread(args.mask, 0)
        
        start_time = time.perf_counter()
        
        Augmentation_key_list = []
        
        if args.shift:
            img, mask, shift_result = shift(img, mask)
            Augmentation_key_list.append("shift")
            shift_list.append(shift_result)
        else:
            shift_list.append(0)
        
        if args.rotation:
            img, mask, angle = rotation(img, mask)
            Augmentation_key_list.append("rotation")
            rotation_list.append(angle)
        else:
            rotation_list.append(0)
        
        if args.scale:
            img, mask, scale_result = scale(img, mask)
            Augmentation_key_list.append("scale")
            scale_list.append(scale_result)
        else:
            scale_list.append(0)
        
        if args.distortion:
            img, mask = distortion(img, mask)
            Augmentation_key_list.append("distortion")
        else:
            distortion_list.append(0)
            
        Augmentation_All = "_".join(Augmentation_key_list)
        
        FileName_out = Name + "_" +Augmentation_All + "_" +str(n) + ext
        FileName_out_list.append(FileName_out)
        
        end_time = time.perf_counter()
        
        processingtime = end_time - start_time
        processingtime_list.append(processingtime)
        
        imwrite(os.path.join(args.output, FileName_out), img)
        
    data = list(zip(FileName_out_list, shift_list, rotation_list, scale_list, distortion_list, processingtime_list))
    columns = ["filename", "shift", "rotation", "scale", "distortion", "processing_time"]
    df = pd.DataFrame(data=data, columns=columns)
    
    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))
    df.to_csv(os.path.join(args.output, "augmentation_result.csv"), index=False)
    
    print("Execution time: {} [sec]".format(np.sum(processingtime_list)))
    print("per image: {} [sec]".format((np.mean(processingtime_list))))
    
if __name__ == "__main__":  
    main()