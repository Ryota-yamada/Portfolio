import sys
import pathlib
import os
import yaml
import cv2
import numpy as np
import statistics
import pandas as pd
from natsort import natsorted
# コマンドライン引数で設定yamlファイルのパスを取得する
def GetPath():
    args = sys.argv
    if len(args) <= 1:
        print("Error: yamlfile is not specified")
        sys.exit()
    else:
        yamlPath = str(args[1])
    return yamlPath

# yamlファイルを読み込みyamlParamを取得
def Read(yamlPath):
    with open(yamlPath,"r",encoding="UTF-8") as yml:
        yamlParam = yaml.safe_load(yml)
        return yamlParam
    
# 輪郭検出に使う画像の前処理を行うジェネレータ
def Preprocess_Image(yamlParam):
    inputDirpath = pathlib.Path(str(yamlParam["Image_Dirpath"]))
    extension = str(yamlParam["extension"])
    filePathList = natsorted([str(path) for path in inputDirpath.glob("*." + extension)])
    for filePath in filePathList:
        file_name = str(os.path.basename(filePath)) 
        CP_name = file_name.replace("." + extension,"")
        img = cv2.imread(filePath)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if yamlParam["SEM_5"] == True:
            img_trim = img[0:400,0:600]
            img_trim_gray = img_gray[0:400,0:600]
        else:
            img_trim = img[0:347, 0:490]
            img_trim_gray = img_gray[0:347, 0:490]
        th, img_binary = cv2.threshold(img_trim_gray,0, 255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        yield img_trim, img_trim_gray, img_binary, CP_name, extension

def Calculate(yamlParam):
    nm_pixel = yamlParam["nm_pixel"]
    pixel_nm = 1/nm_pixel
    OutDirpath = pathlib.Path(str(yamlParam["Out_Dirpath"]))

    # 出力フォルダ作成(全体用)
    os.makedirs(OutDirpath, exist_ok=True)
    # 二値化画像出力ディレクトリ
    os.makedirs(os.path.join(OutDirpath, "binary_Image/"), exist_ok=True)
    # フィッティング画像出力ディレクトリ
    os.makedirs(os.path.join(OutDirpath, "Fitting_Image/"), exist_ok=True)

    resultList = []
    for img_trim, img_trim_gray, img_binary, CP_name, extension in Preprocess_Image(yamlParam):
        contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        contours = list(contours)
        contours =  sorted(contours,key = cv2.contourArea,reverse = True)
        contours = contours[6:21]
        img_contour = cv2.drawContours(img_trim, contours, -1, (0, 255, 0),2)
        areaList = []
        for contour in contours:
            area = cv2.contourArea(contour)
            areaList.append(area)
        area_ave = statistics.mean(areaList)
        # 円形メタ原子の場合
        if str(yamlParam["shape"]) == "circle":
            diameter = 2*np.sqrt((area_ave/3.141592))*pixel_nm
        # 正方形メタ原子の場合
        elif str(yamlParam["shape"]) == "square":
            diameter = np.sqrt(area_ave)
        result = [CP_name, diameter]
        resultList.append(result)
        # 二値化画像を保存
        cv2.imwrite(os.path.join(OutDirpath, "binary_Image/", f"{CP_name}_binary." + extension), img_binary)  
        # フィッティング後の画像を保存
        cv2.imwrite(os.path.join(OutDirpath, "Fitting_Image/", f"{CP_name}_fitting." + extension),img_contour)
        print(f"{CP_name} was done")
    df = pd.DataFrame(data = resultList,columns=["number", "diameter_measured"])
    # 直径計測結果のcsvファイル保存
    df.to_csv(os.path.join(OutDirpath, "measured_width.csv"))

def Main():
    yamlPath = GetPath()
    yamlParam = Read(yamlPath)
    Calculate(yamlParam)

Main()



