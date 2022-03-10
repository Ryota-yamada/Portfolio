import sys
import cv2
import yaml
import pathlib
import os
from PIL import Image
import numpy as np
from natsort import natsorted

# コマンドライン引数で設定YAMLファイルパスを取得する
def GetPath():
    args = sys.argv
    if len(args) <= 1:
        print("Error: yamlfile is not specified")
        sys.exit()
    else :
        yamlPath = str(args[1])
        return yamlPath

# yamlファイルを読み込みyamlParam を取得
def Read(yamlPath):
    with open(yamlPath, "r",encoding="UTF-8") as yml:
        yamlParam = yaml.safe_load(yml)
        return yamlParam

# [[condition1の画像名リスト],[condition2の画像名リスト],,, ] (imageNameList)
def FileSort(yamlParam):
    annotationPathList = [pathlib.Path(path) for path in yamlParam["annotation_path"]]
    extension = yamlParam["extension"]
    imageNameList = []
    for path in annotationPathList:
        # extensionがBMPでも.bmpファイルのものはこの時点ではすべて検索され、リストに含まれる。
        imageName = [str(name) for name in path.glob("*." + extension)]
        # 以下のコードで文字列に対して、検索をかけることで大文字と小文字を別々の拡張子として扱うことができる。また、並べ替えはbasenameによってファイル名のみを取り出すことで, 安全に並べ替えできるようにした。
        imageName = natsorted([name for name in imageName if "." + extension in name],key = lambda x: os.path.basename(x))
        imageNameList.append(imageName)
    return imageNameList

# 合成画像を出力フォルダに保存
def Composite(yamalParam,imageNameList):
    replacement_before = yamalParam["replacement"]["before"]
    replacement_after = yamalParam["replacement"]["after"]
    outputPath = yamalParam["dest_path"]
    imageNum = len(imageNameList[0])
    for num in range(0,imageNum):
        #同じIDの画像名（照明条件は異なる）をリスト化
        imageNameSame = [conditionImage[num] for conditionImage in imageNameList ]
        count = 0
        for imageName in imageNameSame:
            if count == 0:
                # 画像の読み込みには日本語パスを読み込めるようPillowを使った
                imageComp = np.array(Image.open(imageName))
            else:
                image = np.array(Image.open(imageName))
                imageComp = cv2.bitwise_or(imageComp,image)
            count += 1
            # 出力ファイル名のもとになるファイル名（仕様指定のようにannotation_path[0]のフォルダから取得したファイル名を元にする）
            firstDirFileName = str(os.path.basename(imageNameSame[0]))
            # 出力ファイル名
            filenameOutput = firstDirFileName.replace(replacement_before, replacement_after)
        cv2.imwrite(os.path.join(outputPath,filenameOutput),imageComp)
        print(f"{filenameOutput} is saved")
def Main():
    yamlPath = GetPath()
    yamlParam = Read(yamlPath)
    imageNameList = FileSort(yamlParam)
    Composite(yamlParam,imageNameList)
    
Main()