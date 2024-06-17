# 画像データ等のデータファイル作成および読み込み用ライブラリ
import os
import numpy as np
from PIL import Image
import pandas as pd
import cv2

# 指定フォルダから画像とファイル名を読み込む
def _input_image_from_files(folder_path: str, resize_ratio = 0.1):
    
    file_names = []
    image_data = []

    first_image_flag = True
    for file_name in sorted(os.listdir(folder_path)):
        if file_name.endswith('.jpg') or file_name.endswith('.JPG') or file_name.endswith('.jpeg'):

            file_path = os.path.join(folder_path, file_name)
            file_names.append(file_name)

            img = Image.open(file_path).convert('L')  # グレースケールに変換

            # 最初に読みこんだ画像のアスペクト比を使う
            if first_image_flag:
                resize_width = int(img.size[0] * resize_ratio)
                resize_height = int(img.size[1] * resize_ratio)
                first_image_flag = False
            
            img_resized = img.resize((resize_width, resize_height))  # 画像サイズを統一する（拡大縮小が行われる可能性あり）
            img_array = np.array(img_resized)
            
            image_data.append(img_array)

    image_data = np.array(image_data)  # [n,x,y]の3次元データ（n枚、x*yサイズ画像の場合）
    
    return file_names, image_data

#数値データを読み込む
def _input_value_from_files(filename):
    """
    各数値データ
        性別 GHQ CP	NP	A	FC	AC	L	P	AF	SA	B	IP	AG	R	F	H	U	IF	S	H	F
    """
    df = pd.read_csv(filename, header=None, skipinitialspace=True)

    # データの部分を取得
    data = df.iloc[3:, 2:].values  # 4行目かつ3列目以降のデータを取得

    # データをnumpy配列に変換
    data = data.astype(float)

    #画像データが無い番号を消去(提供データ)
    indices_to_remove = [74, 128]  # 75番目、129番目を消去
    data_np = np.delete(data, indices_to_remove, axis=0)

    # 各行を3つずつに拡張（1人につきそれぞれ3枚の画像があるため）
    expanded_data = np.repeat(data_np, 3, axis=0)

    #print(f"{data.ndim} {data_np.ndim} {expanded_data.ndim}")
    return expanded_data

# 目視画像データから数値を読み取る
def _get_labels(filename: str):
    """
    それぞれ、大きさ、位置を表すラベル（主観のラベリング）
    
    Returns
    ------
    siz : int 
        0→はみ出すほど大きい, 1→大きい, 2→小さい
    pt : int
        0→中心付近, 1→第1象限, 2→第2象限, 3→第3象限, 4→第4象限
    dir : int
        0→縦描き, 1→横描き
    withered : int 
        0→枯れ木でない, 1→枯れ木である
    """
    # Excelファイルを読み込む
    df = pd.read_excel(filename, header=None)
    # n行目以降、m列目を指定するとき→df.iloc [n-1:, m-1]
    size = df.iloc[1:, 4]  # 大きさ
    size_np = np.array(size.astype(int))
    point = df.iloc[1:, 5]  # 位置
    point_np = np.array(point.astype(int))
    direction = df.iloc[1:, 6]  # 向き
    direction_np = np.array(direction.astype(int))
    withered = df.iloc[1:, 7]  # 枯れ木か否か
    withered_np = np.array(withered.astype(int))

    # size_np, point_np, direction_np を列方向に結合
    #labels = np.column_stack((size_np, point_np, direction_np))
    return size_np, point_np, direction_np, withered_np

# データをセーブする
def save_data(  this_filename,
                image_data,
                file_names,
                values,
                label_siz, 
                label_pt, 
                label_dir,
                label_withered):
    """
    各データをdata.npzファイルにセーブする。
    """

    #それぞれのnumpy行列をファイルに保存
    np.savez(this_filename,
                image_data = image_data,
                file_names = file_names,
                values = values,
                label_siz = label_siz, 
                label_pt = label_pt, 
                label_dir = label_dir,
                label_withered = label_withered)

# データをロードする
def load_data():
    """
    各データをdata.npzファイルからロードする。
    
    Returns
    ------
    image_data:
        バウムテスト画像群(3次元)
    file_names: 
        バウムテストの画像名
    values:
        数値データ群
        性別 GHQ CP	NP	A	FC	AC	L	P	AF	SA	B	IP	AG	R	F	H	U	IF	S	H	F
    label_siz:
        画像の樹木の大きさラベル(0,1,2)
    label_pt:
        画像の樹木の位置ラベル(0,1,2,3,4)
    label_dir:
        画像の樹木の向きラベル(0,1)
    """

    filename = "data.npz"

    matrixs = np.load(filename)

    image_data = matrixs["image_data"]
    file_names = matrixs["file_names"]
    values = matrixs["values"]
    label_siz = matrixs["label_siz"]
    label_pt = matrixs["label_pt"]
    label_dir = matrixs["label_dir"]

    return image_data, file_names, values, label_siz, label_pt, label_dir

# 特定のデータのみをロードする
def load_certain_data(data_name: str):
    """
    特定のデータのみをロードする

    Parameters
    ------
    data_name:
        データの名前（Returns参照）

    Returns
    ------
    image_data:
        バウムテスト画像群(3次元)
    file_names: 
        バウムテストの画像名
    values:
        数値データ群
        性別 GHQ CP	NP	A	FC	AC	L	P	AF	SA	B	IP	AG	R	F	H	U	IF	S	H	F
    label_siz:
        画像の樹木の大きさラベル(0,1,2)
    label_pt:
        画像の樹木の位置ラベル(0,1,2,3,4)
    label_dir:
        画像の樹木の向きラベル(0,1)
    label_withered:
        画像の樹木が枯れているかのラベル(0,1)
    """
    filename = "data.npz"

    matrixs = np.load(filename)

    return matrixs[data_name]

# データの標準化を行う
def data_process(image_data, values, BIN=False):
    # 2値化
    if BIN:
        _, image_data = cv2.threshold(src=image_data, thresh=250, maxval=255, type=cv2.THRESH_BINARY)

    # データの前処理
    # 画像データを(n,(x*y))の2次元データにする
    image_data = image_data.reshape(image_data.shape[0], -1)

    # 画像データの標準化
    mean_vals = np.mean(image_data, axis=1, keepdims=True)
    std_vals = np.std(image_data, axis=1, keepdims=True)
    std_vals[std_vals == 0] = 1  # 標準偏差が0の場合は1に置き換える
    image_data = (image_data - mean_vals) / std_vals

    # 数値データの標準化
    mean_vals = np.mean(values, axis=0)
    std_vals = np.std(values, axis=0)
    std_vals[std_vals == 0] = 1  # 標準偏差が0の場合は1に置き換える
    values = (values - mean_vals) / std_vals

    return image_data, values


if __name__ == "__main__":
    filename = "data.npz"

    file_names, image_data = _input_image_from_files(folder_path = '../baumtest_data/baumtest_data_word-deleted_2')

    values = _input_value_from_files(filename = "../psydata_3.csv")
    
    label_siz, label_pt, label_dir, label_withered = _get_labels(filename='../baumtest_data/サイズ等目視データ.xlsm')
    
    
    save_data(  this_filename= filename,
                image_data = image_data,
                file_names = file_names,
                values = values,
                label_siz = label_siz, 
                label_pt = label_pt, 
                label_dir = label_dir,
                label_withered = label_withered) 
    
