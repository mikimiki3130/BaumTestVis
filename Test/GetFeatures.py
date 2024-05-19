# 画像の特徴を得る関数群

import cv2
import numpy as np
import SaveDataFile as SDF

# 画像群から、それぞれの輪郭群を得る
def _getContoursList(image_data):
    # 輪郭の配列を作成
    contours_list = []

    # 各画像の輪郭を求める（往々にして輪郭が繋がっていないので、前処理で繋げる）
    for img_array in image_data:
        # 2値化1回目
        img_inv1 = cv2.threshold(img_array, 250, 255, cv2.THRESH_BINARY_INV)[1]

        # ブラーをかける
        img_blur1 = cv2.GaussianBlur(img_inv1, 
                                     (35, 35), # カーネル（大きいほど大雑把になる、奇数にすること）
                                     0)
        # 2値化2回目
        img_inv2 = cv2.threshold(img_blur1, 1, 255, cv2.THRESH_BINARY)[1]

        # 輪郭検出
        contours, _ = cv2.findContours(img_inv2, 
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        contours_list.append(contours)

    return contours_list

# 輪郭群から、最大の面積を持つ輪郭について、その輪郭たちを得る
def _getMaxContour(contours_list):
    max_contour_list = []
    for contours in contours_list:
        max_contour = max(contours, key=cv2.contourArea)
        max_contour_list.append(max_contour)
    return max_contour_list

# 輪郭リストから、それぞれの面積の大きさを求める
def _getContourArea(max_contour_list):
    areas_list = []
    for contour in max_contour_list:
        area = cv2.contourArea(contour)
        areas_list.append(area)
    return areas_list

# それぞれの輪郭の中心点の座標を返す
def _getContourCenters(max_contour_list):
    centers = []
    for contour in max_contour_list:
        # 輪郭のモーメントを計算
        M = cv2.moments(contour)
        
        # 中心点の座標を計算
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        centers.append((cx, cy))
    return centers

# 各画像の画素値の平均値を得る
def _getMean(image_data):
    image_data = 255 - image_data
    means_list = []
    for img_array in image_data:
        # 背景（画素値が0のピクセル）を除外して平均を計算
        non_zero_pixels = img_array[img_array != 0]
        mean = np.mean(non_zero_pixels) if len(non_zero_pixels) > 0 else 0
        means_list.append(mean)
    return means_list
    
# 輪郭の縦横比を得る
def _getAspectRatio(max_contour_list):
    aspect_ratio_list = []
    for contour in max_contour_list:
        _, _, w, h = cv2.boundingRect(contour)

        # 縦横比を計算
        aspect_ratio = w / h
        aspect_ratio_list.append(aspect_ratio)
    return aspect_ratio_list

# 画像の特徴を数値として返す
def getImageFeatures(image_data):
    # 最大の輪郭を取り出す
    contours_list = _getContoursList(image_data)
    max_contour_list = _getMaxContour(contours_list)
    
    # 最大の輪郭が囲む領域の面積（≒樹の大きさ）
    areas_list = _getContourArea(max_contour_list)
    areas_list_array = (np.array(areas_list)).reshape(-1,1)

    # 画素値の平均値（≒線の濃さ）
    means_list = _getMean(image_data)
    means_list_array = (np.array(means_list)).reshape(-1,1)

    # 最大の領域の中心座標（≒樹の位置）
    center_point_list = _getContourCenters(max_contour_list)
    center_point_list_array = np.array(center_point_list)

    # 輪郭の縦横比（≒樹木の向き）
    aspect_ratio_list = _getAspectRatio(max_contour_list)
    aspect_ratio_list_array = (np.array(aspect_ratio_list)).reshape(-1,1)
    
    withered_list = SDF.load_certain_data(data_name= "label_withered")
    withered_list_array = withered_list.reshape(-1,1)

    features_list = np.hstack((areas_list_array, means_list_array, center_point_list_array, aspect_ratio_list_array, withered_list_array))
    return features_list

def SaveFeatures(image_data):
    features_list = getImageFeatures(image_data)
    filename= "data_features.npz"
    #データをファイルに保存
    np.savez(filename,
                features_list = features_list)

def LoadFeatures():
    filename = "data_features.npz"

    matrixs = np.load(filename)
    features_list = matrixs["features_list"]
    return features_list

if __name__ == "__main__":
    image_data, _, _, _, _, _ = SDF.load_data()
    features_list = getImageFeatures(image_data)
    print("features_list.shape:", features_list.shape)
    SaveFeatures(image_data)