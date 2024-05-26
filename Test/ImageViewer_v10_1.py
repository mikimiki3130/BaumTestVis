# 画像特徴を表す数値を用いた次元削減結果のビューワー

import warnings
warnings.filterwarnings("ignore", ".*The 'nopython' keyword.*")
warnings.filterwarnings("ignore", ".*The default value*")

import numpy as np

from gower import gower_matrix

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

import SaveDataFile as SDF
import GetFeatures as GF

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

#=====次元削減==========次元削減==========次元削減==========次元削減==========次元削減=====

# 次元削減
def dim_reduction(features: np.ndarray, DRType="UMAP"):
    """
    対応する次元削減手法を用いて次元削減する。UMAPではGower距離を採用
    
    Params
    ------
    features: np.ndarray
        入力データ
    DRType: str
        指定する次元削減手法（PCA, tSNE, UMAPのいずれか）
        
    Returns
    -------
    np.ndarray
        次元削減されたデータ
    """
    if DRType == "PCA":
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(features)
    elif DRType == "tSNE":
        tsne = TSNE(n_components=2, random_state=0)
        reduced_data = tsne.fit_transform(features)
    elif DRType == "UMAP":
        # 元の順序を保存
        original_indices = np.arange(len(features))

        # Gower距離行列の計算
        gower_dist_matrix = gower_matrix(features)
        # UMAPに距離行列を与える
        umap_model = UMAP(metric='precomputed', n_components=2, random_state=0, n_neighbors=5)
        reduced_data = umap_model.fit_transform(gower_dist_matrix)

        # 次元削減後のデータを元の順序に並べ替える
        reduced_data = reduced_data[np.argsort(original_indices)]
    else:
        print("ERROR: please define DRType(PCA, tSNE, UMAP)")
        return None

    return reduced_data

# データからクラス分けしてラベリングを行う（量的データを想定）
def _data_labeling(data_params, n_clusters=4):
    percentiles = np.linspace(np.min(data_params), np.max(data_params), n_clusters+1)
    labels = np.zeros_like(data_params, dtype=int)

    for i in range(n_clusters):
        condition = (data_params > percentiles[i]) & (data_params <= percentiles[i+1])
        labels[condition] = i

    return labels

# それぞれの数値データについてラベル付け
def create_labels(values, n_clusters=4):
    label_list = []
    for column_index in range(values.shape[1]):
        value = values[:, column_index]
        label = _data_labeling(value, n_clusters)
        label_list.append(label)
    label_list_array = np.array(label_list).T
    return label_list_array

# データロード
image_data_origin, file_names, values_origin, _, _, _ = SDF.load_data()

# 値の規格化
image_data_inverted = 255 - image_data_origin
image_data, values = SDF.data_process(image_data_inverted, values_origin)
image_features_origin = GF.LoadFeatures()
_, image_features = SDF.data_process(image_data_inverted, image_features_origin)

# データの切り出し（1人につき1枚目のみ使用）
image_data = image_data[::3]
image_data_origin = image_data_origin[::3]
image_features = image_features[::3]
values = values[::3]
file_names = file_names[::3]
values_origin_slice = values_origin[::3]

# 各数値データでのラベル付け
n_clusters = 6
labels_list = create_labels(values, n_clusters=n_clusters)
# DR適用
DRType = "UMAP" #"PCA" "tSNE" "UMAP"

reduced_data = dim_reduction(image_features, DRType)

print(DRType, "DR finished")

#=====可視化==========可視化==========可視化==========可視化==========可視化=====
import matplotlib.pyplot as plt
from pycirclize import Circos

import customtkinter as ctk
from CTkListbox import *
from PIL import Image
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path

from scipy.stats import kruskal
from scikit_posthocs import posthoc_dunn

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("dark-blue")


# 使用するラベルの番号とラベル
labels_index = -1
label_name_list = ["sex", "GHQ", "CP", "NP", "A", "FC", "AC", "L", "P", "AF", "SA", "B", "IP", "AG", "R", "F1", "H", "U", "IF", "S", "H", "F2"] 

# KWバー表示用の、数値データの正規化
def values_normalize(values_origin):
    min_vals = np.min(values_origin, axis=0)
    max_vals = np.max(values_origin, axis=0)
    max_vals[max_vals == min_vals] = 1 # 最小値と最大値が等しい列は全て0にする
    values_normalized = (values_origin - min_vals) / (max_vals - min_vals)
    return values_normalized
values_normalized = values_normalize(values_origin)
values_normalized = values_normalized[::3]

# CTkウィンドウの設定
root = ctk.CTk(fg_color = "white")
root.title("Baumtest Analysis")
root.geometry("1200x675")
root.resizable(width = False, height= False)

#====Figure設定==== 
# 散布図用のFigure
fig_scatter, ax_scatter = plt.subplots(figsize=(5.5, 4))
# 散布図の軸等調整
ax_scatter.set_xlabel('dim 1')
ax_scatter.set_ylabel('dim 2')
ax_scatter.xaxis.set_visible(False)
ax_scatter.yaxis.set_visible(False)

# 画像表示用のFigure
fig_image1, ax_image1 = plt.subplots(figsize=(3, 3))
ax_image1.axis('off')  # 軸を非表示にする
ax_image1.set_title("Baumtest image 1")

# 画像表示用のFigure
fig_image2, ax_image2 = plt.subplots(figsize=(3,3))
ax_image2.axis('off')  # 軸を非表示にする
ax_image2.set_title("Baumtest image 2")

# 上にKW検定の結果グラフを描画するFigure
fig_KW, ax_KW = plt.subplots(figsize=(2.5, 4))
ax_KW.axis("off")
ax_KW.set_title("Significant Variable in Selected Clusters")

# chord_diagramの表示用Figure
fig_dunn_chord, ax_dunn_chord = plt.subplots(figsize=(3, 3), subplot_kw=dict(polar=True))
ax_dunn_chord.axis("off")

#====Canvas, Frame設定====
# 各テスト(KW, dunn)用Frame
tests_frame = ctk.CTkFrame(root, fg_color = "white")
tests_frame.pack(side=ctk.BOTTOM, fill=ctk.X, expand=False)

# 次元削減プロット用Canvas
canvas_scatter = FigureCanvasTkAgg(fig_scatter, master=root)
canvas_scatter.get_tk_widget().pack(side=ctk.LEFT, fill=ctk.BOTH, expand=1)

# バウムテスト画像用Frame
image_frame = ctk.CTkFrame(root, fg_color = "white")
image_frame.pack(side=ctk.LEFT, fill=ctk.BOTH, expand=1)

# ====tests_frame下配置====
# TW検定用Frame
KW_frame = ctk.CTkFrame(tests_frame, fg_color = "white")
KW_frame.pack(side=ctk.LEFT, fill=ctk.BOTH, expand=1)

# Dunnの検定用Frame
dunn_frame = ctk.CTkFrame(tests_frame, fg_color = "white")
dunn_frame.pack(side=ctk.RIGHT, fill=ctk.BOTH, expand=1)

# ====iamge_frame下配置==== 
# フレーム作成
image_frame2 = ctk.CTkFrame(image_frame, fg_color = "white")
image_frame2.pack(side=ctk.RIGHT, fill=ctk.BOTH, expand=1)
image_frame1 = ctk.CTkFrame(image_frame, fg_color = "white")
image_frame1.pack(side=ctk.RIGHT, fill=ctk.BOTH, expand=1)

# 画像表示用canvas, スライダ用frame作成
canvas_image1 = FigureCanvasTkAgg(fig_image1, master=image_frame1)
canvas_image1.get_tk_widget().pack(side=ctk.TOP, fill=ctk.BOTH, expand=1)

slider_frame1 = ctk.CTkFrame(image_frame1, fg_color = "white")
slider_frame1.pack(side=ctk.BOTTOM, padx=5)

canvas_image2 = FigureCanvasTkAgg(fig_image2, master=image_frame2)
canvas_image2.get_tk_widget().pack(side=ctk.TOP, fill=ctk.BOTH, expand=1)

slider_frame2 = ctk.CTkFrame(image_frame2, fg_color = "white")
slider_frame2.pack(side=ctk.BOTTOM, padx=5)

# ====KW_frame下配置==== 
canvas_KW = FigureCanvasTkAgg(fig_KW, master=KW_frame)
canvas_KW.get_tk_widget().pack(side=ctk.TOP, fill=ctk.BOTH, expand=1)

# ====dunn_frame下配置==== 
# タイトルのラベル
title_label = ctk.CTkLabel(dunn_frame, text="Significant Pair in Dunn\'s Test", font=("Helvetica", 14), bg_color = "white")
title_label.pack(side=ctk.TOP)

canvas_dunn_chord = FigureCanvasTkAgg(fig_dunn_chord, master=dunn_frame)
canvas_dunn_chord.get_tk_widget().pack(side=ctk.BOTTOM, fill=ctk.BOTH, expand=1)

# 画面表示用のグローバル変数
if True:
    polygon_selector = None
    scatter_focus = None
    selected_points = []
    selected_cluster_data = None
    PolygonFlag = False # ポリゴンのイベント呼び出しに関連
    selection_count = 0 # クラスタ番号の管理
    polygons_list = [] # ポリゴンを複数描写するためのリスト
    indices_list = [] # クラスタに対応する番号リスト
    cluster_data_list = [] # クラスタに対応するデータリスト
    KW_result_list = [] # 検定結果を格納するリスト
    dunn_result_list = []
    dunn_index_list = []
    dunn_pair_list = []
    count_list = []
    rank_sum_list = []
    cluster_index1 = None
    cluster_index2 = None
    chord = None
    cbar = None # カラーバーが複数作られないための変数
    PolygonGetterFlag = False # 作られたクラスタを選択するモードの場合はTrue

    window_level_1 = None
    window_level_2 = None
    window_width_1 = None
    window_width_2 = None

# 散布図の（再）描画
def update_scatter():
    global reduced_data
    global labels_list
    global labels_index
    global label_name_list
    global DRType
    global fig_scatter
    global ax_scatter
    global polygons_list
    global cluster_index1
    global cluster_index2
    global cbar
    global dunn_pair_list

    ax_scatter.clear()  # ax_scatter上の要素をすべてクリア

    ax_scatter.set_title("Baumtest Images Overview")
    ax_scatter.scatter(reduced_data[:, 0], reduced_data[:, 1], c="#050373")
    
    ax_scatter.xaxis.set_visible(False)
    ax_scatter.yaxis.set_visible(False)

    # 概観プロットの上下左右のマージンの計算
    margin = 0.1  # マージンの割合
    x_min, x_max = reduced_data[:, 0].min(), reduced_data[:, 0].max()
    y_min, y_max = reduced_data[:, 1].min(), reduced_data[:, 1].max()
    x_margin = (x_max - x_min) * margin
    y_margin = (y_max - y_min) * margin

    # 軸の範囲を設定
    ax_scatter.set_xlim(x_min - x_margin, x_max + x_margin)
    ax_scatter.set_ylim(y_min - y_margin, y_max + y_margin)

    # ポリゴン描画
    default_alpha = 0.15 # デフォルトの透明度
    for i, poly in enumerate(polygons_list):
        if dunn_pair_list: # ペアがあるなら...
            this_alpha = _det_polygon_alpha(poly_num= i, default_alpha = default_alpha)
            ax_scatter.add_patch(plt.Polygon(poly.get_xy(), edgecolor='r', alpha=this_alpha))
        else:
            ax_scatter.add_patch(plt.Polygon(poly.get_xy(), edgecolor='r', alpha=default_alpha))
        
        # 多角形に番号を表示
        ax_scatter.text(np.mean(poly.get_xy()[:, 0]), np.mean(poly.get_xy()[:, 1]),
                         "C"+str(i+1),
                         horizontalalignment='center', verticalalignment='center', color='blue', fontsize=14)

    # 図を再描画
    fig_scatter.canvas.draw()

# 各クラスタの表示色設定
def _det_polygon_alpha(poly_num, default_alpha = 0.1):
    global cluster_index1
    global cluster_index2
    global dunn_pair_list

    emp_alpha = default_alpha*3
    if emp_alpha > 1:
        emp_alpha = 1
    blr_alpha = default_alpha*0.3

    if cluster_index1 == None and cluster_index2 == None: # 選択クラスタが無いとき
        return default_alpha
    elif cluster_index1 is not None and cluster_index2 is not None: # 選択クラスタが2つのとき
        if cluster_index1 == poly_num or cluster_index2 == poly_num: # 自分が選択クラスタのとき
            return emp_alpha
        else: # 自分が選択クラスタでないとき
            return blr_alpha
    # 以下、選択クラスタが1つのとき
    else:
        if cluster_index1 == poly_num or cluster_index2 == poly_num: # 自分が選択クラスタのとき
            return emp_alpha
        else:
            for pair in dunn_pair_list:
                if cluster_index1 is not None and \
                    cluster_index1 in pair and \
                        poly_num in pair: # 選択クラスタに対応するペアに入っているとき
                    return default_alpha
                elif cluster_index2 is not None and \
                    cluster_index2 in pair and \
                        poly_num in pair: # 選択クラスタに対応するペアに入っているとき
                    return default_alpha
            return blr_alpha # 選択クラスタに対応するペアにいないとき

# 選択したクラスタ群に対して検定を実施
def cluster_test(threshold = 0.05):
    global cluster_data_list
    global label_name_list
    global dunn_result_list
    global dunn_index_list
    global KW_result_list

    KW_result_list.clear()
    dunn_result_list.clear()
    dunn_index_list.clear()

    # Kruskal-Wallis検定
    for i, variable_name in enumerate(label_name_list):
        variable_data = [data[:, i] for data in cluster_data_list]
        
        kruskal_result = kruskal(*variable_data)
        
        # P-valueが閾値以下であるかどうかを判定
        if kruskal_result.pvalue < threshold:
            # Dunn's Testを実施
            posthoc_results = posthoc_dunn(variable_data, p_adjust='bonferroni')
            dunn_result_list.append(posthoc_results)
            dunn_index_list.append(i)

        KW_result_list.append(kruskal_result)
    
    update_KW(threshold= threshold)

# バーのクリックイベントに関連する処理
def on_bar_click(event):
    global labels_index

    if event.mouseevent.name == 'button_press_event':
        if event.artist.get_picker():
            index = int(round(event.artist.get_x() + event.artist.get_width() / 2))
            labels_index = index

            update_KW()
            save_dunn_pair(index= index)
            save_count()
            save_rank_sum()
            update_dunn_chord()
fig_KW.canvas.mpl_connect('pick_event', on_bar_click)

# 全体の平均値を表示+規定クラスタ群についてのKW検定の結果
def update_KW(threshold=0.05):
    global ax_KW
    global KW_result_list
    global values_normalized
    global labels_index
    global label_name_list

    ax_KW.clear()
    ax_KW.set_title("Significant Variable in Selected Clusters")
    ax_KW.set_ylim(0, 1)  # y軸の範囲を設定

    # バーの幅を設定
    bar_width = 0.8

    values_mean = np.mean(values_normalized, axis=0)

    # バーを描画
    for i, label in enumerate(label_name_list):
        if KW_result_list != []:
            color = "coral" if KW_result_list[i].pvalue < threshold else "lightgray"
            edgecolor = "orangered" if i == labels_index else None
        else:
            color = "lightgray"
            edgecolor = None

        bar = ax_KW.bar(i, values_mean[i], color=color, width=bar_width, linewidth=1.5, edgecolor = edgecolor)

        for b in bar:
            b.set_picker(True)

    ax_KW.set_xticks(np.arange(len(label_name_list)))
    ax_KW.set_xticklabels(label_name_list, fontsize=9)
    canvas_KW.draw()

# それぞれのクラスタのnクラス分類の分布をセーブ
def save_count():
    global count_list
    global labels_index
    global labels_list
    global n_clusters

    count_list.clear()
    for i, path in enumerate(polygons_list):
        selected_points = [j for j, (x, y) in enumerate(reduced_data) if path.contains_point((x, y))]
        selected_cluster_data = labels_list[selected_points]
        counts = np.bincount(selected_cluster_data[:, labels_index])
        counts_full = np.zeros(n_clusters, dtype=int)
        counts_full[:len(counts)] = counts

        count_list.append(counts_full)

# 各クラスタの順位和の値をセーブ
def save_rank_sum():
    # 順位和を計算
    def _calculate_row_rank_sums(data_sets):
        # すべてのデータセットの要素を結合
        combined_data = [item for sublist in data_sets for item in sublist]
        # 昇順にソートし、順位をつける
        sorted_data = sorted(combined_data)
        ranks = {}
        current_rank = 1
        for value in sorted_data:
            if value not in ranks:
                ranks[value] = []
            ranks[value].append(current_rank)
            current_rank += 1
        
        # 各要素の平均同順位を計算
        for value, value_ranks in ranks.items():
            avg_rank = sum(value_ranks) / len(value_ranks)
            ranks[value] = avg_rank
        # 各データセットの行の順位和を計算
        row_rank_sums = []
        for data in data_sets:
            row_rank_sum = sum(ranks[value] for value in data)
            row_rank_sums.append(row_rank_sum/ len(data))
        
        return row_rank_sums

    global polygons_list
    global labels_index
    global values
    global rank_sum_list

    label_value = values[:, labels_index]

    value_sets = []
    for i, path in enumerate(polygons_list):
        selected_points = [j for j, (x, y) in enumerate(reduced_data) if path.contains_point((x, y))]
        selected_values = label_value[selected_points]
        value_sets.append(selected_values)
    rank_sum_list = _calculate_row_rank_sums(value_sets)

# chord_diagramの更新
def update_dunn_chord():
    global fig_dunn_chord
    global ax_dunn_chord
    global dunn_pair_list
    global chord
    global polygons_list
    global n_clusters
    global indices_list

    ax_dunn_chord.clear()
    if polygons_list:
        num_nodes = len(polygons_list)
        r_lim = (70, 100)

        default_color = "gray"

        sectors = {f"C{i+1}": 1 for i in range(num_nodes)}
        chord = Circos(sectors, space=30)

        # 弧描画
        for i, sector in enumerate(chord.sectors):
            track = sector.add_track(r_lim= r_lim)
            color = _det_chord_color(i, default_color)
            track.axis(fc=color)
            track.text(sector.name, color="black", size=12, r=110)
            #track.text(f"{len(indices_list[i])} samples", color="black", size=8, r=110)
        
        if dunn_pair_list: # ペアがあるなら...
            # リンク設定
            link_list = []
            for i, dunn_pair in enumerate(dunn_pair_list):
                color = _det_chord_linkcolor(dunn_pair, default_color)
                
                pair_str1 = f"C{dunn_pair[0] +1}"
                pair_str2 = f"C{dunn_pair[1] +1}"
                link_list.append((color, pair_str1, pair_str2))

            # 色が "orangered", "gray", "whitesmoke" の順に並び替え
            link_list.sort(key=lambda x: ("whitesmoke", "gray", "orangered").index(x[0]))

            # リンク描画
            for i, link in enumerate(link_list):
                color = link[0]
                pair_str1 = link[1]
                pair_str2 = link[2]
                # リンクを作成
                chord.link((pair_str1, 0.35, 0.65), (pair_str2, 0.35, 0.65), 
                            alpha=0.8,
                            color=color,
                            r1 = 65, r2 = 65,
                            allow_twist=False)
        
        if count_list: # カウントが行われている=変数が決められているなら...
            # ヒストグラム描画
            for i, sector in enumerate(chord.sectors):
                x = np.linspace(sector.start, sector.end, num=n_clusters+2, endpoint=True)[1:-1]
                y= count_list[i]
                bar_track = sector.add_track(r_lim= r_lim)
                bar_track.axis(ec=None)
                width = 0.1 * (sector.end - sector.start)
                vmax = max(y)*1.2
                bar_track.bar(x, y, color = "black", alpha = 0.4,
                            width = width,
                            vmax = vmax)

        chord.plotfig(ax=ax_dunn_chord)
    fig_dunn_chord.canvas.draw()

# chord_diagramの弧の色設定
def _det_chord_color(number, default_color= "gray"):
    global cluster_index1
    global cluster_index2
    global rank_sum_list

    color_1 = "orange" # 選択クラスタが一つの時の色
    color_s = "lightsalmon" # 順位和が小さいほうの色
    color_b = "orangered" # 順位和が大きいほうの色
    
    #if rank_sum_list and number>=0:
    #    print(f"number {number}:{rank_sum_list[number]}")

    if cluster_index1 == number: #選択クラスタ1が自分のとき
        if cluster_index2 is None:
            return color_1
        elif rank_sum_list[number] < rank_sum_list[cluster_index2]: # 自分の方が順位和が小さいとき
            return color_s
        else:
            return color_b
    elif cluster_index2 == number: # 選択クラスタ2が自分のとき
        if cluster_index1 is None:
            return color_1
        elif rank_sum_list[number] < rank_sum_list[cluster_index1]: # 自分の方が順位和が小さいとき
            return color_s
        else:
            return color_b
    else: # 選択クラスタでないとき
        return default_color

# dhord_diagramのリンク色設定
def _det_chord_linkcolor(dunn_pair, default_color = "gray"):
    global cluster_index1
    global cluster_index2
    dunn_pair = np.array(dunn_pair)

    # カラー設定のif文
    if cluster_index1 == None and cluster_index2 == None: # 選択クラスタが無いとき（グレー表示）
        return default_color
    elif cluster_index1 is None or cluster_index2 is None: # 選択クラスタが一つのとき
        if np.any(dunn_pair == cluster_index1) or np.any(dunn_pair == cluster_index2): # dunn_pairのどちらかにcluster_indexが一致するとき（グレー表示）
            return default_color
        else: # dunn_pairのどちらかにcluster_indexが一致しないとき（不可視化）
            return "whitesmoke"
    else: # 選択クラスタが二つのとき
        if dunn_pair[0] == cluster_index1 and dunn_pair[1] == cluster_index2 or\
            dunn_pair[1] == cluster_index1 and dunn_pair[0] == cluster_index2: # 選択クラスタ群=dunnペア群のとき（強調表示）
            return "orangered"
        else: # 選択クラスタ群≠dunnペア群のとき（不可視化）
            return "whitesmoke"

# Dunnの検定による有意差のあるペアの表示
def save_dunn_pair(threshold=0.05, index=None):
    global dunn_index_list
    global dunn_result_list
    global cluster_index1
    global cluster_index2
    global dunn_pair_list
    global PolygonGetterFlag

    # リストボックスに表示する結果を選択
    count=0
    dunn_pair_list.clear()

    if index in dunn_index_list:
        j = dunn_index_list.index(index)
        posthoc_result = dunn_result_list[j]
        # セルの背景色を変更
        rows, cols = posthoc_result.shape
        
        for r in range(0, rows):
            for c in range(0, cols):
                value = float(posthoc_result.iloc[r, c])
                if value < threshold and r < c:
                    count+=1
                    dunn_pair_list.append([r,c])

    if count==0:
        print("No Significant Result in Dunn's Test.")
    else:
        PolygonGetterFlag = True
    
# 画像表示1のスライダ管理
def image1_init():
    global indices_list
    global cluster_index1

    indices = indices_list[cluster_index1]

    # スライダーの from_ と to の値を更新
    image_slider1.configure(from_=0, to=(len(indices) -1))

    _show_image1(0)

# 画像表示2のスライダ管理
def image2_init():
    global indices_list
    global cluster_index2

    indices = indices_list[cluster_index2]

    # スライダーの from_ と to の値を更新
    image_slider2.configure(from_=0, to=(len(indices) -1))
    _show_image2(0)

# 画像表示1
def _show_image1(number):
    global indices_list
    global cluster_index1
    global window_level_1, window_width_1

    indices = indices_list[cluster_index1]

    index = indices[number]

    # 関数外の値を読み込んでいるので注意
    img_array = image_data_origin[index]
    focus_point = reduced_data[index]

    global scatter_focus
    if scatter_focus:
        scatter_focus.remove()
        scatter_focus = None
    
    scatter_focus = ax_scatter.scatter(focus_point[0], focus_point[1], c="orange")

    # Numpy行列から画像を作成してTkinterで表示
    img = Image.fromarray(np.uint8(img_array))

    # 画像をMatplotlibのAxesに描画
    ax_image1.clear()
    ax_image1.set_title(f"C{cluster_index1 +1} Baumtest image")
    ax_image1.imshow(img_array, cmap="gray")
    ax_image1.set_xticks([])  # x軸の値を無くす
    ax_image1.set_yticks([])  # y軸の値を無くす
    ax_image1.set_frame_on(True)  # 枠線を表示

    # グラフおよび画像を描画
    canvas_scatter.draw()
    canvas_image1.draw()

    # ウィンドウレベルとウィンドウ幅の自動設定
    window_level_1, window_width_1 = _auto_windowing(img_array)

# 画像表示2
def _show_image2(number):
    global indices_list
    global cluster_index2
    global window_level_2, window_width_2

    indices = indices_list[cluster_index2]

    index = indices[number]

    # 関数外の値を読みこんでいるので注意
    img_array = image_data_origin[index]
    focus_point = reduced_data[index]

    global scatter_focus
    if scatter_focus:
        scatter_focus.remove()
        scatter_focus = None
    
    scatter_focus = ax_scatter.scatter(focus_point[0],focus_point[1], c= "orange")

    # Numpy行列から画像を作成してTkinterで表示
    img = Image.fromarray(np.uint8(img_array))
    #img = img.resize((200, 200))  # 画像サイズを調整

    # 画像をMatplotlibのAxesに描画
    ax_image2.clear()
    ax_image2.set_title(f"C{cluster_index2 +1} Baumtest image")
    ax_image2.imshow(img_array, cmap = "gray")
    ax_image2.set_xticks([]) # x軸の値を無くす
    ax_image2.set_yticks([]) # y軸の値を無くす
    ax_image2.set_frame_on(True)  # 枠線を表示

    # グラフおよび画像を描画
    canvas_scatter.draw()
    canvas_image2.draw()

    # ウィンドウレベルとウィンドウ幅の自動設定
    window_level_2, window_width_2 = _auto_windowing(img_array)

# WW・WL変更後の画像更新
def update_image1():
    global indices_list
    global cluster_index1
    global window_level_1, window_width_1
    number = var_slider1.get()

    indices = indices_list[cluster_index1]

    index = indices[number]

    # 関数外の値を読み込んでいるので注意
    img_array = image_data_origin[index]

    # ウィンドウレベルとウィンドウ幅から、最大値と最小値を計算
    window_min = window_level_1 - window_width_1
    window_max = window_level_1 + window_width_1

    # window_min以下の画素値を0に、window_max以上の画素値を255に変換
    img_array_windowed = np.where(img_array <= window_min, 0, img_array)
    img_array_windowed = np.where(img_array >= window_max, 255, img_array_windowed)
    
    # window_min以上、window_max以下の画素値を0～255の範囲で線形変換
    img_array_windowed = np.where((0 < img_array_windowed) & (img_array_windowed < 255) ,
                                  ((img_array_windowed - window_min) / (window_max - window_min)) * 255,
                                  img_array_windowed)

    # 範囲外の値をクリップ（念のため） 
    img_array_windowed = np.clip(img_array_windowed, 0, 255)

    # print(f"M:{window_max} m:{window_min}")

    # Numpy行列から画像を作成してTkinterで表示
    img = Image.fromarray(np.uint8(img_array_windowed))

    # 画像をMatplotlibのAxesに描画
    ax_image1.clear()
    ax_image1.set_title(f"Cluster {cluster_index1} Baumtest image")
    ax_image1.imshow(img_array_windowed, cmap="gray")
    ax_image1.set_xticks([])  # x軸の値を無くす
    ax_image1.set_yticks([])  # y軸の値を無くす
    ax_image1.set_frame_on(True)  # 枠線を表示

    # グラフおよび画像を描画
    canvas_image1.draw()

# WW・WL変更後の画像更新
def update_image2():
    global indices_list
    global cluster_index2
    global window_level_2, window_width_2

    number = var_slider2.get()

    indices = indices_list[cluster_index2]

    index = indices[number]

    # 関数外の値を読み込んでいるので注意
    img_array = image_data_origin[index]

    # ウィンドウレベルとウィンドウ幅から、最大値と最小値を計算
    window_min = window_level_2 - window_width_2 
    window_max = window_level_2 + window_width_2

    # window_min以下の画素値を0に、window_max以上の画素値を255に変換
    img_array_windowed = np.where(img_array <= window_min, 0, img_array)
    img_array_windowed = np.where(img_array >= window_max, 255, img_array_windowed)
    
    # window_min以上、window_max以下の画素値を0～255の範囲で線形変換
    img_array_windowed = np.where((0 < img_array_windowed) & (img_array_windowed < 255) ,
                                  ((img_array_windowed - window_min) / (window_max - window_min)) * 255,
                                  img_array_windowed)

    # 範囲外の値をクリップ（念のため） 
    img_array_windowed = np.clip(img_array_windowed, 0, 255)

    # Numpy行列から画像を作成してTkinterで表示
    img = Image.fromarray(np.uint8(img_array_windowed))

    # 画像をMatplotlibのAxesに描画
    ax_image2.clear()
    ax_image2.set_title(f"Cluster {cluster_index1} Baumtest image")
    ax_image2.imshow(img_array_windowed, cmap="gray")
    ax_image2.set_xticks([])  # x軸の値を無くす
    ax_image2.set_yticks([])  # y軸の値を無くす
    ax_image2.set_frame_on(True)  # 枠線を表示

    # グラフおよび画像を描画
    canvas_image2.draw()

# 画像表示1での、マウスドラッグによるWW・WL更新
def on_mouse_move_image1(event):
    global window_level_1, window_width_1

    if event.button == 1:  # 左ボタンが押されている場合
        # マウスの移動量を取得
        dx = event.xdata - on_mouse_move_image1.prev_x if on_mouse_move_image1.prev_x is not None else 0
        dy = event.ydata - on_mouse_move_image1.prev_y if on_mouse_move_image1.prev_y is not None else 0

        # マウスの移動に応じてWL/WWを更新
        window_level_1 -= dx
        window_width_1 -= dy

        # 上限と下限を設定
        window_level_1 = np.clip(window_level_1, 1, 254)
        if window_level_1 < 255 - window_level_1:
            window_width_1 = np.clip(window_width_1, 255 - window_level_1, window_level_1)
        else:
            window_width_1 = np.clip(window_width_1, window_level_1, 255 - window_level_1)
        #print("WL1=", window_level_1, "WW1=", window_width_1)

        # 画像再表示
        update_image1()

    # マウス座標の更新
    on_mouse_move_image1.prev_x = event.xdata
    on_mouse_move_image1.prev_y = event.ydata
on_mouse_move_image1.prev_x = None
on_mouse_move_image1.prev_y = None
canvas_image1.mpl_connect('motion_notify_event', on_mouse_move_image1)

# 画像表示2での、マウスドラッグによるWW・WL更新
def on_mouse_move_image2(event):
    global window_level_2, window_width_2

    if event.button == 1:  # 左ボタンが押されている場合
        # マウスの移動量を取得
        dx = event.xdata - on_mouse_move_image2.prev_x if on_mouse_move_image2.prev_x is not None else 0
        dy = event.ydata - on_mouse_move_image2.prev_y if on_mouse_move_image2.prev_y is not None else 0

        # マウスの移動に応じてWL/WWを更新
        window_level_2 -= dx
        window_width_2 -= dy

        # 更新後不正な値を取らないよう調整
        window_level_2 = np.clip(window_level_2, 1, 254)
        if window_level_2 < 255 - window_level_2:
            window_width_2 = np.clip(window_width_2, 255 - window_level_2, window_level_2)
        else:
            window_width_2 = np.clip(window_width_2, window_level_2, 255 - window_level_2)

        # 画像再表示
        update_image2()

    # マウス座標の更新
    on_mouse_move_image2.prev_x = event.xdata
    on_mouse_move_image2.prev_y = event.ydata
on_mouse_move_image2.prev_x = None
on_mouse_move_image2.prev_y = None
canvas_image2.mpl_connect('motion_notify_event', on_mouse_move_image2)

# 画像から自動的にウィンドウレベル、ウィンドウ幅を計算する関数
def _auto_windowing(image_array):
    # 画像のヒストグラムを計算
    hist, bins = np.histogram(image_array, bins=256, range=(0, 255))

    # ヒストグラムのピーク（最頻値）を見つける
    peak_index = np.argmax(hist)
    window_level = bins[peak_index]

    # 画像の最小値と最大値を使用してウィンドウ幅を計算
    min_value = np.min(image_array)
    max_value = np.max(image_array)
    window_width = max_value - min_value

    return window_level, window_width

# 選択されたデータを保存する関数
def save_selected_data(selected_indices, cluster_data):
    global selection_count
    global indices_list
    global cluster_data_list
    
    indices_list.append(selected_indices)
    cluster_data_list.append(cluster_data)

    selection_count += 1  # カウントを増やす

# クラスタ選択モードのとき、選択されたクラスタのindexを返す
def select_polygon(event):
    global polygons_list
    global cluster_index1
    global cluster_index2

    # クリックした座標を取得
    x_click = event.xdata
    y_click = event.ydata
    if x_click is None or y_click is None:
        print("OUT OF RANGE")
        return  # クリックした座標がデータ座標系でない場合は処理を中止

    # 選択されたクラスタの番号を取得
    for i, poly in enumerate(polygons_list):
        path = poly.get_path()
        if path.contains_point((x_click, y_click)):
            selected_cluster_index = i

            # 選択クラスタの管理
            if cluster_index1 == None:
                if cluster_index2 == None:
                    cluster_index1 = selected_cluster_index
                elif cluster_index2 == selected_cluster_index:
                    cluster_index2 = None
                else:
                    cluster_index1 = selected_cluster_index
            elif cluster_index1 == selected_cluster_index:
                cluster_index1 = None
            else:
                if cluster_index2 == None:
                    cluster_index2 = selected_cluster_index
                elif cluster_index2 == selected_cluster_index:
                    cluster_index2 = None
                else:
                    cluster_index2 = selected_cluster_index
            
            if cluster_index1 is not None:
                image1_init()
            if cluster_index2 is not None:
                image2_init()

            break

# 網掛け範囲選択のコールバック関数
def _on_polygon_select(vertices):
    global PolygonFlag
    global selected_points
    global selected_cluster_data
    global polygons_list

    # ポリゴン完成後に点選択したとき、範囲選択による更新が起こらないようフラグ管理
    if not PolygonFlag:
        PolygonFlag = True
        # 選択範囲内の点のindex群を取得
        path = Path(vertices)
        selected_points = [i for i, (x, y) in enumerate(reduced_data) if path.contains_point((x, y))]
        if not selected_points:
            print("NO POINTS")
            return  # 範囲内にデータ点がない場合は処理を中止
        
        # データ点の重心を計算
        centroid = np.mean(reduced_data[selected_points], axis=0)

        # 各データ点と重心の距離を計算し、リストに追加
        distances = [(i, np.linalg.norm(reduced_data[i] - centroid)) for i in selected_points]

        # 距離に基づいてリストをソート
        sorted_distances = sorted(distances, key=lambda x: x[1])

        # ソートされたリストからインデックスを取得し、selected_points を更新
        selected_points = [idx for idx, _ in sorted_distances]

        # 選択されたクラスタ内のデータを取得
        selected_cluster_data = values[selected_points]

        # データの保存
        poly = plt.Polygon(vertices, edgecolor="r", alpha=0.2)
        polygons_list.append(poly)
        save_selected_data(selected_points, selected_cluster_data)

        update_scatter()  # 散布図を更新
        update_dunn_chord()

        if len(polygons_list) >= 2:
            cluster_test()

# 散布図でクリックが押されたとき
def on_scatter_click(event):
    global PolygonFlag
    global polygon_selector
    global selected_points
    global PolygonGetterFlag

    if PolygonGetterFlag:
        select_polygon(event)
        update_dunn_chord()
        update_scatter()
    else:
        # 範囲が閉じられている場合、古いものを消去
        if PolygonFlag:
            polygon_selector.set_visible(False)
            del polygon_selector
            polygon_selector = None
            PolygonFlag = False

        if not polygon_selector:
            # PolygonSelectorを作成
            polygon_selector = LassoSelector(ax_scatter, onselect=_on_polygon_select, props={'color': 'blue'})
canvas_scatter.mpl_connect('button_press_event', on_scatter_click)

#===スライダー・ボタン======スライダー・ボタン======スライダー・ボタン======スライダー・ボタン===

# 選択クラスタ群のリセット(Rキーで呼び出し)
def _reset_clusters():
        global polygons_list
        global indices_list
        global cluster_data_list
        global selection_count
        global KW_result_list
        global dunn_index_list
        global dunn_result_list
        global cluster_index1
        global cluster_index2
        global selected_label_index
        global labels_index
        global dunn_pair_list
        global PolygonGetterFlag

        PolygonGetterFlag = False

        labels_index = -1

        selected_label_index = 1
        cluster_index1 = None
        cluster_index2 = None

        KW_result_list.clear()
        dunn_result_list.clear()
        dunn_index_list.clear()
        dunn_pair_list.clear()

        selection_count = 0
        polygons_list.clear()
        indices_list.clear()
        cluster_data_list.clear()

        update_scatter()
        update_KW()
        update_dunn_chord()

# キーボードクリックの反応
def on_key_press(event):
    if event.char == "r" or event.char == "R":
        _reset_clusters()
    elif event.char == "x" or event.char == "X":
        global PolygonGetterFlag
        global cluster_index1
        global cluster_index2
        
        PolygonGetterFlag = False
        cluster_index1 = None
        cluster_index2 = None
root.bind("<KeyPress>", on_key_press)

var_slider1 = ctk.IntVar(master=slider_frame1)
label_var1 = ctk.CTkLabel(master=slider_frame1, width=10, textvariable=var_slider1, bg_color = "white")
image_slider1 = ctk.CTkSlider(master=slider_frame1, width=100, variable=var_slider1, bg_color = "white",
                              command= lambda value: _show_image1(var_slider1.get()))
label_var1.grid(row=0, column=0, padx=0, pady=0)
image_slider1.grid(row=0, column=1, padx=0, pady=0)

var_slider2 = ctk.IntVar(master=slider_frame2)
label_var2 = ctk.CTkLabel(master=slider_frame2, width=10, textvariable=var_slider2, bg_color = "white")
image_slider2 = ctk.CTkSlider(slider_frame2, width=100, variable=var_slider2, bg_color = "white",
                              command= lambda value: _show_image2(var_slider2.get()))
label_var2.grid(row=0, column=0, padx=0, pady=0)
image_slider2.grid(row=0, column=1, padx=0, pady=0)

update_scatter()
update_KW()

#===メインループ======メインループ======メインループ======メインループ===
root.mainloop()