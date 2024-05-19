# BaumTestVis

バウムテスト画像の視覚的分析を目的とする研究

## システム概要
このシステムでは、バウムテスト画像を樹木特徴で分類し、それに基づいて統計的処理によって心理的特徴の比較を行います。ImageViewerを実行すると、以下のようなシステムが起動します。

![システム概観](https://github.com/mikimiki3130/BaumTestVis/tree/main/Images_for_readme/SystemView.png "システム概観")

各パーツの構成は以下の通りです。
![システム概観](https://github.com/mikimiki3130/BaumTestVis/tree/main/Images_for_readme/SystemAbstracts.png "システム概観")

システムは大きく分けて以下の機能で構成されます。
- バウムテスト画像群の概観可視化
-- 概観プロット（左上）
-- 代表画像プロット（右上）
- 心理的特徴の可視化
-- クラスタ形成要因バーチャート（左下）
-- クラスタペア比較コードダイアグラム（右下）

使用手順は、  
概観プロット（左上）で分析対象のクラスタを複数囲う  
↓  
クラスタ形成要因バーチャート（左下）で注目する項目をクリック  
↓  
クラスタペア比較コードダイアグラム（右下）で注目するクラスタペア2つを決定  
↓  
概観プロット（左上）で注目クラスタペアをクリック  
↓  
代表画像プロット（右上）で各クラスタのバウムテスト画像群を比較  
という流れになります。

### 各パーツの特徴について
各パーツの特徴を簡単に説明します。

#### 概観プロット
バウムテスト画像群を概観プロットする際、画像データをそのまま次元削減するのではなく、事前に樹木特徴の数値化を行います。画像データを直接次元削減にかける場合、データサイズの問題でデータ拡張（Data Augmentation）を用いたいのですが、ここでバウムテストで重要視される特徴が大きく欠落する可能性があるためです。

![システム概観](https://github.com/mikimiki3130/BaumTestVis/tree/main/Images_for_readme/DR.png "次元削減前の数値化")

#### 代表画像プロット
バウムテスト画像群を実際に比較する際、各クラスタの画像群を並べ替えて表示します。各クラスタの重心から近い順に並べ替えることで、画像群の比較の高速化を目指します。

![代表画像](https://github.com/mikimiki3130/BaumTestVis/tree/main/Images_for_readme/RepresentativeImages.png "代表画像")

#### クラスタ形成要因バーチャート
分析対象のクラスタ群について、心理的特徴の差異を発見するため、数値データの有意差検定を行います。Kruskal-Wallis検定と呼ばれる有意差検定により、有意差のある項目 = 心理的特徴を発見します。

![KW検定](https://github.com/mikimiki3130/BaumTestVis/tree/main/Images_for_readme/KWChart.png "KW検定")

#### クラスタペア比較コードダイアグラム
クラスタ形成要因バーチャートから決定された心理項目について、比較を行う2クラスタを決定するためのコードダイアグラムを作成します。Dunn検定によるペアごとの有意差検定による色づけとヒストグラムによる分布比較により、心理的特徴の差異を視覚化します。

![Dunn検定コードダイアグラム](https://github.com/mikimiki3130/BaumTestVis/tree/main/Images_for_readme/DunnChord.png "Dunn検定コードダイアグラム")

## データファイルについて
Dataフォルダに格納されたデータは、外部の方に提供されたものではなく、実行テスト用の仮データです。

- psydata.csv
  - バウムテスト以外の心理検査の結果による数値データ。ここでは乱数を格納したものを用意
- SizeVisuallyData.xlsm
  - バウムテスト画像の特徴（一部）を目視で数値データ化したもの。乱数を格納した物を用意
- BaumTestData
  - バウムテスト画像のフォルダ（現在は空）。
