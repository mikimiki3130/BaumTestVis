# ファイル名の数字を改訂するためのコード
# 例： "1-1 20.jpg" → "001-1 20.jpg"

import os

def add_zeros_to_filename(folder_path):
    # フォルダ内のファイル名を取得
    files = os.listdir(folder_path)
    
    for file_name in files:
        # ファイル名と拡張子を分割
        base_name, ext = os.path.splitext(file_name)
        
        # ファイル名を空白で分割し、先頭の数を取得
        parts = base_name.split()

        # 先頭の数が数字とハイフンで構成されている場合にのみ処理を行う
        if len(parts) > 0 and all((c.isdigit() or (c == '-' and i in (1, 2))) for i, c in enumerate(parts[0])):
            number = parts[0]
            
            # ハイフンを基準にして、左側と右側の部分を取得
            left, right = number.split('-')
            
            # 左側の数を3桁に変更
            new_left = left.zfill(3)
            
            # 新しいファイル名を生成
            new_number = new_left + '-' + right
            new_base_name = new_number + ' ' + ' '.join(parts[1:])
            new_file_name = new_base_name + ext
            
            # ファイル名を変更
            old_path = os.path.join(folder_path, file_name)
            new_path = os.path.join(folder_path, new_file_name)
            os.rename(old_path, new_path)

if __name__ == "__main__":
    folder_path = './Data/baumtest_data_word-deleted_2'
    add_zeros_to_filename(folder_path)