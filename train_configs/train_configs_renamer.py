import os
import glob

def rename_yaml_configs(path: str = "."):
    # 找到當前目錄下所有的yaml檔案
    yaml_files = glob.glob(os.path.join(path, '*.yaml'))
    
    # 依照檔案名稱排序
    yaml_files.sort()
    
    # 重新命名檔案
    for i, file in enumerate(yaml_files, 1):
        new_name = os.path.join(path, f'train_config_{i}.yaml')
        try:
            os.rename(file, new_name)
            print(f'已將 {file} 重新命名為 {new_name}')
        except OSError as e:
            print(f'重新命名 {file} 時發生錯誤: {e}')

if __name__ == '__main__':
    rename_yaml_configs("./train_configs")
