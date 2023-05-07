"""
For every subfolder 'sub_folder' in 'source_folder', check if that subfolder exists in 'target_folder'. If it does exist in target_folder, print f"{sub_folder} exists". If it doesn't exist, print f"{sub_folder doesnt exist}  Copying now"} and then copy sub_folder and all of its contents to target_folder.
"""

import os, shutil

def copy_folders(source_folder,target_folder):
    for sub_folder in os.listdir(source_folder):
        if os.path.exists(f'{target_folder}\\{sub_folder}'):
            print(f'{sub_folder} exists')
        else:
            print(f'{sub_folder} doesnt exist. Copying now')
            shutil.copytree(f'{source_folder}\\{sub_folder}',f'{target_folder}\\{sub_folder}')
            assert(0)