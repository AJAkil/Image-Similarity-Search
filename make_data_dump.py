import os
import shutil
src = './data/animals-10/raw-img'
dest = './data/animals-10/animal-all-data'
main_src_files = os.listdir(src)
for folder_name in main_src_files:
    src_files = os.listdir(os.path.join(src,folder_name))
    # full_file_name = os.path.join(src, file_name)
    src2 = src + '/' + folder_name
    for file_name in src_files:
        full_file_name = os.path.join(src2,file_name)
        # print(full_file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, dest)