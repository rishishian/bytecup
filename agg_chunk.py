''' aggeregate train/val bin file in multiple directories to one directory'''
import os
import shutil

root_path = '/home/lxx/nlp/bytecup/Bytecup2018/processed'
bin_dir = os.path.join(root_path, 'bin')

dest_dir = os.path.join(bin_dir,'all_train_chunk')
if not os.path.exists(dest_dir): os.makedirs(dest_dir)

article_dir_list = ['train_%d'%i for i in range(9)]
article_dir_list.append('val')

def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)

for article_dir_name in article_dir_list:
    bin_chunk_article_dir = os.path.join(bin_dir, article_dir_name)
    copytree(bin_chunk_article_dir,dest_dir)
