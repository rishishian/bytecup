# preprocess all bytecup data
# (zip file ->) txt file -> split into multiple article and corresponding title

import argparse
import ast
import os
import zipfile

from enum import Enum

parser = argparse.ArgumentParser(description='split bytecup data into multiple article and corresponding title')
parser.add_argument('--root_path', default='/home/lxx/nlp/bytecup/Bytecup2018', type=str)
parser.add_argument('--print_freq', default=10000, type=int)

opt = parser.parse_args()

class Mode(Enum):
    TRAIN = 1
    VAL = 2

total_num = 0
junk_num = 0

def read_file(txt_file_path):
    print('load data from %s' % txt_file_path)
    with open(txt_file_path) as txt_file:
        data = txt_file.readlines()
    data = [x.strip() for x in data]
    cnt = len(data)
    chunk_id = txt_file_path.split('/')[-1].split('.')[3]
    print('total %d articles in this file' % cnt)

    if 'title' in data[0]:
        mode = Mode.TRAIN
    else:
        mode = Mode.VAL

    train_article_dir = os.path.join(opt.root_path, 'processed', 'train_'+chunk_id)
    if not os.path.exists(train_article_dir):
        os.makedirs(train_article_dir)

    val_article_dir = os.path.join(opt.root_path, 'processed', 'val')
    if not os.path.exists(val_article_dir):
        os.makedirs(val_article_dir)

    for i in range(cnt):
        # if i % opt.print_freq == 0:
            # print(i)
        try:
            item = ast.literal_eval(data[i])  # covert data to dict in python
            global total_num
            total_num += 1
        except:
            global junk_num
            junk_num += 1
            continue
        id = item['id']
        content = item['content']
        if mode == Mode.TRAIN:
            # following the official style, save multiple (article, title) tuples
            article_file_name = chunk_id + '_' + str(id) + '.article'
            article_file_path = os.path.join(train_article_dir, article_file_name)
            title = item['title']
            article = content+'\n@highlight\n'+title
            with open(article_file_path, 'w') as f:
                f.write(article)
        else:
            article_file_name = 'val_' + str(id) + '.article'
            article_file_path = os.path.join(val_article_dir, article_file_name)
            # for data without title, for robustness of code, add title manually
            title = 'title ' + str(id) + '\n'
            article = content+'\n@highlight\n'+title
            with open(article_file_path, 'w') as f:
                f.write(article)

        # for debugging
        # if i == 0:
        #     print(chunk_id)
        #     print(article_file_path)
        #     os._exit()
    return


if __name__ == '__main__':
    txt_file_name = 'bytecup.corpus.validation_set.txt'
    txt_file_path = os.path.join(opt.root_path, txt_file_name)
    read_file(txt_file_path)
    print('total num for eval:%d'%total_num)
    print('junk num:%d'%junk_num)
