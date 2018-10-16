# preprocess bytecup data
# (zip file ->) txt file -> split into multiple article and corresponding title

import argparse
import ast
import os
import zipfile

from enum import Enum

parser = argparse.ArgumentParser(description='split bytecup data into multiple article and corresponding title')
parser.add_argument('--root_path', default='/home/lxx/nlp/bytecup/Bytecup2018', type=str)
parser.add_argument('--zip_file_name', default='bytecup.corpus.train.8.zip', type=str)
parser.add_argument('--test_size', default=10000, type=int)
parser.add_argument('--print_freq', default=2000, type=int)

opt = parser.parse_args()

chunk_id = opt.zip_file_name.split('.')[3]


class Mode(Enum):
    TRAIN = 1
    VAL = 2


def read_file(txt_file_path):
    print('load data from %s' % txt_file_path)
    with open(txt_file_path) as txt_file:
        data = txt_file.readlines()
    data = [x.strip() for x in data]
    cnt = len(data)
    print('total %d articles in this file' % cnt)

    if 'title' in data[0]:
        mode = Mode.TRAIN
    else:
        mode = Mode.VAL

    train_article_dir = os.path.join(opt.root_path, 'train_article')
    if not os.path.exists(train_article_dir):
        os.makedirs(train_article_dir)
    # train_titles_dir = os.path.join(opt.root_path, 'train_title')
    # if not os.path.exists(train_titles_dir):
    #     os.makedirs(train_titles_dir)

    val_article_dir = os.path.join(opt.root_path, 'val_article')
    if not os.path.exists(val_article_dir):
        os.makedirs(val_article_dir)

    for i in range(cnt):
        if i % opt.print_freq == 0:
            print(i)
        if i == opt.test_size:
            print('test size: %d' % opt.test_size)
            break
        item = ast.literal_eval(data[i])
        id = item['id']
        content = item['content']
        if mode == Mode.TRAIN:
            # split article and title, and save them in different dirs
            # article_file_name = chunk_id + '_' + str(id) + '.article'
            # article_file_path = os.path.join(opt.root_path, 'train_article', article_file_name)
            # with open(article_file_path, 'w') as f:
            #     f.write(content)
            #
            # title = item['title']
            # title_file_name = chunk_id + '_' + str(id) + '.title'
            # title_file_path = os.path.join(opt.root_path, 'train_title', title_file_name)
            # with open(title_file_path, 'w') as f:
            #     f.write(title)

            # following the official style, save multiple (article, title) tuples
            article_file_name = chunk_id + '_' + str(id) + '.article'
            article_file_path = os.path.join(opt.root_path, 'train_article', article_file_name)
            title = item['title']
            article = content+'\n@highlight\n'+title
            with open(article_file_path, 'w') as f:
                f.write(article)
        else:
            # article_file_name = chunk_id + '_' + str(id) + '.article'
            # article_file_path = os.path.join(opt.root_path, 'val_article', article_file_name)
            # with open(article_file_path, 'w') as f:
            #     f.write(content)
            article_file_name = chunk_id + '_' + str(id) + '.article'
            article_file_path = os.path.join(opt.root_path, 'val_article', article_file_name)
            # for data without title, for robustness of code, add title manually
            title = 'title\n'
            article = content+'\n@highlight\n'+title
            with open(article_file_path, 'w') as f:
                f.write(article)
    return


if __name__ == '__main__':

    print('zip file name: %s' % opt.zip_file_name)
    zip_file_path = os.path.join(opt.root_path, opt.zip_file_name)
    txt_file_path = zip_file_path.replace('zip', 'txt')

    if not os.path.exists(txt_file_path):
        print('unzip file %s' % zip_file_path)
        zip_ref = zipfile.ZipFile(zip_file_path, 'r')
        target_dir = opt.root_path
        zip_ref.extractall(target_dir)
        zip_ref.close()
    else:
        print('txt file already exists')

    read_file(txt_file_path)
