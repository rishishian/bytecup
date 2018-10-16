import os

root_path = '/home/lxx/nlp/bytecup/Bytecup2018/log/pg_byte_all_800.5.20/decode_val_800maxenc_4beam_5mindec_20maxdec_ckpt-176034'
txt_dir = os.path.join(root_path, 'result')

file_cnt = 0

# what we need to get
LEN_MIN = 5 - 1
LEN_MAX = 20 + 1
len_arr = list(range(LEN_MIN, LEN_MAX))

min_len = 50
max_len = 0
avg_len = 0

unk_cnt = 0
unicode_cnt = 0
multi_lines_cnt = 0

for file_name in os.listdir(txt_dir):
    file_cnt += 1
    file_path = os.path.join(txt_dir, file_name)
    with open(file_path, 'r') as f:
        title = f.readlines()
        if len(title) != 1:
            multi_lines_cnt += 1
            print('multi lines: %s' % file_name[:-4])
            tmp = title
            title = ''
            for line in tmp:
                title += line
        else:
            title = title[0]
        tokens = title.split(' ')
        length = len(tokens)
        if length > max_len:
            max_len = length
        elif length < min_len:
            min_len = length
        avg_len += length
        try:
            len_arr[length - LEN_MIN] += 1
        except:
            print('%s length too large:%d' % (file_name, length))

        for token in tokens:
            if token == '[UNK]':
                unk_cnt += 1
            elif token == '\\':
                unicode_cnt += 1

for i in range(len(len_arr)):
    print('len:%d, cnt:%d' % (LEN_MIN + i, len_arr[i]))
avg_len = avg_len / float(file_cnt)
print('avg_len:%d' % avg_len)
print('unk_cnt:%d' % unk_cnt)
print('unicode_cnt:%d' % unicode_cnt)
print('multi_lines_cnt:%d' % multi_lines_cnt)
