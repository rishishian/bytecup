# rename decoded result to standard answer

import os
import shutil

root_path = '/home/lxx/nlp/bytecup/Bytecup2018/log/pg_byte_all_800.5.23'
log_name = 'decode_val_800maxenc_4beam_5mindec_23maxdec_ckpt-101903'
decoded_dir = os.path.join(root_path, log_name, 'decoded')
reference_dir  = os.path.join(root_path, log_name, 'reference')
renamed_dir = os.path.join(root_path, log_name, 'raw_result')
if not os.path.exists(renamed_dir): os.makedirs(renamed_dir)
new_result_dir = os.path.join(root_path, log_name, 'result')
if not os.path.exists(new_result_dir): os.makedirs(new_result_dir)

# rename result to standard name style
for decode_file_name in os.listdir(decoded_dir):
    reference_file = os.path.join(reference_dir, decode_file_name.replace('decoded', 'reference'))
    # print(reference_file)
    with open(reference_file, 'r') as f:
        reference = f.readline()   # "title 213 ."
        id = reference.split(' ')[1]
    renamed_path = os.path.join(renamed_dir, '%s.txt'%id)
    decode_file = os.path.join(decoded_dir, decode_file_name)
    shutil.copy(decode_file,renamed_path)

# junk file or difficult file, just copy answer from validation result sample
copy_ids = ['79','355','825']
for id in copy_ids:
    sample_path =  os.path.join('/home/lxx/nlp/bytecup/Bytecup2018/bytecup_validation_sample/result', '%s.txt'%id)
    renamed_path = os.path.join(renamed_dir, '%s.txt'%id)
    shutil.copy(sample_path,renamed_path)

# post-process
multi_lines_cnt = 0
eliminate_list = ['[UNK]', '.', ',', '?', '\'', '`', '[UNK]', ':']
for file_name in os.listdir(renamed_dir):
    file_path = os.path.join(renamed_dir, file_name)
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
        # if file_name == '419.txt':
        #     print(file_name)
        #     print(tokens)
        for idx, token in enumerate(tokens):
            if tokens[idx] in eliminate_list:
                tokens[idx] = ''
            elif tokens[idx] == '\\':
                tokens[idx] = ''
                try:
                    tokens[idx + 1] = ''
                except:
                    print('tokens idx out of range: %s' % file_name)
            # elif tokens[idx]=='$':  # number maybe should be saved also
            #     continue
            # elif len(tokens[idx])==1 and not tokens[idx][0].isalpha():
            #     tokens[idx] = ''
        new_tokens = []
        for i in range(len(tokens)):
            new_tokens.append(tokens[i])
            if tokens[i] != '':
                new_tokens.append(' ')
        new_title = ''
        for i in range(len(new_tokens)):
            new_title = new_title + new_tokens[i]
        new_title = new_title[:-1]
        new_title_path = os.path.join(new_result_dir, file_name)
        with open(new_title_path, 'w') as wf:
            wf.writelines(new_title)
