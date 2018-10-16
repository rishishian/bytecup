# convert preprocessed article to tokenized version and (chunk) binary version
# split article to content and title, then convert to binary version
# save different training subset into different directory

import os
import struct
import subprocess
import sys

from tensorflow.core.example import example_pb2

root_path = '/home/lxx/nlp/bytecup/Bytecup2018/processed'
token_dir = os.path.join(root_path, 'token')
bin_dir = os.path.join(root_path, 'bin_val')
if not os.path.exists(token_dir): os.makedirs(token_dir)
if not os.path.exists(bin_dir): os.makedirs(bin_dir)

dm_single_close_quote = u'\u2019'  # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote,
              ")"]  # acceptable ways to end a sentence

# We use these to separate the summary sentences in the .bin datafiles
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'
CHUNK_SIZE = 1000  # num examples per chunk, for the chunked data


def chunk_file(bin_path, bin_chunk_dir):
    print('Original bin file:%s' %bin_path)
    print('Chunked bin dir:%s' %bin_chunk_dir)

    reader = open(bin_path, "rb")
    chunk = 0
    finished = False
    set_name = bin_chunk_dir.split('/')[-1]
    while not finished:
        chunk_fname = os.path.join(bin_chunk_dir, '%s_%03d.bin' % (set_name, chunk))  # new chunk
        with open(chunk_fname, 'wb') as writer:
            for _ in range(CHUNK_SIZE):
                len_bytes = reader.read(8)
                if not len_bytes:
                    finished = True
                    break
                str_len = struct.unpack('q', len_bytes)[0]
                example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
                writer.write(struct.pack('q', str_len))
                writer.write(struct.pack('%ds' % str_len, example_str))
            chunk += 1

def tokenize_stories(stories_dir, tokenized_stories_dir):
    """Maps a whole directory of .story files to a tokenized version using Stanford CoreNLP Tokenizer"""
    print("Preparing to tokenize %s to %s..." % (stories_dir, tokenized_stories_dir))
    stories = os.listdir(stories_dir)
    # make IO list file
    print("Making list of files to tokenize...")
    with open("mapping.txt", "w") as f:
        for s in stories:
            f.write("%s \t %s\n" % (os.path.join(stories_dir, s), os.path.join(tokenized_stories_dir, s)))
    # '-cp', '"/home/lxx/nlp/stanford-corenlp-full-2018-02-27"'
    command = ['java', 'edu.stanford.nlp.process.PTBTokenizer', '-ioFileList', '-preserveLines', 'mapping.txt']
    print("Tokenizing %i files in %s and saving in %s..." % (len(stories), stories_dir, tokenized_stories_dir))
    subprocess.call(command)
    print("Stanford CoreNLP Tokenizer has finished.")
    os.remove("mapping.txt")

    # Check that the tokenized stories directory contains the same number of files as the original directory
    num_orig = len(os.listdir(stories_dir))
    num_tokenized = len(os.listdir(tokenized_stories_dir))
    if num_orig != num_tokenized:
        raise Exception(
            "The tokenized stories directory %s contains %i files, but it should contain the same number as %s (which has %i files). Was there an error during tokenization?" % (
                tokenized_stories_dir, num_tokenized, stories_dir, num_orig))
    print("Successfully finished tokenizing %s to %s.\n" % (stories_dir, tokenized_stories_dir))


def read_text_file(text_file):
    lines = []
    with open(text_file, "r") as f:
        for line in f:
            lines.append(line.strip())
    return lines


def fix_missing_period(line):
    """Adds a period to a line that is missing a period"""
    if "@highlight" in line: return line
    if line == "": return line
    if line[-1] in END_TOKENS: return line
    return line + " ."


def get_art_abs(story_file):
    lines = read_text_file(story_file)

    # Lowercase everything
    lines = [line.lower() for line in lines]

    # Put periods on the ends of lines that are missing them (this is a problem in the dataset because many image captions don't end in periods; consequently they end up in the body of the article as run-on sentences)
    lines = [fix_missing_period(line) for line in lines]

    # Separate out article and abstract sentences
    article_lines = []
    highlights = []
    next_is_highlight = False
    for idx, line in enumerate(lines):
        if line == "":
            continue  # empty line
        elif line.startswith("@highlight"):
            next_is_highlight = True
        elif next_is_highlight:
            highlights.append(line)
        else:
            article_lines.append(line)

    # Make article into a single string
    article = ' '.join(article_lines)

    # Make abstract into a signle string, putting <s> and </s> tags around the sentences
    abstract = ' '.join(["%s %s %s" % (SENTENCE_START, sent, SENTENCE_END) for sent in highlights])

    return article, abstract


def write_to_bin(tokenized_stories_dir, out_file):
    junk_num = 0
    story_fnames = [name for name in os.listdir(tokenized_stories_dir)]
    num_stories = len(story_fnames)

    with open(out_file, 'wb') as writer:
        for idx, s in enumerate(story_fnames):
            if idx % 1000 == 0:
                print("Writing story %i of %i; %.2f percent done" % (
                    idx, num_stories, float(idx) * 100.0 / float(num_stories)))

            # Look in the tokenized story dirs to find the .story file corresponding to this url
            if os.path.isfile(os.path.join(tokenized_stories_dir, s)):
                story_file = os.path.join(tokenized_stories_dir, s)
                # print(story_file)
            else:
                print('Error: no data.')

            # Get the strings to write to .bin file
            article, abstract = get_art_abs(story_file)
            if idx == 0:
                print('data example:')
                print('article:')
                print(article)
                print('\nabstract:')
                print(abstract)

            # Write to tf.Example
            tf_example = example_pb2.Example()
            try:
                tf_example.features.feature['article'].bytes_list.value.extend([article.encode()])
                tf_example.features.feature['abstract'].bytes_list.value.extend([abstract.encode()])
                tf_example_str = tf_example.SerializeToString()
                str_len = len(tf_example_str)
                writer.write(struct.pack('q', str_len))
                writer.write(struct.pack('%ds' % str_len, tf_example_str))
            except:    # some may not use ascii coding
                junk_num += 1
                print('error file:%s' % story_file)

    print("Finished writing file %s\n" % out_file)
    print('total:%d, junk:%d' % (num_stories, junk_num))


if __name__ == '__main__':
    # default directory
    article_dir_list = []
    article_dir_list.append('val')

    for article_dir_name in article_dir_list:
        article_dir = os.path.join(root_path, article_dir_name)
        print('article dir: %s' % article_dir)

        token_article_dir = os.path.join(token_dir, article_dir_name)

        # Create some new directories
        if not os.path.exists(token_article_dir):
            os.makedirs(token_article_dir)
        else:
            print('already tokenized:%s, tokenize again' % token_article_dir)
        tokenize_stories(article_dir, token_article_dir)


        # Read the tokenized stories, do a little postprocessing then write to bin files
        bin_path = os.path.join(bin_dir, '%s.bin'%article_dir_name)
        write_to_bin(token_article_dir, bin_path)

        # Chunk the data. This splits each of train_*.bin into smaller chunks, each containing e.g. 1000 examples, and saves them
        bin_chunk_article_dir = os.path.join(bin_dir, article_dir_name)
        if not os.path.exists(bin_chunk_article_dir): os.makedirs(bin_chunk_article_dir)
        chunk_file(bin_path, bin_chunk_article_dir)
