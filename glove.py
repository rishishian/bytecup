'''
according to vocab, extract some lines from glove word embeddings and sort them
'''
import numpy as np

dim = 100
vocab_path = '/data2/lixixian/bytecup/vocab'
glove_path = '/data4/zhangdehao/lxx/glove.twitter.27B.%dd.txt' % dim
selected_embedding_path = '/data2/lixixian/bytecup/selected_glove.npy'
selected_embedding = None
glove_embedding = dict()

zero_embedding = np.zeros(dim, dtype=np.float32)

with open(glove_path, 'r') as glove_file:
    glove = glove_file.readlines()

    for word_embedding in glove:
        word = word_embedding.split(' ')[0]
        word = word[1:-1]  # remove < and >
        embedding = np.float32(word_embedding.split(' ')[1:])
        glove_embedding[word] = embedding
    print('finish reading glove embedding')

    with open(vocab_path, 'r') as vocab_file:
        for line in vocab_file.readlines():
            word = line.split(' ')[0]
            if word in glove_embedding.keys():
                embedding = glove_embedding[word]
            else:
                embedding = zero_embedding

            if selected_embedding is None:
                selected_embedding = embedding
            else:
                selected_embedding = np.row_stack((selected_embedding, embedding))


np.save(selected_embedding_path, selected_embedding)

# load glove embedding
# reference code:
# W = tf.get_variable(name="W", shape=embedding.shape, initializer=tf.constant_initializer(embedding),
#                     trainable=False)

# selected_embedding_path = '/data2/lixixian/bytecup/selected_glove.npy'
# selected_embedding = np.load(selected_embedding_path)
# embedding = tf.get_variable('embedding', [vsize, hps.emb_dim], dtype=tf.float32,
#                             initializer=tf.constant_initializer(selected_embedding), trainable=True)
