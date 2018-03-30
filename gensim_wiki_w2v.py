#coding=utf-8
import os
from gensim.models import word2vec
import thulac
from bs4 import BeautifulSoup
import chardet
import time
import codecs
import re
import sys
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

data_path = './in/'
model_path= './model/'
reload(sys)                      # reload 才能调用 setdefaultencoding 方法
sys.setdefaultencoding('utf-8')  # 设置 'utf-8'

# Set values for various parameters
num_features = 50    # Word vector dimensionality
min_word_count = 20   # Minimum word count
num_workers = 6       # Number of threads to run in parallel
context = 10          # Context window size
downsampling = 1e-3   # Downsample setting for frequent words
sg = 1                # 1 means ues skip-gram
hs = 1                # 0 means using negative sampling
model_name = 'w2v_wiki_'+str(num_features)+'.model'


def read_data(dirname):
    sentences = []
    for fname in os.listdir(dirname):
        sentences += word2vec.LineSentence(os.path.join(dirname,fname))
    return sentences

def mkdir(path):
    isExists = os.path.exists(path)
    if not isExists:
        print path + ' build success.'
        os.makedirs(path)
        return True
    else:
        print path + ' already exist.'
        return False

def train_model(sentences):
    print 'build model path...'
    mkdir(model_path)
    print
    print 'train word2vec ...'
    model = word2vec.Word2Vec(sentences, workers=num_workers, size=num_features, min_count=min_word_count,
                              window=context, sample=downsampling,sg=sg,hs=hs)
    print
    print 'save model...'
    model.save(model_path + model_name)
    print
    print 'finish.'

# print 'Step 1: read clean data .... '
# sentences = read_data(data_path)
# for l in sentences[:20]:
#     print l
# print
# start = time.time()
# train_model(sentences)
# end = time.time()
# elapsed = end - start
# print 'The time token for training gensim-SG model : ',elapsed/60,'min'
# 
# wordcut('std_zh_wiki_00')
# wordcut('std_zh_wiki_01')
# wordcut('std_zh_wiki_02')

model = word2vec.Word2Vec.load(model_path+model_name)
print model.similarity(u'男人',u'女人')


def sentence_word2vec_sim(text1,text2, model, dim):
    text1vec = np.asarray([0. for _ in range(dim)])
    text2vec = np.asarray([0. for _ in range(dim)])

    for word in text1:
        text1vec = np.add(text1vec,model.wv.word_vec(word))

    for word in text2:
        text2vec = np.add(text2vec,model.wv.word_vec(word))
    return cosine_similarity(text1vec,text2vec)



sentence_word2vec_sim(u'今晚饭还在煮就闻到烧焦的味了',u'机器有异味', model, 50)



