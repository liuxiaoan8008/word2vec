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
sentences = read_data(data_path)
for l in sentences[:20]:
    print l
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
model = model.train(sentences)
print 'save model...'
model.save(model_path + 'jiuyang_'+model_name)
print
print 'finish.'

# print model.similarity(u'男人',u'女人')

import math
def cosine_similarity(v1,v2):
    "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]; y = v2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
    return sumxy/math.sqrt(sumxx*sumyy)

def sentence_word2vec_sim(text1,text2, model, dim):
    text1vec = np.asarray([0. for _ in range(dim)])
    text2vec = np.asarray([0. for _ in range(dim)])

    for word in text1:
        try:
            text1vec = np.add(text1vec,model.wv.word_vec(word))
        except Exception as e:
            print e
            text1vec = np.add(text1vec,np.asarray([0. for _ in range(dim)]))

    for word in text2:
        try:
            text2vec = np.add(text2vec,model.wv.word_vec(word))
        except Exception as e:
            print e
            text2vec = np.add(text2vec,np.asarray([0. for _ in range(dim)]))

    return cosine_similarity(text1vec,text2vec)



# print sentence_word2vec_sim(u'今晚饭还在煮就闻到烧焦的味了',u'机器有异味。', model, 50)

import jieba
import operator

def get_sim_text(in_file,sim_file,out_file,model):
    raw = []
    with open(in_file) as f:
        for line in f:
            line = unicode(line.strip(), 'utf-8')
            raw.append(line)

    question = []
    with open(sim_file) as f:
        for line in f:
            line = unicode(line.strip(), 'utf-8')
            question.append(line)

    seg_raw = []
    for x in raw:
        seg_raw.append(jieba.lcut(x))

    seg_question = []
    for x in question:
        seg_question.append(jieba.lcut(x))

    out_f = open(out_file, 'w')
    for raw_words,i in zip(seg_raw,range(len(raw))):
        sim = []
        for question_words in seg_question:
            sim.append(sentence_word2vec_sim(raw_words,question_words,model,50))
        max_index, max_value = max(enumerate(sim), key=operator.itemgetter(1))
        print max_value
        out_f.write('{raw_new},{sim_new},{si_new}\n'.format(raw_new=raw[i], sim_new=question[max_index], si_new=max_value))
    out_f.close()


# get_sim_text('./data/jiuyang_raw_data.txt','./data/jiuyang_question.txt','./data/out_question_sim.txt',model)

