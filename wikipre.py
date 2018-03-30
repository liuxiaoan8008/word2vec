#encoding=utf-8
import re
import sys
import codecs
import thulac
from bs4 import BeautifulSoup

in_path = './data/'
out_path = './data/'
model_path = ''

reload(sys)                      # reload 才能调用 setdefaultencoding 方法
sys.setdefaultencoding('utf-8')  # 设置 'utf-8'

def myfun(input_file):
    '''
    繁体转简体后，一些符号修正。
    :param input_file:
    :return: new file
    '''
    p1 = re.compile(ur'-\{.*?(zh-hans|zh-cn):([^;]*?)(;.*?)?\}-')
    p2 = re.compile(ur'[（\(][，；。？！\s]*[）\)]')
    p3 = re.compile(ur'[「『]')
    p4 = re.compile(ur'[」』]')
    outfile = codecs.open('std_' + input_file, 'w', 'utf-8')
    with codecs.open(input_file, 'r', 'utf-8') as myfile:
        for line in myfile:
            line = p1.sub(ur'\2', line)
            line = p2.sub(ur'', line)
            line = p3.sub(ur'“', line)
            line = p4.sub(ur'”', line)
            outfile.write(line)
    outfile.close()
    print 'finish file :',input_file


# data clean
def getsentence(input_file):
    '''
       1. 去掉标签
       2. 去标点符号
       3。去多余的空格
       :param input_file:
       :return:
       '''
    punc = re.compile(ur'[，；＃＄％＆＇（）()＊＋，·－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.\s]')
    outfile = codecs.open(out_path+'se_' + input_file, 'w', 'utf-8')
    with codecs.open(in_path+input_file, 'r', 'utf-8') as myfile:
        for line in myfile:
            line = BeautifulSoup(line.strip(), 'lxml').get_text()
            line = punc.sub(' ',line)
            line = re.sub('[ ]+', ' ', line)  # 去掉重复掉空格
            if len(line.strip()) == 0:
                continue
            outfile.write(line.strip()+'\n')

def wordcut(input_file):
    '''
    1. 切分成句子
    2。分词
    :param input_file:
    :return:
    '''
    thu1 = thulac.thulac(seg_only=True)
    outfile = codecs.open(out_path + 'in_' + input_file, 'w', 'utf-8')
    with codecs.open(in_path + input_file, 'r', 'utf-8') as myfile:
        for line in myfile:
            sublines = re.split(ur'[！？｡。]',line)
            for l in sublines:
                if len(l.strip()) == 0:
                    continue
                l = thu1.cut(l.strip(),text=True)
                outfile.write(l.strip()+'\n')


getsentence('jiuyang_log_v2.txt')
wordcut('se_jiuyang_log_v2.txt')
