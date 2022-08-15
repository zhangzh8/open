import re

import pandas as pd
import jieba
import csv
def deleteByStartAndEnd(s):
    s = s.lower()
    while(s.__contains__('@')):
        x1 = s.index('@')
        if(s.__contains__(':')):
            if(s.index(':')>s.index('@')):
                x2 = s.index(':')+1
            else:
                return s
        elif(s.__contains__(' ')  ):
            if(s.index(' ')>s.index('@')):
                x2 = s.index(' ')+1
            else:
                return s
        else:
            x2 = x1+1
        x3 = s[x1:x2]
        result = s.replace(x3, "")
        s=result

    return s

dataset = pd.read_csv("weibo_senti_100k.csv", engine='python', header=None)
from torchtext.legacy import data
LABEL = data.LabelField() # 标签
REVIEW = data.Field() # 内容/文本
fields = [('label',LABEL),('review',REVIEW)]
reviewDataset = data.TabularDataset(
   path = 'weibo_senti_100k.csv',
   format = 'CSV',
   fields = fields,
   skip_header = True
)
punctuation = ']/\!,;:?、，；。['
for i in range(len(reviewDataset)):
    str0=''
    str0="".join(reviewDataset.examples[i].label)
    str1=''
    str1 = "".join(reviewDataset.examples[i].review)
    if(str0 == '1'):
        f = open(r"pos_all.txt", 'a', encoding="utf-8")
        str1 = deleteByStartAndEnd(str1)
        #while(s.__contains__('@')):  result = s.replace(s[s.index('@'):s.index(':')], "")
        str1 = re.sub(r'[{}]+'.format(punctuation), '', str1)#punctuation = '/\!,;:?、，；。'
        f.write(str1)
        f.write('\n')
    else:
        f = open(r"neg_all.txt", 'a', encoding="utf-8")
        str1 = deleteByStartAndEnd(str1)
        str1 = re.sub(r'[{}]+'.format(punctuation), '', str1)
        f.write(str1)
        f.write('\n')

