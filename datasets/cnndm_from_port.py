import pdb
import random
from datasets import load_dataset
from datasets import load_metric
import json

dataset = load_dataset('cnn_dailymail', '3.0.0')


train_dataset = dataset['test']
fw=open('cnndm_test.json','w')
for case in train_dataset:
    ARTICLE = case['article']
    highlights = case['highlights']
    content={}

    content['src']=ARTICLE
    src_length=len(ARTICLE.split())
    content['tgt']=highlights
    tgt_length=len(highlights.split())
    if tgt_length!=0:
        ratio=int(src_length/tgt_length)
        if ratio==0 or ratio==1:
            continue
        content['idx']='cnndm'
        json.dump(content,fw)
        fw.write('\n')


