import pdb
import random
import json
data_name='train'
f1=open('cnndm_{}.json'.format(data_name))
lines1=f1.readlines()
new_lines1=[]
for line in lines1:
    content=json.loads(line)
    content['idx']='cnndm'
    new_line=json.dumps(content)
    new_lines1.append(new_line)

f2=open('wikihow_{}.json'.format(data_name))
lines2=f2.readlines()
new_lines2=[]
for line in lines2:
    content=json.loads(line)
    content['idx']='wiki'
    new_line=json.dumps(content)
    new_lines2.append(new_line)

f3=open('pubmed_{}.json'.format(data_name))
lines3=f3.readlines()
new_lines3=[]
for line in lines3:
    content=json.loads(line)
    content['idx']='pubmed'
    new_line=json.dumps(content)
    new_lines3.append(new_line)

all_lines=new_lines2+new_lines1+new_lines3
random.shuffle(all_lines)
fw=open('cnndm_wiki_pubmed_{}.json'.format(data_name),'w')
fw.write('\n'.join(all_lines))