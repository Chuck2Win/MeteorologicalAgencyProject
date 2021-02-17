# -*- coding: utf-8 -*-
import re
import kss
import pandas as pd
from kobert_transformers import get_tokenizer
import torch
import argparse

parser = argparse.ArgumentParser(description = '필요한 변수')
# Input data
parser.add_argument('--max_len', default = 512, type = int)
parser.add_argument('--stopword', default = ['사진','기자','배포금지','무단배포','@','뉴스룸','닷컴','저작권',"좋아요", "스크랩하기", "공유하기", "글씨", "작게보기", "고화질", "표준화질", "키보드", "컨트롤", "동영상영역", "댓글", "크게보기"], type = list)
parser.add_argument('--data_file', default = './data/Final_data.csv', type = str)
parser.add_argument('--return_file', default = './data/preprocessed_data', type = str)

# 불용어 제거
def is_not_in(sentence,stopword):
    for i in stopword:
        if i in sentence:
            return False
    return True

# cleansing 작업
def cleansing(text,stopword):
    # \n+은 ' '로 치환
    text=re.sub('\n+',' ',text)
    # 괄호 제거
    text=re.sub('\(.+?\)|\[.+?\]|\<.+?\>','',text)
    # 한글, ., ,숫자가 아닌 경우 빈칸으로 치환
    text=re.sub('[^ ㄱ-ㅣ가-힣,.0-9]+','',text)
    # 문장 분리기를 활용해서 불용어가 있는 경우 제거
    result=[]
    for sentence in kss.split_sentences(text):
        if is_not_in(sentence,stopword):
            result.append(sentence)
    return ' '.join(result)

class preprocessing:
    def __init__(self):
        self.data = pd.read_csv(data_file)
        self.data.rename(columns = {'피해' : 'damage', 'Title':'title'}, inplace = True)
        self.data['Title'] = self.data['title'].apply(lambda i : cleansing(i,stopword))
        self.data['Title'] = self.data['maintext'].apply(lambda i : cleansing(i,stopword))
        self.data['Total'] = self.data['Title']+'. '+self.data['Title']
        # BERT tokenizer로 tokenizing
        self.data['tokenized']=self.data['Total'].apply(lambda i : tokenizer.encode(i))
        # length
        self.data['len']=self.data['tokenized'].apply(lambda i : len(i))
        
        # 길이가 10이하인 경우 제거
        self.data = self.data.loc[self.data['len']>10,:]
        
        # max len으로 자르고, 모자란 부분은 패딩으로 채움
        self.data['ids'] = self.data['Total'].apply(lambda i : tokenizer.encode(i,add_special_tokens=True,truncation=True,padding='max_length',max_length=max_len))
        # attention mask - mask될 부분은 0, 아닌 부분은 1
        self.data['mask']=(torch.tensor(self.data['ids'].tolist()).eq(1)==0).long().tolist()
    def return_data(self):
        self.data.loc[:,['ids','mask','len','damage','Total']].to_pickle(return_file)

if __name__ == '__main__':
    tokenizer = get_tokenizer()
    args = parser.parse_args()
    max_len = args.max_len
    stopword = args.stopword
    data_file = args.data_file
    return_file = args.return_file
    preprocessing().return_data()

