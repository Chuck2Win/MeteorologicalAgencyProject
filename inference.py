# -*- coding: utf-8 -*-
import pandas as pd
from pandas import DataFrame as df
import numpy as np
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
import os
from pandas import DataFrame as df
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import copy
from kobert_transformers import get_kobert_model,get_tokenizer
from model import kobert_classifier
import argparse
parser = argparse.ArgumentParser(description = '필요한 변수')
# Input data

parser.add_argument('--stopword', default = ['사진','기자','배포금지','무단배포','@','뉴스룸','닷컴','저작권',"좋아요", "스크랩하기", "공유하기", "글씨", "작게보기", "고화질", "표준화질", "키보드", "컨트롤", "동영상영역", "댓글", "크게보기"], type = list)
parser.add_argument('--data_file', default = './data/inference_data', type = str)
parser.add_argument('--batch_size', default = 16, type = int)
parser.add_argument('--max_len', default = 512, type = int)
parser.add_argument('--min_model', default = './min_model', type = str)
parser.add_argument('--predict', default = 'True', type = str)
parser.add_argument('--predict_file', default = './predict.csv', type = str)

class Config(dict): 
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

def load_dataloader():
    data = pd.read_pickle(args.data_file)
    if 'damage' in data.columns:
        data = TensorDataset(torch.LongTensor(data['ids'].tolist()), torch.LongTensor(data['mask'].tolist()), torch.LongTensor(data['len'].tolist()),torch.LongTensor(data['damage'].tolist()))
        
    else:
        data = TensorDataset(torch.LongTensor(data['ids'].tolist()), torch.LongTensor(data['mask'].tolist()), torch.LongTensor(data['len'].tolist()))
        
    dataloader = DataLoader(data, batch_size=args.batch_size,drop_last=False)
    return dataloader

def test():
    with torch.no_grad():
        model.eval()
        total_loss = 0
        Predicted=[]
        Actual=[]
        N = 0
        for batch in dataloader:
            batch = tuple(t.to(device) for t in batch)
            # ids, mask, ord, length 순
            input_ids, attention_mask, length, labels = batch
            outputs = model.forward(input_ids, attention_mask, length)
            loss = F.cross_entropy(outputs, labels)
            predicted = outputs.argmax(-1).tolist()
            Predicted.extend(predicted)
            Actual.extend(labels.tolist())
            N+=len(labels)
            total_loss += loss.item()
        avg_loss = total_loss / N   
        print("")
        print(" Average Loss: {0:.5f}".format(avg_loss))
        print(classification_report(Actual,Predicted,digits=4))
        print("")

def predict():
    with torch.no_grad():
        model.eval()
        total_loss = 0
        Predicted=[]
        N = 0
        for batch in dataloader:
            batch = tuple(t.to(device) for t in batch)
            input_ids, attention_mask, length = batch
            outputs = model.forward(input_ids, attention_mask, length)
            predicted = outputs.argmax(-1).tolist()
            Predicted.extend(predicted)
            N+=len(labels)
        result = df({'damage':Predict})
        result.to_csv(args.predict_file)
    
if __name__=='__main__':
    args = parser.parse_args()
    # BERT tokenizer
    tokenizer = get_tokenizer()
    # BERT model
    kobert = get_kobert_model()
    # dataloader
    dataloader=load_dataloader()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = kobert_classifier(kobert).to(device)
    # model load
    model.load_state_dict(torch.load(args.min_model))
    #predict()
    if args.predict=='True':
        predict()
    else:
        test()