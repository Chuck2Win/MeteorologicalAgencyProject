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
parser.add_argument('--train_file', default = './data/train_data', type = str)
parser.add_argument('--val_file', default = './data/val_data', type = str)
parser.add_argument('--test_file', default = './data/test_data', type = str)
parser.add_argument('--generated_sentence_file', default = './data/generated_sentence', type = str)
parser.add_argument('--val_size', default = 0.3, type = float)
parser.add_argument('--batch_size', default = 8, type = int)
parser.add_argument('--max_len', default = 512, type = int)
parser.add_argument('--learning_rate', default = 1e-6, type = float)
parser.add_argument('--betas', default = [0.9, 0.999], type = list)
parser.add_argument('--eps', default = 1e-8, type = float)
parser.add_argument('--weight_decay', default = 1e-2, type = float)
parser.add_argument('--epochs', default = 100, type = int)
parser.add_argument('--T_0',default = 10, type = int)
parser.add_argument('--T_mult',default = 1, type = int)
parser.add_argument('--eta_min',default = 1e-9, type = int)
parser.add_argument('--model', default = 'Augmentation', type = str)
# model : Augmentation, None, WeightedRandomSample
parser.add_argument('--early_stop', default = True, type = bool)
parser.add_argument('--show_process', default = 10, type = int)
parser.add_argument('--how_many_epochs_more', default = 5, type = int)
parser.add_argument('--min_model', default = './min_model', type = str)

def train():
    # for early stopping
    min_epoch = None
    min_value = None
    min_model = None
    min_count = 0
    for epoch in tqdm(range(1, args.epochs+1),desc='epoch',mininterval = args.show_process*60):
        # ========================================
        #               Training
        # ========================================
        total_loss = 0
        Predicted=[]
        Actual=[]
        model.train()
        for batch in train_dataloader:
            optimizer.zero_grad()
            batch = tuple(t.to(device) for t in batch)
            input_ids, attention_mask, length, labels = batch
            outputs = model.forward(input_ids, attention_mask, length)
            loss = F.cross_entropy(outputs, labels)
            predicted = outputs.argmax(-1).tolist()
            Predicted.extend(predicted)
            Actual.extend(labels.tolist())
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        scheduler.step()
        avg_train_loss = total_loss / len(train_dataloader)
        
        with torch.no_grad():
            model.eval()
            total_loss = 0
            Predicted=[]
            Actual=[]
            for batch in val_dataloader:
                batch = tuple(t.to(device) for t in batch)
                # ids, mask, ord, length 순
                input_ids, attention_mask,length, labels = batch
                outputs = model.forward(input_ids, attention_mask, length)
                loss = F.cross_entropy(outputs, labels)
                predicted = outputs.argmax(-1).tolist()
                Predicted.extend(predicted)
                Actual.extend(labels.tolist())
                total_loss += loss.item()
            avg_val_loss = total_loss / len(val_dataloader) 
            
            if min_value is None:
                min_value = avg_val_loss
                min_model = copy.deepcopy(model.state_dict())
            else:
                current = avg_val_loss
                if min_value<current:
                    if args.early_stop:
                        if min_count == args.how_many_epochs_more-1:
                            print('early stop')
                            print('min_epoch : %d'%min_epoch)
                            break
                        else:
                            min_count += 1
                       
                else:
                    min_value = current
                    min_model = copy.deepcopy(model.state_dict())
                    min_count = 0
                    min_epoch = epoch            
            
            if epoch%args.show_process == 0:
                print(epoch)
                print("")
                print("  Average training loss: {0:.5f}".format(avg_train_loss))
                print(classification_report(Actual,Predicted,digits=4))
                print("")
                print(" Val Average training loss: {0:.5f}".format(avg_val_loss))
                print(classification_report(Actual,Predicted,digits=4))
                print("")
    torch.save(min_model, args.min_model)

class Config(dict): 
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

def generated_sentence_preprocess():
    generated_sentence = pd.read_pickle(args.generated_sentence_file)
    generated_sentence = df({'Total':generated_sentence})
    generated_sentence['damage']=1
    generated_sentence['len']=generated_sentence['Total'].apply(lambda i : len(tokenizer.encode(i)))
    generated_sentence = generated_sentence.loc[generated_sentence['len']>10,:]
    generated_sentence['ids'] = generated_sentence['Total'].apply(lambda i : tokenizer.encode(i,add_special_tokens=True,truncation=True,padding='max_length',max_length=args.max_len))
        # attention mask - mask될 부분은 0, 아닌 부분은 1
    generated_sentence['mask']=(torch.tensor(generated_sentence['ids'].tolist()).eq(1)==0).long().tolist()
    return generated_sentence

def mix_train_val_generated(train_data,val_data,generated_sentence):
    mix_data = pd.concat([train_data,val_data])
    how_many=(mix_data.damage==0).sum()-(mix_data.damage==1).sum()
    sampled_generated_sentence=generated_sentence.sample(how_many)
    mix_data = pd.concat([mix_data,sampled_generated_sentence])
    new_train_data,new_val_data = train_test_split(mix_data,test_size = args.val_size,shuffle=True)
    return new_train_data,new_val_data
    
def load_things():
    train_data = pd.read_pickle(args.train_file)
    val_data = pd.read_pickle(args.val_file)
    test_data = pd.read_pickle(args.test_file)
    if args.model == 'Augmentation':
        generated_sentence = generated_sentence_preprocess()
        train_data,val_data = mix_train_val_generated(train_data,val_data,generated_sentence)
        train_data.to_pickle('./data/augmentation_train_data')
        val_data.to_pickle('./data/augmentation_val_data')
    # Tensor Dataset
    elif args.model == 'WeightedRandomSample':
        counts = np.array([(train_data['damage']==0).sum(),(train_data['damage']==1).sum()]) # 0,1
        weights = 1./counts
        train_target = train_data['damage'].values
        train_samples_weight = torch.FloatTensor([weights[t] for t in train_target])
        train_sampler = WeightedRandomSampler(train_samples_weight,len(train_samples_weight))
        val_target = val_data['damage'].values
        val_samples_weight = torch.FloatTensor([weights[t] for t in val_target])
        val_sampler = WeightedRandomSampler(val_samples_weight,len(val_samples_weight))
        
    train_data = TensorDataset(torch.LongTensor(train_data['ids'].tolist()), torch.LongTensor(train_data['mask'].tolist()), torch.LongTensor(train_data['len'].tolist()),torch.LongTensor(train_data['damage'].tolist()))
    val_data = TensorDataset(torch.LongTensor(val_data['ids'].tolist()), torch.LongTensor(val_data['mask'].tolist()), torch.LongTensor(val_data['len'].tolist()),torch.LongTensor(val_data['damage'].tolist()))
    test_data = TensorDataset(torch.LongTensor(test_data['ids'].tolist()), torch.LongTensor(test_data['mask'].tolist()), torch.LongTensor(test_data['len'].tolist()),torch.LongTensor(test_data['damage'].tolist()))
    # data loader
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size,drop_last=False)
    val_dataloader = DataLoader(val_data, batch_size=args.batch_size,drop_last=False)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size,drop_last=False)    

    if args.model == 'WeightedRandomSample':
        train_dataloader = DataLoader(train_data, batch_size=args.batch_size,drop_last=False,sampler=train_sampler)
        val_dataloader = DataLoader(val_data, batch_size=args.batch_size,drop_last=False,sampler=val_sampler)
    return train_dataloader, val_dataloader, test_dataloader
    
if __name__=='__main__':
    args = parser.parse_args()
    # BERT tokenizer
    tokenizer = get_tokenizer()
    # BERT model
    kobert = get_kobert_model()
    # dataloader
    train_dataloader, val_dataloader, test_dataloader=load_things()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = kobert_classifier(kobert).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr = args.learning_rate, betas = args.betas, eps = args.eps, weight_decay = args.weight_decay,)
    epochs = args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.T_0, T_mult=args.T_mult, eta_min=args.eta_min, last_epoch=-1, verbose=False)
    train()
