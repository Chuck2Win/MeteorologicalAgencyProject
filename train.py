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
import time,datetime
from sklearn.metrics import classification_report
from tqdm import tqdm
import copy
from kobert_transformers import get_kobert_model,get_tokenizer
from matplotlib import pyplot as plt
import seaborn as sns
from model import kobert_classifier
import argparse
parser = argparse.ArgumentParser(description = '필요한 변수')
# Input data
parser.add_argument('--max_len', default = 64, type = int)
parser.add_argument('--class_1_max_len', default = 512, type = int)
parser.add_argument('--stopword', default = ['재배포 금지','무단배포', '무단전재'], type = list)
parser.add_argument('--oversampling', default = True, type = bool)
parser.add_argument('--train_file', default = './data/train_data', type = str)
parser.add_argument('--val_file', default = './data/val_data', type = str)
parser.add_argument('--test_file', default = './data/test_data', type = str)
parser.add_argument('--batch_size', default = 16, type = int)
parser.add_argument('--learning_rate', default = 1e-6, type = float)
parser.add_argument('--betas', default = (0.9, 0.999), type = tuple)
parser.add_argument('--eps', default = 1e-8, type = float)
parser.add_argument('--weight_decay', default = 1e-2, type = float)
parser.add_argument('--epochs', default = 100, type = int)
parser.add_argument('--T_0',default = 10, type = int)
parser.add_argument('--T_mult',default = 1, type = int)
parser.add_argument('--eta_min',default = 1e-9, type = int)
parser.add_argument('--model', default = 'Augmented', type = str)
parser.add_argument('--early_stop', default = True, type = bool)
parser.add_argument('--show_process', default = 10, type = int)
parser.add_argument('--how_many_epochs_more', default = 5, type = int)
parser.add_argument('--min_model', default = './min_model', type = str)
# model1
# model2
# model3
def train():
    # for early stopping
    min_epoch = None
    min_value = None
    min_model = None
    min_count = 0
    for epoch in tqdm(range(1, args.epochs+1),desc='epoch',mininterval = 300):
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
                print(classification_report(Actual,Predicted,digits=3))
                print("")
                print(" Val Average training loss: {0:.5f}".format(avg_val_loss))
                print(classification_report(Actual,Predicted,digits=3))
                print("")
    torch.save(min_model, args.min_model)
    
if __name__=='__main__':
    args = parser.parse_args()
    # BERT tokenizer
    tokenizer = get_tokenizer()
    # BERT model
    kobert = get_kobert_model()
    
    if args.model == 'Augmented':
       print('아직 미완') 
    else:
        train_data = pd.read_pickle(args.train_file)
        val_data = pd.read_pickle(args.val_file)
        test_data = pd.read_pickle(args.test_file)
    
    # Tensor Dataset
    train_data = TensorDataset(torch.LongTensor(train_data['ids'].tolist()), torch.LongTensor(train_data['mask'].tolist()), torch.LongTensor(train_data['len'].tolist()),torch.LongTensor(train_data['damage'].tolist()))
    val_data = TensorDataset(torch.LongTensor(val_data['ids'].tolist()), torch.LongTensor(val_data['mask'].tolist()), torch.LongTensor(val_data['len'].tolist()),torch.LongTensor(val_data['damage'].tolist()))
    test_data = TensorDataset(torch.LongTensor(test_data['ids'].tolist()), torch.LongTensor(test_data['mask'].tolist()), torch.LongTensor(test_data['len'].tolist()),torch.LongTensor(test_data['damage'].tolist()))
    # data loader
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size,drop_last=False)
    val_dataloader = DataLoader(val_data, batch_size=args.batch_size,drop_last=False)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size,drop_last=False)    
    
    if args.model == 'WeightedRandomSample':
        counts = np.array([(train_data['damage']==0).sum(),(train_data['damage']==1).sum()]) # 0,1
        weights = 1./counts
        train_target = train_data['damage'].values
        train_samples_weight = torch.FloatTensor([weights[t] for t in train_target])
        train_sampler = WeightedRandomSampler(train_samples_weight,len(train_samples_weight))
        val_target = val_data['damage'].values
        val_samples_weight = torch.FloatTensor([weights[t] for t in val_target])
        val_sampler = WeightedRandomSampler(val_samples_weight,len(val_samples_weight))
    
    # data loader
        train_dataloader = DataLoader(train_data, batch_size=args.batch_size,drop_last=False,sampler=train_sampler)
        val_dataloader = DataLoader(val_data, batch_size=args.batch_size,drop_last=False,sampler=val_sampler)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = kobert_classifier(kobert).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr = args.learning_rate, betas = args.betas, eps = args.eps, weight_decay = args.weight_decay)
    epochs = args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.T_0, T_mult=args.T_mult, eta_min=args.eta_min, last_epoch=-1, verbose=False)
    train()
