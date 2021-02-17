# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

class kobert_classifier(nn.Module):
    def __init__(self, kobert):
        super().__init__()
        self.bert = kobert
        self.classifier = nn.Linear(768+1,2)
    def forward(self,input_ids,attention_mask,length):
        output = self.bert.forward(input_ids = input_ids, attention_mask = attention_mask)
        length = length.unsqueeze(1)
        input = torch.cat([output.pooler_output,length],dim=1)
        predict = self.classifier.forward(input)
        return predict