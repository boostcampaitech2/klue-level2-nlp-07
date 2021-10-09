import pickle as pickle
import os
import pandas as pd
import torch
import sklearn
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer
from load_data import *
import argparse
def num_to_label(label):
  num_label = []
  with open('dict_num_to_label.pkl', 'rb') as f:
    dict_num_to_label = pickle.load(f)
  for v in label:
    num_label.append(dict_num_to_label[v])
  
  return num_label

def klue_re_micro_f1(preds, labels):
    """KLUE-RE micro f1 (except no_relation)"""
    label_list = ['no_relation', 'org:top_members/employees', 'org:members',
       'org:product', 'per:title', 'org:alternate_names',
       'per:employee_of', 'org:place_of_headquarters', 'per:product',
       'org:number_of_employees/members', 'per:children',
       'per:place_of_residence', 'per:alternate_names',
       'per:other_family', 'per:colleagues', 'per:origin', 'per:siblings',
       'per:spouse', 'org:founded', 'org:political/religious_affiliation',
       'org:member_of', 'per:parents', 'org:dissolved',
       'per:schools_attended', 'per:date_of_death', 'per:date_of_birth',
       'per:place_of_birth', 'per:place_of_death', 'org:founded_by',
       'per:religion']
    no_relation_label_idx = label_list.index("no_relation")
    label_indices = list(range(len(label_list)))
    label_indices.remove(no_relation_label_idx)
    return sklearn.metrics.f1_score(labels, preds, average="micro", labels=label_indices) * 100.0

def klue_re_auprc(probs, labels):
    """KLUE-RE AUPRC (with no_relation)"""
    labels = np.eye(30)[labels]

    score = np.zeros((30,))
    for c in range(30):
        targets_c = labels.take([c], axis=1).ravel()
        preds_c = probs.take([c], axis=1).ravel()
        precision, recall, _ = sklearn.metrics.precision_recall_curve(targets_c, preds_c)
        score[c] = sklearn.metrics.auc(recall, precision)
    return np.average(score) * 100.0

def compute_metrics(pred):
  """ validation을 위한 metrics function """
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  probs = pred.predictions

  # calculate accuracy using sklearn's function
  f1 = klue_re_micro_f1(preds, labels)
  auprc = klue_re_auprc(probs, labels)
  acc = accuracy_score(labels, preds) # 리더보드 평가에는 포함되지 않습니다.

  return {
      'micro f1 score': f1,
      'auprc' : auprc,
      'accuracy': acc,
  }

def label_to_num(label):
  num_label = []
  with open('dict_label_to_num.pkl', 'rb') as f:
    dict_label_to_num = pickle.load(f)
  for v in label:
    num_label.append(dict_label_to_num[v])
  
  return num_label

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_NAME='klue/roberta-large'
train_dataset = load_data("../dataset/train/train_85.csv")
dev_dataset = load_data("../dataset/train/eval_85.csv") # validation용 데이터는 따로 만드셔야 합니다.
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_label = label_to_num(train_dataset['label'].values)
if True:
  train_rel_label = train_dataset['rel'].values
  dev_rel_label = dev_dataset['rel'].values
dev_label = label_to_num(dev_dataset['label'].values)

# tokenizing dataset
tokenized_train = tokenized_dataset(train_dataset, tokenizer, MODEL_NAME)
tokenized_dev = tokenized_dataset(dev_dataset, tokenizer, MODEL_NAME)
RE_train_rel_dataset = RE_Dataset(tokenized_train, train_rel_label)
RE_dev_rel_dataset = RE_Dataset(tokenized_dev, dev_rel_label)
RE_train_dataset = RE_Dataset(tokenized_train, train_label,True)
RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label,True)
model_config =  AutoConfig.from_pretrained(MODEL_NAME)
model_config.num_labels = 2
model_rel =  AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
model_config.num_labels = 29
model_main = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
model_main.to(device)
model_rel.to(device)
targs=TrainingArguments(
    output_dir='./results/sep/rel',          # output directory
    save_total_limit=2,              # number of total save model.
    save_steps=500,                 # model saving step.
    num_train_epochs=1,              # total number of training epochs
    learning_rate=1e-5,               # learning_rate
    per_device_train_batch_size=1,  # batch size per device during training
    per_device_eval_batch_size=1,   # batch size for evaluation
    #warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=100,              # log saving step.
    evaluation_strategy='steps', # evaluation strategy to adopt during training
                                  # `no`: No evaluation during training.
                                  # `steps`: Evaluate every `eval_steps`.
                                  # `epoch`: Evaluate every end of epoch.
    eval_steps = 500,            # evaluation step.

    load_best_model_at_end = True 
  )
T_rel=Trainer(
    model=model_rel,                         # the instantiated 🤗 Transformers model to be trained
    args=targs,                  # training arguments, defined above
    train_dataset=RE_train_rel_dataset,         # training dataset
    eval_dataset=RE_dev_rel_dataset,            # evaluation dataset
    compute_metrics=compute_metrics         # define metrics function
  )
dl_rel=T_rel.get_train_dataloader()
for id,dat in enumerate(dl_rel):
    print(dat)
    print(''.join(map(tokenizer.decode,dat['input_ids'])))
    print(model_rel(dat['input_ids'].to(device)))
    break