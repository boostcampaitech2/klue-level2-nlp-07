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
  global phase
  
  """ validation을 위한 metrics function """
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  probs = pred.predictions

  # calculate accuracy using sklearn's function
  f1 = klue_re_micro_f1(preds, labels)
  auprc = klue_re_auprc(probs, labels)
  acc = accuracy_score(labels, preds) # 리더보드 평가에는 포함되지 않습니다.
  
  phase +=1
  draw_confusion_matrix(preds, labels)
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


def train(args):
  # load model and tokenizer
  MODEL_NAME = args.model_name
  EPOCHS = args.epochs
  BATCH_SIZE = args.bsz
  SAVE_DIR = args.save_dir
  DEV_SET = False if args.dev_set.lower() in ['false', 'f', 'no', 'none'] else True
  NER_TAG = False if args.ner_tag.lower() in ['false', 'f', 'no', 'none'] else True


  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  
  # load dataset

  if DEV_SET is True:
    train_dataset = load_data("../dataset/train/train_0.8.csv", NER_TAG)
    dev_dataset = load_data("../dataset/train/eval_0.8.csv", NER_TAG) # validation용 데이터는 따로 만드셔야 합니다.

    train_label = label_to_num(train_dataset['label'].values)
    dev_label = label_to_num(dev_dataset['label'].values)

    # tokenizing dataset
    tokenized_train = tokenized_dataset(train_dataset, tokenizer, MODEL_NAME, NER_TAG)
    tokenized_dev = tokenized_dataset(dev_dataset, tokenizer, MODEL_NAME, NER_TAG)
    
    # make dataset for pytorch.
    RE_train_dataset = RE_Dataset(tokenized_train, train_label)
    RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)

  else:
    train_dataset = load_data("../dataset/train/train.csv", NER_TAG)
    train_label = label_to_num(train_dataset['label'].values)
    tokenized_train = tokenized_dataset(train_dataset, tokenizer, MODEL_NAME, NER_TAG)
    RE_train_dataset = RE_Dataset(tokenized_train, train_label)
    RE_dev_dataset = RE_Dataset(tokenized_train, train_label)



  print(device)
  # setting model hyperparameter
  model_config =  AutoConfig.from_pretrained(MODEL_NAME)
  model_config.num_labels = 30

  model =  AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
  print(model.config)
  model.parameters
  model.to(device)
    
  # 사용한 option 외에도 다양한 option들이 있습니다.
  # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
  training_args = TrainingArguments(
    output_dir='./results',          # output directory

    save_total_limit=2,              # number of total save model.
    save_steps=500,                 # model saving step.
    num_train_epochs=EPOCHS,              # total number of training epochs
    learning_rate=5e-5,               # learning_rate
    per_device_train_batch_size=BATCH_SIZE,  # batch size per device during training
    per_device_eval_batch_size=BATCH_SIZE,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler

    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=250,              # log saving step.
    evaluation_strategy='steps', # evaluation strategy to adopt during training
                                  # `no`: No evaluation during training.
                                  # `steps`: Evaluate every `eval_steps`.
                                  # `epoch`: Evaluate every end of epoch.
    eval_steps = 500,            # evaluation step.


    load_best_model_at_end = True 

  )
  trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=RE_train_dataset,         # training dataset

    eval_dataset=RE_dev_dataset,            # evaluation dataset

    compute_metrics=compute_metrics         # define metrics function
  )

  # train model
  
  trainer.train()
  save_directory = './best_model/' + SAVE_DIR
  model.save_pretrained(save_directory)

def main(args):
  train(args)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--model_name', type=str, default="klue/roberta-large")
  parser.add_argument('--bsz', type=int, default=32)
  parser.add_argument('--epochs', type=int, default=5)
  parser.add_argument('--save_dir', type=str, default="")
  parser.add_argument('--dev_set', type=str, default="False")
  parser.add_argument('--ner_tag', type=str, default="False")
  args = parser.parse_args()
  
  print(args)
  main(args)