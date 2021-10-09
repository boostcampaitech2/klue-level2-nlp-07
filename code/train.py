import pickle as pickle
import os
import pandas as pd
import torch
import sklearn
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer
from load_data import *
from train_binary_classifier import train_binary_classifier
import argparse

def draw_confusion_matrix(true, pred, binary, phase):
    num = 29 if binary else 30
    cm = confusion_matrix(true, pred)
    df = pd.DataFrame(cm/np.sum(cm, axis=1)[:, None],
                index=list(range(num)), columns=list(range(num)))    
    df = df.fillna(0)  # NaN 값을 0으로 변경
    plt.figure(figsize=(16, 16))
    plt.tight_layout()
    plt.suptitle('Confusion Matrix')
    sns.heatmap(df, annot=True, cmap=sns.color_palette("Blues"))
    plt.xlabel("Predicted Label")
    plt.ylabel("True label")
    plt.savefig(f"./confusion_matrixs/confusion_matrix_{phase}.png")
    plt.close('all')


def klue_re_micro_f1(preds, labels, binary):
    """KLUE-RE micro f1 (except no_relation)"""
    if binary:
      label_list = ['org:top_members/employees', 'org:members',
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
      label_indices = list(range(len(label_list)))
    
    else:
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

def klue_re_auprc(probs, labels, binary):
    """KLUE-RE AUPRC (with no_relation)"""
    num = 29 if binary else 30
    labels = np.eye(num)[labels]
    score = np.zeros((num,))
    for c in range(num):
        targets_c = labels.take([c], axis=1).ravel()
        preds_c = probs.take([c], axis=1).ravel()
        precision, recall, _ = sklearn.metrics.precision_recall_curve(targets_c, preds_c)
        score[c] = sklearn.metrics.auc(recall, precision)
    return np.average(score) * 100.0

def compute_metrics(pred):
  global phase
  global BINARY

  """ validation을 위한 metrics function """
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  probs = pred.predictions

  # calculate accuracy using sklearn's function
  f1 = klue_re_micro_f1(preds, labels, BINARY)
  auprc = klue_re_auprc(probs, labels, BINARY)
  acc = accuracy_score(labels, preds) # 리더보드 평가에는 포함되지 않습니다.
  
  phase += 1
  draw_confusion_matrix(preds, labels, BINARY, phase)
  return {
      'micro f1 score': f1,
      'auprc' : auprc,
      'accuracy': acc,
  }

def label_to_num(label, binary):
  num_label = []
  with open('dict_label_to_num.pkl', 'rb') as f:
    dict_label_to_num = pickle.load(f)
  for v in label:
    if binary:
          num_label.append(dict_label_to_num[v]-1)
    else:
      num_label.append(dict_label_to_num[v])
  
  return num_label

def train(args):
  # load model and tokenizer
  MODEL_NAME = args.model_name
  BINARY = args.binary
  EPOCHS = args.epochs
  BATCH_SIZE = args.bsz
  SAVE_DIR = args.save_dir
  DEV_SET = False if args.dev_set.lower() in ['false', 'f', 'no', 'none'] else True
  NER_MARKER = False if args.ner_marker.lower() in ['false', 'f', 'no', 'none'] else True
  PREPROCESSED = False if args.preprocessed.lower() in ['false', 'f', 'no', 'none'] else True
  TRAIN_SET = "../dataset/train/" + args.train_set
  LEARNING_RATE = args.lr
  SAVE_STEPS = args.save_steps
  phase = 0

  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  
  # load dataset

  if DEV_SET is True:
    train_dataset = load_data(TRAIN_SET, PREPROCESSED, NER_MARKER, BINARY)
    dev_dataset = load_data("../dataset/train/eval_0.8.csv", PREPROCESSED, NER_MARKER, BINARY) # validation용 데이터는 따로 만드셔야 합니다.

    train_label = label_to_num(train_dataset['label'].values, BINARY)
    dev_label = label_to_num(dev_dataset['label'].values, BINARY)

    # tokenizing dataset
    tokenized_train = tokenized_dataset(train_dataset, tokenizer, MODEL_NAME, NER_MARKER)
    tokenized_dev = tokenized_dataset(dev_dataset, tokenizer, MODEL_NAME, NER_MARKER)
    
    # make dataset for pytorch.
    RE_train_dataset = RE_Dataset(tokenized_train, train_label)
    RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)

  else:
    train_dataset = load_data("../dataset/train/train.csv", PREPROCESSED, NER_MARKER, BINARY)
    train_label = label_to_num(train_dataset['label'].values, BINARY)
    tokenized_train = tokenized_dataset(train_dataset, tokenizer, MODEL_NAME, NER_MARKER)
    RE_train_dataset = RE_Dataset(tokenized_train, train_label)
    RE_dev_dataset = RE_Dataset(tokenized_train, train_label)

  print(device)
  # setting model hyperparameter
  model_config = AutoConfig.from_pretrained(MODEL_NAME)
  model_config.num_labels = 29 if BINARY else 30

  model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
  print(model.config)
  model.parameters
  model.to(device)
    
  # 사용한 option 외에도 다양한 option들이 있습니다.
  # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
  training_args = TrainingArguments(

    output_dir='./results',                    # output directory
    save_total_limit=2,                        # number of total save model.
    save_steps=SAVE_STEPS,                     # model saving step.
    num_train_epochs=EPOCHS,                   # total number of training epochs
    learning_rate=LEARNING_RATE,               # learning_rate
    per_device_train_batch_size=BATCH_SIZE,    # batch size per device during training
    per_device_eval_batch_size=BATCH_SIZE,     # batch size for evaluation
    # warmup_steps=500,                        # number of warmup steps for learning rate scheduler
    weight_decay=0.01,                         # strength of weight decay
    logging_dir='./logs',                      # directory for storing logs
    logging_steps=100,                         # log saving step.
    evaluation_strategy='steps',               # evaluation strategy to adopt during training
                                               # `no`: No evaluation during training.
                                               # `steps`: Evaluate every `eval_steps`.
                                               # `epoch`: Evaluate every end of epoch.
    eval_steps = SAVE_STEPS,                   # evaluation step.

    load_best_model_at_end = True,
    metric_for_best_model = "micro f1 score",
    greater_is_better = True,

  )
  trainer = Trainer(
    model=model,                               # the instantiated 🤗 Transformers model to be trained
    args=training_args,                        # training arguments, defined above
    train_dataset=RE_train_dataset,            # training dataset
    eval_dataset=RE_dev_dataset,               # evaluation dataset
    compute_metrics=compute_metrics            # define metrics function
  )

  # train model
  
  trainer.train()
  save_directory = './best_model/' + SAVE_DIR
  model.save_pretrained(save_directory)

def main(args):
  if args.binary:
    train_binary_classifier(args)
  train(args)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--model_name', type=str, default="klue/roberta-large")
  parser.add_argument('--bsz', type=int, default=32)
  parser.add_argument('--epochs', type=int, default=2)
  parser.add_argument('--save_dir', type=str, default="")
  parser.add_argument('--dev_set', type=str, default="True")
  parser.add_argument('--preprocessed', type=str, default="False")
  parser.add_argument('--train_set', type=str, default="train.csv")
  parser.add_argument('--lr', type=float, default=1e-5)
  parser.add_argument('--save_steps', type=int, default=500)
  parser.add_argument('--ner_marker', type=str, default="False")

  parser.add_argument('--binary', type=bool, default=False)
  parser.add_argument('--binary_model_name', type=str, default="klue/roberta-large")
  parser.add_argument('--binary_bsz', type=int, default=32)
  parser.add_argument('--binary_epochs', type=int, default=1)
  parser.add_argument('--binary_learning_rate', type=float, default=3e-5)
  parser.add_argument('--binary_save_dir', type=str, default="/opt/ml/code/binary_best_model/")
  parser.add_argument('--binary_dev_set', type=str, default="True")

  args = parser.parse_args()
  
  print(args)
  main(args)
