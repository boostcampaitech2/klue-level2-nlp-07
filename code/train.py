from operator import mod
import pickle as pickle
import os
import pandas as pd
import torch
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer
from train_binary_classifier import train_binary_classifier
from load_data import *
import argparse
import torch.optim

def draw_confusion_matrix(true, pred):
    cm = confusion_matrix(true, pred)
    df = pd.DataFrame(cm/np.sum(cm, axis=1)[:, None],
                index=list(range(29)), columns=list(range(29)))
    df = df.fillna(0)  # NaN 값을 0으로 변경
    plt.figure(figsize=(16, 16))
    plt.tight_layout()
    plt.suptitle('Confusion Matrix')
    sns.heatmap(df, annot=True, cmap=sns.color_palette("Blues"))
    plt.xlabel("Predicted Label")
    plt.ylabel("True label")
    plt.savefig(f"/opt/ml/code/confusion_matrixs/confusion_matrix.png")
    plt.close('all')

def klue_re_micro_f1(preds, labels):
    """KLUE-RE micro f1 (except no_relation)"""
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
    # no_relation_label_idx = label_list.index("no_relation")
    label_indices = list(range(len(label_list)))
    # label_indices.remove(no_relation_label_idx)
    return sklearn.metrics.f1_score(labels, preds, average="micro", labels=label_indices) * 100.0

def klue_re_auprc(probs, labels):
    """KLUE-RE AUPRC (with no_relation)"""
    labels = np.eye(29)[labels] # labels의 shape대로 eye 행렬이 채워짐
    # print(labels, labels.shape) # labels.shape = (32470, 30)
    score = np.zeros((29,))
    # print(score.shape) # (30,)
    for c in range(29):
        targets_c = labels.take([c], axis=1).ravel() # take values along axis
        preds_c = probs.take([c], axis=1).ravel() # ravel : contiguous flattened array
        precision, recall, _ = sklearn.metrics.precision_recall_curve(targets_c, preds_c)
        score[c] = sklearn.metrics.auc(recall, precision)
    return np.average(score) * 100.0

def compute_metrics(pred): # pred : 'EvalPrediction' object
  """ validation을 위한 metrics function """
  labels = pred.label_ids # (32470,)
  preds = pred.predictions.argmax(-1) # (32470, 30)
  probs = pred.predictions # (32470, 30)
  print('probs :', probs)
  # calculate accuracy using sklearn's function
  f1 = klue_re_micro_f1(preds, labels)
  auprc = klue_re_auprc(probs, labels)
  acc = accuracy_score(labels, preds) # 리더보드 평가에는 포함되지 않습니다.
  draw_confusion_matrix(labels, preds)

  return {
      'micro f1 score': f1,
      'auprc' : auprc,
      'accuracy': acc,
  }

def label_to_num(label):
  num_label = []
  with open('/opt/ml/code/dict_label_to_num.pkl', 'rb') as f:
    dict_label_to_num = pickle.load(f)
  for v in label: 
    num_label.append(dict_label_to_num[v]-1)

  return num_label

# roberta-large에서는 29개 클래스 데이터만 학습시키기! 
def train(args):
  # load model and tokenizer
  # MODEL_NAME = "bert-base-uncased"
  # MODEL_NAME = "klue/bert-base"
  # MODEL_NAME = "klue/roberta-large"
  MODEL_NAME = args.model_name
  EPOCHS = args.epochs
  LEARNING_RATE = args.learning_rate
  BATCH_SIZE = args.bsz
  SAVE_DIR = args.save_dir
  DEV_SET = False if args.dev_set.lower() in ['false', 'f', 'no', 'none'] else True

  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

  # load dataset
  if DEV_SET is True:
    train_dataset = load_data("/opt/ml/dataset/train/train_0.8.csv")
    dev_dataset = load_data("/opt/ml/dataset/train/eval_0.2.csv") # validation용 데이터는 따로 만드셔야 합니다.

    train_label = label_to_num(train_dataset['label'].values)
    dev_label = label_to_num(dev_dataset['label'].values)

    # tokenizing dataset
    tokenized_train = tokenized_dataset(train_dataset, tokenizer, MODEL_NAME)
    tokenized_dev = tokenized_dataset(dev_dataset, tokenizer, MODEL_NAME)

    # make dataset for pytorch.
    RE_train_dataset = RE_Dataset(tokenized_train, train_label)
    RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)
  
  else:
    train_dataset = load_data("/opt/ml/dataset/train/train.csv")
    # print(train_dataset['label'].unique())    
    train_label = label_to_num(train_dataset['label'].values)
    # print(len(train_label))

    tokenized_train = tokenized_dataset(train_dataset, tokenizer, MODEL_NAME)
    RE_train_dataset = RE_Dataset(tokenized_train, train_label)
    RE_dev_dataset = RE_Dataset(tokenized_train, train_label)

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  
  print(device)
  # setting model hyperparameter
  model_config = AutoConfig.from_pretrained(MODEL_NAME)
  model_config.num_labels = 29
  # print("=====================")
  # model_config.id2label = {'1':'LABEL_1', '2':'LABEL_2', '3':'LABEL_3', '4':'LABEL_4', \
  #   '5':'LABEL_5', '6':'LABEL_6', '7':'LABEL_7', '8':'LABEL_8', '9':'LABEL_9', \
  #     '10':'LABEL_10', '11':'LABEL_11', '12':'LABEL_12', '13':'LABEL_13', '14':'LABEL_14', \
  #       '15':'LABEL_15', '16':'LABEL_16', '17':'LABEL_17', '18':'LABEL_18', '19':'LABEL_19', \
  #         '20':'LABEL_20', '21':'LABEL_21', '22':'LABEL_22', '23':'LABEL_23', \
  #           '24':'LABEL_24', '25':'LABEL_25', '26':'LABEL_26', '27': 'LABEL_27', \
  #             '28':'LABEL_28', '29':'LABEL_29'}

  # model_config.label2id = {value:key for key, value in model_config.id2label.items()}

  # model_config.
  print(model_config)

  model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
  # print(model.config)
  model.parameters
  model.to(device)
  
  # 사용한 option 외에도 다양한 option들이 있습니다.
  # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
  training_args = TrainingArguments(
    output_dir='/opt/ml/code/results',          # output directory
    save_total_limit=5,              # number of total save model.
    save_steps=500,                 # model saving step.
    num_train_epochs=EPOCHS,              # total number of training epochs
    learning_rate=LEARNING_RATE,           # learning_rate
    per_device_train_batch_size=BATCH_SIZE,  # batch size per device during training
    per_device_eval_batch_size=BATCH_SIZE,   # batch size for evaluation
#    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='/opt/ml/code/logs',            # directory for storing logs
    logging_steps=100,              # log saving step.
    evaluation_strategy='steps', # evaluation strategy to adopt during training
                                # `no`: No evaluation during training.
                                # `steps`: Evaluate every `eval_steps`.
                                # `epoch`: Evaluate every end of epoch.
    eval_steps = 500,            # evaluation step.
    load_best_model_at_end = True,
    metric_for_best_model='micro f1 score',
    greater_is_better=True,
  )
  
  trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=RE_train_dataset,         # training dataset
    eval_dataset=RE_dev_dataset,             # evaluation dataset
    compute_metrics=compute_metrics         # define metrics function
  )
  
  # train model
  print('Train roberta-large')

  trainer.train() 
  # return global_step(int), training_loss(float), metrics(Dict[str,float]) 
  model.save_pretrained(SAVE_DIR)
  
def main(args):
  # wandb.init(project="nlp_klue")
  train_binary_classifier(args)
  train(args)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--model_name', type=str, default="klue/roberta-large")
  parser.add_argument('--bsz', type=int, default=32)
  parser.add_argument('--epochs', type=int, default=3)
  parser.add_argument('--learning_rate', type=float, default=3e-5)
  parser.add_argument('--save_dir', type=str, default="/opt/ml/code/best_model/")
  parser.add_argument('--dev_set', type=str, default="True")

  parser.add_argument('--binary_model_name', type=str, default="klue/roberta-large")
  parser.add_argument('--binary_bsz', type=int, default=32)
  parser.add_argument('--binary_epochs', type=int, default=1)
  parser.add_argument('--binary_learning_rate', type=float, default=3e-5)
  parser.add_argument('--binary_save_dir', type=str, default="/opt/ml/code/binary_best_model/")
  parser.add_argument('--binary_dev_set', type=str, default="True")
  
  args = parser.parse_args()

  print(args)
  main(args)
