# model의 input : RE_train_dataset, RE_dataset(tokenized_train, train_label)

# tokenized_train = tokenized_dataset(train_dataset, tokenizer, MODEL_NAME)
# train_label = label_to_num(train_dataset['label'].values)
# label_to_num, train_dataset['label'].values
# 이미 pretrain된 모델을 사용해서 fine tuning을 해야 한다. 

# tokenized_dataset() : 데이터 sentence 특수 문자 전처리해서 cleaned_dataset & entity들을 
# 'subject entity + [SEP] + object entity' 형태로 list에 모두 집어넣어서 concat_entity를
# 만들고, concat_entity와 cleaned_dataset을 tokenizer 함수의 인자로 넣고 리턴값을 리턴하는 함수

# 데이터셋을 만든 후, 데이터셋을 tokenizing한다. 이 작업은 tokenized_dataset()에서 리턴값을
# 받아 오는 형식으로 진행한다. 

# train.csv -> data에 저장 -> 

from load_data import load_data, tokenized_dataset, RE_Dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
import pickle
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def draw_confusion_matrix(true, pred):
    cm = confusion_matrix(true, pred)
    df = pd.DataFrame(cm/np.sum(cm, axis=1)[:, None],
                index=list(range(2)), columns=list(range(2)))
    df = df.fillna(0)  # NaN 값을 0으로 변경
    plt.figure(figsize=(8, 8))
    plt.tight_layout()
    plt.suptitle('Confusion Matrix')
    sns.heatmap(df, annot=True, cmap=sns.color_palette("Blues"))
    plt.xlabel("Predicted Label")
    plt.ylabel("True label")
    plt.savefig(f"./confusion_matrixs/confusion_matrix_binary.png")
    plt.close('all')

def klue_re_micro_f1(preds, labels):
    """KLUE-RE micro f1"""
    # label_list = ['no_relation', 'org:top_members/employees', 'org:members',
    #    'org:product', 'per:title', 'org:alternate_names',
    #    'per:employee_of', 'org:place_of_headquarters', 'per:product',
    #    'org:number_of_employees/members', 'per:children',
    #    'per:place_of_residence', 'per:alternate_names',
    #    'per:other_family', 'per:colleagues', 'per:origin', 'per:siblings',
    #    'per:spouse', 'org:founded', 'org:political/religious_affiliation',
    #    'org:member_of', 'per:parents', 'org:dissolved',
    #    'per:schools_attended', 'per:date_of_death', 'per:date_of_birth',
    #    'per:place_of_birth', 'per:place_of_death', 'org:founded_by',
    #    'per:religion']
    # no_relation_label_idx = label_list.index("no_relation")
    label_list = ['no_relation', 'relation']
    label_indices = list(range(len(label_list)))
    # label_indices.remove(no_relation_label_idx)
    return sklearn.metrics.f1_score(labels, preds, average="micro", labels=label_indices) * 100.0

def klue_re_auprc(probs, labels):
    """KLUE-RE AUPRC"""
    labels = np.eye(2)[labels] # labels의 shape대로 eye 행렬이 채워짐
    # print(labels, labels.shape) # labels.shape = (32470, 30)
    score = np.zeros((2,))
    # print(score.shape) # (30,)
    for c in range(2):
        targets_c = labels.take([c], axis=1).ravel() # take values along axis
        preds_c = probs.take([c], axis=1).ravel() # ravel : contiguous flattened array
        precision, recall, _ = sklearn.metrics.precision_recall_curve(targets_c, preds_c)
        score[c] = sklearn.metrics.auc(recall, precision)
    return np.average(score) * 100.0

def label_to_num(label):
  num_label = []
  with open('dict_label_to_num.pkl', 'rb') as f:
    dict_label_to_num = pickle.load(f)
  for v in label:
    num_label.append(dict_label_to_num[v])
  
  return num_label

def compute_metrics(pred): # pred : 'EvalPrediction' object
  """ validation을 위한 metrics function """
  labels = pred.label_ids # (32470,)
  preds = pred.predictions.argmax(-1) # (32470, 30)
  probs = pred.predictions # (32470, 30)
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

def train_binary_classifier(args):
    DEV_SET = args.binary_dev_set
    EPOCHS = args.binary_epochs
    SAVE_DIR = args.binary_save_dir
    BATCH_SIZE = args.binary_bsz
    LEARNING_RATE = args.binary_learning_rate
    MODEL_NAME = args.binary_model_name

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if DEV_SET is True:
        train_dataset = load_data('/opt/ml/dataset/train/train_0.8.csv')
        dev_dataset = load_data("/opt/ml/dataset/train/eval_0.2.csv")

        train_label = label_to_num(train_dataset['label'].values)
        dev_label = label_to_num(dev_dataset['label'].values)

        binary_train_label = []
        for label in train_label:
            if label != 0:
                binary_train_label.extend([1])
            else:
                binary_train_label.extend([0])

        binary_dev_label = []
        for label in dev_label:
            if label != 0:
                binary_train_label.extend([1])
            else:
                binary_train_label.extend([0])

        # tokenizing dataset
        tokenized_train = tokenized_dataset(train_dataset, tokenizer, MODEL_NAME)
        tokenized_dev = tokenized_dataset(dev_dataset, tokenizer, MODEL_NAME)

        # make dataset for pytorch.
        RE_train_dataset = RE_Dataset(tokenized_train, binary_train_label)
        RE_dev_dataset = RE_Dataset(tokenized_dev, binary_dev_label)
    
    else:
        train_data = load_data('/opt/ml/dataset/train/train.csv') # train.csv를 원하는 형식의 DataFrame으로 변경
        train_label = label_to_num(train_data['label'].values)
        binary_label = []
        for i in train_label:
            if i != 0:
                binary_label.extend([1])
            else:
                binary_label.extend([0])
        
        tokenized_train = tokenized_dataset(train_data, tokenizer, MODEL_NAME)
        RE_train_dataset = RE_Dataset(tokenized_train, binary_label) # 전체 32470개 데이터
        RE_dev_dataset = RE_Dataset(tokenized_train, binary_label)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Binary classfier device -', device)

    model_config = AutoConfig.from_pretrained(MODEL_NAME)
    model_config.num_labels = 2

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
    model.parameters
    model.to(device)

    training_args = TrainingArguments(
        output_dir='./binary_results',          # output directory
        save_total_limit=5,              # number of total save model.
        save_steps=500,                 # model saving step.
        num_train_epochs=EPOCHS,              # total number of training epochs
        learning_rate=LEARNING_RATE,           # learning_rate
        per_device_train_batch_size=BATCH_SIZE,  # batch size per device during training
        per_device_eval_batch_size=BATCH_SIZE,   # batch size for evaluation
    #    warmup_steps=500,                # number of warmup steps for learning rate scheduler
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
    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=RE_train_dataset,         # training dataset
        eval_dataset=RE_dev_dataset,             # evaluation dataset
        compute_metrics=compute_metrics         # define metrics function
    )
    # train model
    print('Train binary classifier')
    trainer.train()
    model.save_pretrained(SAVE_DIR)

# train_binary_classifier()