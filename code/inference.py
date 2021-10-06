from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from load_data import *
import pandas as pd
import torch
import torch.nn.functional as F

import pickle as pickle
import numpy as np
import argparse
from tqdm import tqdm
from inference_binary_classifier import RE_binary_Dataset, inference_binary_classifier


def inference(model, model_name, tokenized_sent, device, batch_size):
  """
    test dataset을 DataLoader로 만들어 준 후,
    batch_size로 나눠 model이 예측 합니다.
  """
  dataloader = DataLoader(tokenized_sent, batch_size=batch_size, shuffle=False)
  model.eval()
  output_pred = []
  output_prob = []
  for i, data in enumerate(tqdm(dataloader)):
    # print(data, len(data), type(data))
    with torch.no_grad():
      if 'roberta' not in model_name:
        outputs = model(
            input_ids=data['input_ids'].to(device),
            attention_mask=data['attention_mask'].to(device),
            token_type_ids=data['token_type_ids'].to(device)
            )
      else:
        outputs = model(
            input_ids=data['input_ids'].to(device),
            attention_mask=data['attention_mask'].to(device),
            )
    logits = outputs[0] # 정제되지 않은 output
    # if i == 242:
    #   print(len(logits))
    prob = F.softmax(logits, dim=-1).detach().cpu().numpy() # 확률값(0~1)으로 변환
    logits = logits.detach().cpu().numpy()
    result = np.argmax(logits, axis=-1) # label 번호 얻음
    # print(logits.shape, prob.shape, result.shape) # (32, 29), (32, 29), (32,)

    output_pred.append(result)
    output_prob.append(prob)
  # return array->list, (7765,), (7765, 30) # test data 7765개
  return np.concatenate(output_pred).tolist(), np.concatenate(output_prob, axis=0).tolist()

def num_to_label(label):
  """
    숫자로 되어 있던 class를 원본 문자열 라벨로 변환 합니다.
  """
  origin_label = []
  with open('dict_num_to_label.pkl', 'rb') as f:
    dict_num_to_label = pickle.load(f)
  for v in label:
    origin_label.append(dict_num_to_label[v])
  
  return origin_label

def load_test_dataset(dataset_dir, tokenizer, model_name):
  """
    test dataset을 불러온 후,
    tokenizing 합니다.
  """
  # test_dataset = load_data(dataset_dir)
  dataset = pd.read_csv(dataset_dir)
  test_dataset = preprocessing_dataset(dataset)
  # 여기서 0, 1 나눠서 1인 데이터만 Test_dataset에 저장하기
  test_label = list(map(int,test_dataset['label'].values)) # label column -> list
  # tokenizing dataset, return indexes, tokenized dataset, labels
  tokenized_test = tokenized_dataset(test_dataset, tokenizer, model_name)
  return test_dataset['id'], tokenized_test, test_label

def load_relation_dataset(dataframe, tokenizer, model_name):
  dataset = preprocessing_dataset(dataframe)
  # print(dataset['label'].unique()) # 아마 100일 것이다. 
  test_label = list(map(int, dataset['label'].values))
  tokenized_test = tokenized_dataset(dataset, tokenizer, model_name)
  test_id = list(map(int, dataset['id'].values))
  return test_id, tokenized_test, test_label

def main(args):
  """
    주어진 dataset csv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
  """
  result_dict = inference_binary_classifier(args)
  
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  # load tokenizer
  # Tokenizer_NAME = "klue/bert-base"
  # Tokenizer_NAME = "klue/roberta-large"
  Tokenizer_NAME = args.tokenizer
  BSZ = args.bsz

  tokenizer = AutoTokenizer.from_pretrained(Tokenizer_NAME)

  ## load my model
  model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
  model.parameters
  model.to(device)

  ## load test datset
  test_dataset_dir = "../dataset/test/test_data.csv"
  # test_id, test_dataset, test_label = load_test_dataset(test_dataset_dir, tokenizer, Tokenizer_NAME)
  # Re_test_dataset = RE_Dataset(test_dataset, test_label)
  test_dataframe = pd.read_csv(test_dataset_dir)
  relation_dataframe = test_dataframe[test_dataframe['id'].isin(result_dict.keys())] 
  no_relation_dataframe = test_dataframe.loc[set(test_dataframe.index) - set(relation_dataframe.index)]
  # print(relation_dataframe['label'].unique())
  # print(no_relation_dataframe)
  # print(relation_dataframe['id'])
  relation_id, relation_dataset, relation_label = load_relation_dataset(relation_dataframe, tokenizer, Tokenizer_NAME)
  # print(relation_id) # id column
  # relation_dataframe만 학습된 모델로 inference 진행
  
  Re_test_dataset = RE_binary_Dataset(relation_dataset, relation_label, relation_id)
  # print(Re_test_dataset[-1])

  ## predict answer, inference()에서 prob no relation 추가 
  pred_answer, probs = inference(model, Tokenizer_NAME, Re_test_dataset, device, BSZ) 
  pred_answer = [label+1 for label in pred_answer]
  # print(pred_answer)
  output_prob = []
  single_prob = [0]
  # print(prob[0], len(prob)) # 이중 list, 밖 길이 4018, 안 길이 29
  for prob in probs:
    for class_prob in prob:
      single_prob.append(class_prob)
    output_prob.append(single_prob)
    single_prob = [0]
  # print(len(output_prob), len(output_prob[0]))
  # print('size :', no_relation_dataframe.size) # 22482

  no_relation_pred_label = []
  for _ in range(len(no_relation_dataframe['id'])):
    no_relation_pred_label.append('no_relation')

  no_relation_probs = []
  no_relation_prob = [1]
  for _ in range(len(no_relation_dataframe['id'])):
    for _ in range(29):
      no_relation_prob.append(0)
    no_relation_probs.append(no_relation_prob)
    no_relation_prob = [1]

  # model에서 class 추론
  pred_answer = num_to_label(pred_answer) # 숫자로 된 class를 원래 문자열 라벨로 변환.
  # print(pred_answer)
  ## make csv file with predicted answer
  #########################################################
  # 아래 directory와 columns의 형태는 지켜주시기 바랍니다.
  print(len(no_relation_pred_label), len(no_relation_probs))
  # input()
  output_relation = pd.DataFrame({'id':relation_dataframe['id'], 'pred_label':pred_answer, 'probs':output_prob,})
  output_no_relation = pd.DataFrame({'id':no_relation_dataframe['id'],'pred_label':no_relation_pred_label, 'probs':no_relation_probs})
  # print(output_relation)
  # print(output_no_relation)
  output = pd.concat([output_no_relation, output_relation], ignore_index=True)
  output.sort_values(by=['id'], inplace=True, ignore_index=True)
  # print(output)
  output.to_csv('./prediction/submission_binary.csv', index=False) # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.
  #### 필수!! ##############################################
  print('---- Finish! ----')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  
  # model dir
  parser.add_argument('--model_dir', type=str, default="/opt/ml/code/best_model")
  parser.add_argument('--tokenizer', type=str, default="klue/roberta-large")
  parser.add_argument('--bsz', type=int, default=32)

  parser.add_argument('--binary_save_dir', type=str, default="/opt/ml/code/binary_best_model/")
  parser.add_argument('--binary_tokenizer', type=str, default="klue/roberta-large")
  parser.add_argument('--binary_bsz', type=int, default=32)

  args = parser.parse_args()
  print(args)
  main(args)
  
