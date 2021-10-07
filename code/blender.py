import pandas as pd
import ast
import numpy as np
import pickle
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
output1_data = pd.read_csv('./firstpiece.csv')
output2_data = pd.read_csv('./lastfinal.csv')

output1_probs = []
for string in output1_data['probs']:
    probs = ast.literal_eval(string)
    
    output1_probs.append(probs)

output2_probs = []
for string in output2_data['probs']:
    probs = ast.literal_eval(string)
    
    output2_probs.append(probs)

weight1 = 0.5
weight2 = 0.5
output_probs = []
single_probs = []
for probs1, probs2 in zip(output1_probs, output2_probs):
    for prob1, prob2 in zip(probs1, probs2):
        single_probs.append(prob1 * weight1 + prob2 * weight2)
    output_probs.append(single_probs)
    single_probs = []

output_preds = np.argmax(output_probs, axis=1).tolist()
output_preds = num_to_label(output_preds)
for output_prob in output_probs:
    output_prob = str(output_prob)

output = pd.DataFrame({'id':output1_data['id'],'pred_label':output_preds,'probs':output_probs,})
output.to_csv('./georgeklueni.csv', index=False)

a = output['pred_label']
from collections import Counter

print(Counter(a))
