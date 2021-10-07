from sklearn.model_selection import train_test_split
import pandas as pd
file_name = "train_combined_point9"
file_name2 = "eval_combined_point1"

data=pd.read_csv('/opt/ml/klue-level2-nlp-07/dataset/train/combined3.csv')
train,eval=train_test_split(data,train_size=0.9,test_size=0.1,stratify=data['label'])
train.to_csv(f'../dataset/train/{file_name}.csv')
eval.to_csv(f'../dataset/train/{file_name2}.csv')