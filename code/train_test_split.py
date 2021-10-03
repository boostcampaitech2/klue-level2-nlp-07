from sklearn.model_selection import train_test_split
import pandas as pd

data = pd.read_csv('../dataset/train/train.csv')
train, eval = train_test_split(data,train_size=0.85, test_size=0.15, stratify=data['label'])

train.to_csv('../dataset/train/train85.csv')
eval.to_csv('../dataset/train/dev15.csv')