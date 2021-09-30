from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
data=pd.read_csv('../dataset/train/train.csv')
train,eval=train_test_split(data,train_size=0.8,test_size=0.2,stratify=data['label'])
#train.to_csv('../dataset/train/{file_name}.csv')
#eval.to_csv('../dataset/train/{file_name}.csv')
with open('dict_num_to_label.pkl','rb') as f:
    dict_label_to_num=pickle.load(f)
data.loc[data['label'].isin(['per:place_of_residence']),['sentence','subject_entity','object_entity']].to_csv('../dataset/class.csv')
