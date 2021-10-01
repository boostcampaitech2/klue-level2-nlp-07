import pickle as pickle
import os
import pandas as pd
import torch
import re
from collections import OrderedDict
import random
from koeda import AEDA
from tqdm import tqdm
import time
import gc
aeda = AEDA(
    morpheme_analyzer="Okt", punc_ratio=0.3, punctuations=[".", ",", "!", "?", ";", ":"]
)

class RE_Dataset(torch.utils.data.Dataset):
  """ Dataset 구성을 위한 class."""
  def __init__(self, pair_dataset, labels):
    
    self.pair_dataset = pair_dataset
    self.labels = labels
    

  def __getitem__(self, idx):
    item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
    item['labels'] = torch.tensor(self.labels[idx])
    return item

  def __len__(self):
    return len(self.labels)

def preprocessing_dataset(dataset):
  """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
  subject_entity = []
  object_entity = []
  
  for i,j in zip(dataset['subject_entity'], dataset['object_entity']):
    i = i[1:-1].split(',')[0].split(':')[1]
    j = j[1:-1].split(',')[0].split(':')[1]
    
    subject_entity.append(i)
    object_entity.append(j)
    
  out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':dataset['sentence'],'subject_entity':subject_entity,'object_entity':object_entity,'label':dataset['label'],})
  return out_dataset

def load_data(dataset_dir):
  """ csv 파일을 경로에 맞게 불러 옵니다. """
  pd_dataset = pd.read_csv(dataset_dir)
  #dataset = data_pruning(pd_dataset)
  dataset = preprocessing_dataset(pd_dataset)
  
  return dataset

# def data_pruning(dataset,switch=True):
#     from tqdm import tqdm
#     if switch == True:
#         print("================================================================================")
#         print("The length of dataset before pruning is : ",len(dataset))
#         dataset = pd.DataFrame(dataset)
#         data0 = dataset.loc[dataset['label'] == 'no_relation']
#         # data1 = dataset.loc[dataset['label'] == 'org:top_members/employees']
#         # data6 = dataset.loc[dataset['label'] == 'per:employee_of']
#         others = dataset.loc[dataset['label'] != 'no_relation']
#         #& dataset['label'] != 'org:top_members/employees' & dataset['label'] != 'per:employee_of']
        
#         for id in tqdm(range(len(data0)),desc="Pruning....."):
#             prob = random.randint(0,10)
            
#             if prob >= 2:
#                data0 = data0.drop(data0[data0.id == id].index)
#         dataset = pd.concat([data0,others])
        

#         return dataset

#     elif switch == False:
#         return dataset
    
         
def clean_punc(text):
    punct_mapping = {'ʿ': '', 'ū': 'u', 'è': 'e', 'ȳ': 'y', 'ồ': 'o', 'ề': 'e', 'â': 'a', 'æ': 'ae', 'ő': 'o', 'ῶ': 'ω', '𑀕': 'Λ', 'ß': 'β', 'ヶ': 'ケ', '‘': "'", '₹': 'e', '´': "'", '°': '', '€': 'e', '™': 'tm', '√': ' sqrt ', '×': 'x', '²': '2', '—': '-', '–': '-', '’': "'", '_': '-', '`': "'", '“': '"', '”': '"', '£': 'e', '∞': 'infinity', '÷': '/', '•': '.', 'à': 'a', '−': '-', 'Ῥ': 'Ρ', 'ầ': 'a', '́': "'", 'ò': 'o', 'Ö': 'O', 'Š': 'S', 'ệ': 'e', 'Ś': 'S', 'ē': 'e', 'ä': 'a', 'ć': 'c', 'ë': 'e', 'å': 'a', 'Ǧ': 'G', 'ạ': 'a', 'ņ': 'n', 'İ': 'I', 'ğ': 'g', 'ê': 'e', 'Č': 'C', 'ã': 'a', 'ḥ': 'h', 'ả': 'a', 'ễ': 'e', '％': '%', 'ợ': 'o', 'Ú': 'U', 'ư': 'u', 'Ž': 'Z', 'ú': 'u', 'É': 'E', 'Ó': 'O', 'ü': 'u', 'é': 'e', 'ā': 'a', 'š': 's', '𑀥': 'D', 'í': 'i', 'û': 'u', 'ý': 'y', 'ī': 'i', 'ï': 'i', 'ộ': 'o', 'ì': 'i', 'ọ': 'o', 'ş': 's', 'ó': 'o', 'ñ': 'n', 'ậ': 'a', 'Â': 'A', 'ù': 'u', 'ô': 'o', 'ố': 'o', 'Á': 'A', 'ö': 'o', 'ơ': 'o', 'ç': 'c', 'ˈ': "'", 'µ': 'μ', '／': '/', '（': '(', 'ｍ': 'm', '˘': ' ', '𑀫': 'ma', '？': '?', 'ł': 'l', 'Đ': 'D', '：': ':', '･': ',', 'Ç': 'C', 'ı': 'i', '，': ',', '𥘺': '祉', '·': ',', '＇': "'", ' ': ' ', '）': ')', '１': '1', 'ø': 'o', '～': '~', '³': '3', '(˘ ³˘)': '', '˹': '"', '｢': '"', '｣': '"', '«': '<<', '˼': '"', '»': '>>', '®': 'R'}

    for p in punct_mapping:
        text=re.sub(p, punct_mapping[p],text)
    return text

def tokenized_dataset_for_train(dataset, tokenizer):
    copied_dataset = list(dataset['sentence'])
    now = time.localtime()
    cleaned_dataset = []
    cleaned_dataset2 = []
    file = 'aeda_train_data.pkl'
    if os.path.exists(file):
        with open(file, "rb") as f:
            cleaned_dataset2 = pickle.load(f)

    else:
        for sentence in tqdm(copied_dataset,desc="lighter preprocessing, won't take long..."):

            sentence = clean_punc(sentence)
            sentence = re.sub(',','',sentence)
            sentence = re.sub('[^0-9a-zA-Z가-힣一-龥()]',' ',sentence)
            sentence = re.sub('\s+',' ',sentence)
            cleaned_dataset.append(sentence)
            


        for sentence in tqdm(cleaned_dataset,desc="Augmentation on progress..."):
            sentence_a = sentence[:len(sentence)//2]
            sentence_b = sentence[len(sentence)//2:]
            augmented_a = aeda(sentence_a)
            augmented_b = aeda(sentence_b)
            augmented = augmented_a+augmented_b
            cleaned_dataset2.append(augmented)
            
            
            
            
        with open(file,"wb") as f:
            pickle.dump(cleaned_dataset2,f)
        
        

    """ tokenizer에 따라 sentence를 tokenizing 합니다."""
    concat_entity = []
    for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
      temp = ''
      temp = e01 + '[SEP]' + e02
      concat_entity.append(temp)

    tokenized_sentences = tokenizer(
        concat_entity,
        cleaned_dataset2, 
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
        add_special_tokens=True,
        return_token_type_ids=False,
        )
    return tokenized_sentences


def tokenized_dataset_for_dev(dataset, tokenizer):
    copied_dataset = list(dataset['sentence'])
    
    cleaned_dataset = []
    cleaned_dataset2 = ['']*len(copied_dataset)
    file = 'aeda_dev_data.pkl'
    if os.path.exists(file):
        with open(file, "rb") as f:
            cleaned_dataset2 = pickle.load(f)

    else:
        for sentence in tqdm(copied_dataset,desc="lighter preprocessing, won't take long..."):

            sentence = clean_punc(sentence)
            sentence = re.sub(',','',sentence)
            sentence = re.sub('[^0-9a-zA-Z가-힣一-龥()]',' ',sentence)
            sentence = re.sub('\s+',' ',sentence)
            cleaned_dataset.append(sentence)


        for sentence in tqdm(range(len(cleaned_dataset)), desc = "Real Preprocessing..."):
            augmented = aeda(cleaned_dataset[sentence])
            cleaned_dataset2[sentence] = augmented
        with open(file,"wb") as f:
            pickle.dump(cleaned_dataset2,f)
        
        

    """ tokenizer에 따라 sentence를 tokenizing 합니다."""
    concat_entity = []
    for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
      temp = ''
      temp = e01 + '[SEP]' + e02
      concat_entity.append(temp)

    tokenized_sentences = tokenizer(
        concat_entity,
        cleaned_dataset2, 
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
        add_special_tokens=True,
        return_token_type_ids=False,
        )
    return tokenized_sentences

def tokenized_dataset_for_test(dataset, tokenizer):
    copied_dataset = list(dataset['sentence'])
    
    cleaned_dataset = []
    cleaned_dataset2 = ['']*len(copied_dataset)
    file = 'aeda_test_data.pkl'
    if os.path.exists(file):
        with open(file, "rb") as f:
            cleaned_dataset2 = pickle.load(f)

    else:
        for sentence in tqdm(copied_dataset,desc="lighter preprocessing, won't take long..."):

            sentence = clean_punc(sentence)
            sentence = re.sub(',','',sentence)
            sentence = re.sub('[^0-9a-zA-Z가-힣一-龥()]',' ',sentence)
            sentence = re.sub('\s+',' ',sentence)
            cleaned_dataset.append(sentence)


        
        with open(file,"wb") as f:
            pickle.dump(cleaned_dataset,f)
        
        

    """ tokenizer에 따라 sentence를 tokenizing 합니다."""
    concat_entity = []
    for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
      temp = ''
      temp = e01 + '[SEP]' + e02
      concat_entity.append(temp)

    tokenized_sentences = tokenizer(
        concat_entity,
        cleaned_dataset, 
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
        add_special_tokens=True,
        return_token_type_ids=False,
        )
    
    return tokenized_sentences