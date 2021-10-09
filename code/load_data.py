import pickle as pickle
import os
import pandas as pd
import torch
import re
from collections import OrderedDict
import random

def drop_no_relation_data(dataset):
  indexes_no_relation = dataset[dataset['label'] == 'no_relation'].index
  dataset.drop(indexes_no_relation, inplace=True)
  return dataset

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

    i = i.split("'word': ")[1].split(", 'start_idx'")[0]
    j = j.split("'word': ")[1].split(", 'start_idx'")[0]

    subject_entity.append(i)
    object_entity.append(j)
  
  output_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':dataset['sentence'],'subject_entity':subject_entity,'object_entity':object_entity,'label':dataset['label'],})
  output_dataset['sentence'] = output_dataset['sentence'].apply(lambda x: re.sub(r'(\d+),(\d+)', r'\1\2', x))
  output_dataset['subject_entity'] = output_dataset['subject_entity'].apply(lambda x: re.sub(r'(\d+),(\d+)', r'\1\2', x))
  output_dataset['object_entity'] = output_dataset['object_entity'].apply(lambda x: re.sub(r'(\d+),(\d+)', r'\1\2', x))
  return output_dataset


def preprocessing_dataset_ner(dataset):
  """
    처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다.
    NER tagging이 된 ner_tagged_sent column이 포함된 DataFrame을 리턴합니다.
  """
  subject_entity = []
  subject_ner = []
  object_entity = []
  object_ner = []
  
  for i,j in zip(dataset['subject_entity'], dataset['object_entity']):
    i, j = eval(i), eval(j)  # convert str to dict
    sbj_ntt, sbj_ner = i["word"], i["type"]
    obj_ntt, obj_ner = j["word"], j["type"]

    subject_entity.append(sbj_ntt)
    subject_ner.append(sbj_ner)

    object_entity.append(obj_ntt)
    object_ner.append(obj_ner)

  out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':dataset['sentence'],'subject_entity':subject_entity,
                              'subject_ner':subject_ner,'object_entity':object_entity,'object_ner':object_ner,
                              'label':dataset['label']})

  out_dataset['ner_
              ged_sent'] = [add_ner_marker(row) for idx, row in out_dataset.iterrows()]

  out_dataset['sentence'] = out_dataset['sentence'].apply(lambda x: re.sub(r'(\d+),(\d+)', r'\1\2', x))
  out_dataset['subject_entity'] = out_dataset['subject_entity'].apply(lambda x: re.sub(r'(\d+),(\d+)', r'\1\2', x))
  out_dataset['object_entity'] = out_dataset['object_entity'].apply(lambda x: re.sub(r'(\d+),(\d+)', r'\1\2', x))

  return out_dataset

def add_ner_marker(row):
    """ Entity의 앞뒤에 해당 Entity 의 NER 정보를 marking 해줍니다.
        subject와 object entity의 NER type은 각각 ^, *으로 감쌉니다.
        전체 Entity는 @로 감싸서 전처리 합니다.
        ex) @^ORG^광주여대@(총장 @*PER*이선재@) 평생교육원은 수료식을 실시했다고 밝혔다.
    """
    sent = row.sentence

    sent = sent.replace(row.subject_entity, f'@^{row.subject_ner}^{row.subject_entity}@')
    sent = sent.replace(row.object_entity, f'@*{row.object_ner}*{row.object_entity}@')
    
    return sent

def load_data(dataset_dir, preprocessed=False, NER_marker=False, Binary=False):
  """ csv 파일을 경로에 맞게 불러 옵니다. """
  pd_dataset = pd.read_csv(dataset_dir)
 
  if Binary:
    pd_dataset = drop_no_relation_data(pd_dataset)
  if preprocessed:
    return pd_dataset
  elif NER_marker:
    dataset = preprocessing_dataset_ner(pd_dataset)
  else:
    dataset = preprocessing_dataset(pd_dataset)  
  return dataset



def clean_sentence(sentence):

    punct_mapping = {'ū': 'u', 'è': 'e', 'ȳ': 'y', 'ồ': 'o', 'ề': 'e', 'â': 'a', 'æ': 'ae', 'ő': 'o', 'α': 'alpha', 'ß': 'ss', 'β': 'beta', 'ヶ': 'ケ', '₹': 'e', '°': '', '€': 'euro', '™': 'tm', '√': ' sqrt ', '–': '-', '£': 'e', '∞': 'infinity', '÷': '/', 'à': 'a', '−': '-', 'Ῥ': 'Ρ', 'ầ': 'a', '́': "'", 'ò': 'o', 'Ö': 'O', 'Š': 'S', 'ệ': 'e', 'Ś': 'S', 'ē': 'e', 'ä': 'a', 'ć': 'c', 'ë': 'e', 'å': 'a', 'Ǧ': 'G', 'ạ': 'a', 'ņ': 'n', 'İ': 'I', 'ğ': 'g', 'ê': 'e', 'Č': 'C', 'ã': 'a', 'ḥ': 'h', 'ả': 'a', 'ễ': 'e', 'ợ': 'o', 'Ú': 'U', 'ư': 'u', 'Ž': 'Z', 'ú': 'u', 'É': 'E', 'Ó': 'O', 'ü': 'u', 'ā': 'a', 'š': 's', '𑀥': 'D', 'í': 'i', 'û': 'u', 'ý': 'y', 'ī': 'i', 'ï': 'i', 'ộ': 'o', 'ì': 'i', 'ọ': 'o', 'ş': 's', 'ó': 'o', 'ñ': 'n', 'ậ': 'a', 'Â': 'A', 'ù': 'u', 'ô': 'o', 'ố': 'o', 'Á': 'A', 'ö': 'o', 'ơ': 'o', 'ç': 'c', 'ˈ': "'", 'µ': 'μ', '／': '/', '（': '(', '˘': ' ', '？': '?', 'ł': 'l', 'Đ': 'D', '･': ',', 'Ç': 'C', 'ı': 'i', '𥘺': '祉', '＇': "'", ' ': ' ', '）': ')', '１': '1', 'ø': 'o', '～': '~', '³': '3', '(˘ ³˘)': '', '˹': '<', '«': '<', '˼': '>', '»': '>'}
    sub_pat='‘｢♀▼女，◆㎜㈜+?😍●㎝『》不〔Ⅰ!〉´♡️「②＆\'=∙｣㎖㎡金•▲ｔ☆♥▷‧․ᆞ①㎏℃⑤』%Ⅱ│○】×─✔”〕_,&｜²☞→↑【#};◇━]理＝⠀😂👉⊙`(💕👍△％《▶③é:|＜；*⑦/〈😭※~―@"—≫✨[㎎⑥㏊∼ㆍ＞－^❤ℓ：>🤣★ㅤ李<ｍ·Ⅲ＋.◈㎢■…$≪㎞‥□」🏻㎾＂④・{😆“㎥’'
    for p in punct_mapping:
        sentence=re.sub(p, punct_mapping[p],sentence)
    sentence = re.sub(f'[^- ㄱ-ㅎㅏ-ㅣ가-힣0-9a-zA-Zぁ-ゔァ-ヴー々〆〤一-龥(){re.escape(sub_pat)}]',' ',sentence)
    sentence = re.sub('\s+',' ',sentence)
    sentence = re.sub('\([, ]*\)','',sentence)
    return sentence



def tokenized_dataset(dataset, tokenizer, model, NER_marker=False):

    if NER_marker:
      cleaned_dataset = [clean_sentence(sent) for sent in dataset.ner_tagged_sent]
    else:
      cleaned_dataset = [clean_sentence(sent) for sent in dataset.sentence]

        

    """ tokenizer에 따라 sentence를 tokenizing 합니다."""
    concat_entity = []
    for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
      temp = ''
      temp = e01 + '[SEP]' + e02
      concat_entity.append(temp)

    tokenized_sentences = tokenizer(

          concat_entity,
          cleaned_dataset, #여기를 수정해서 돌려주시면 됩니다. cleaned dataset으로.
          return_tensors="pt",
          padding=True,
          truncation=True,
          max_length=256,  # default: 256
          add_special_tokens=True,
          return_token_type_ids=False if 'roberta' in model else True,
          )

    return tokenized_sentences
