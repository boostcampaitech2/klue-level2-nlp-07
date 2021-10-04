import pickle as pickle
import os
import pandas as pd
import torch
import re
from collections import OrderedDict
import random



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
  subject_start, subject_end = [], []
  object_entity = []
  object_ner = []
  object_start, object_end = [], []
  
  for i,j in zip(dataset['subject_entity'], dataset['object_entity']):
    i, j = eval(i), eval(j)  # convert str to dict
    sbj_ntt, sbj_ner, sbj_start, sbj_end = i["word"], i["type"], i["start_idx"], i["end_idx"]
    obj_ntt, obj_ner, obj_start, obj_end = j["word"], j["type"], j["start_idx"], j["end_idx"]

    subject_entity.append(sbj_ntt)
    subject_ner.append(sbj_ner)
    subject_start.append(sbj_start)
    subject_end.append(sbj_end)

    object_entity.append(obj_ntt)
    object_ner.append(obj_ner)
    object_start.append(obj_start)
    object_end.append(obj_end)

  out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':dataset['sentence'],'subject_entity':subject_entity,
                              'subject_ner':subject_ner,'subject_start':subject_start,'subject_end':subject_end,
                              'object_entity':object_entity,'object_ner':object_ner,'object_start':object_start,
                              'object_end':object_end,'label':dataset['label']})

  out_dataset['ner_tagged_sent'] = [add_ner_tagging(row) for idx, row in out_dataset.iterrows()]
  out_dataset.drop(["subject_start", "subject_end", "object_start", "object_end"], axis=1, inplace=True)

  return out_dataset

def add_ner_tagging(row):
    """ Entity의 앞뒤에 해당 Entity 의 NER 정보를 tagging 해줍니다. """

    sbj_start = row.subject_start
    sbj_end = row.subject_end
    obj_start = row.object_start
    obj_end = row.object_end

    if sbj_start > obj_start:  # object가 앞에 있을 때
        sent = row.sentence[:obj_start] + f"@*{row.object_ner}*" + row.object_entity + f"@" + row.sentence[obj_end+1:sbj_start]
        sent += f"@*{row.subject_ner}*" + row.subject_entity + f"@" + row.sentence[sbj_end+1:]

    else:  # subject가 앞에 있을 때
        sent = row.sentence[:sbj_start] + f"@*{row.subject_ner}*" + row.subject_entity + f"@" + row.sentence[sbj_end+1:obj_start]
        sent += f"@*{row.object_ner}*" + row.object_entity + f"@" + row.sentence[obj_end+1:]

    return sent

def load_data(dataset_dir, NER_tagging=False):
  """ csv 파일을 경로에 맞게 불러 옵니다. """
  pd_dataset = pd.read_csv(dataset_dir)
  # dataset = data_pruning(pd_dataset)
  if NER_tagging:
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



def tokenized_dataset(dataset, tokenizer, model, NER_tagging=False):

    if NER_tagging:
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
          max_length=256,
          add_special_tokens=True,
          return_token_type_ids=False if 'roberta' in model else True,
          )

    return tokenized_sentences