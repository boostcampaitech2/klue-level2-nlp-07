import pickle as pickle
import os
import pandas as pd
import torch
import re


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

def preprocessing_dataset_ner(dataset):
  """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
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
  out_dataset['ner_cleaned_sent'] = [clean_punc_ner(sent) for sent in out_dataset.ner_tagged_sent]
  out_dataset.drop(["subject_start", "subject_end", "object_start", "object_end", "ner_tagged_sent"], axis=1, inplace=True)

  return out_dataset


def load_data(dataset_dir):
  """ csv 파일을 경로에 맡게 불러 옵니다. """
  pd_dataset = pd.read_csv(dataset_dir)
  # dataset = preprocessing_dataset(pd_dataset)
  dataset = preprocessing_dataset_ner(pd_dataset)
  
  return dataset


def add_ner_tagging(row):

    sbj_start = row.subject_start
    sbj_end = row.subject_end
    obj_start = row.object_start
    obj_end = row.object_end

    if sbj_start > obj_start:  # object가 앞에 있을 때
        sent = row.sentence[:obj_start] + f"డ{row.object_ner}ఢ " + row.object_entity + f" డఎ{row.object_ner}ఢ " + row.sentence[obj_end+1:sbj_start]
        sent += f"డ{row.subject_ner}ఢ " + row.subject_entity + f" డఎ{row.subject_ner}ఢ " + row.sentence[sbj_end+1:]
        # print(sent)
    else:  # subject가 앞에 있을 때
        sent = row.sentence[:sbj_start] + f"డ{row.subject_ner}ఢ " + row.subject_entity + f" డఎ{row.subject_ner}ఢ " + row.sentence[sbj_end+1:obj_start]
        sent += f"డ{row.object_ner}ఢ " + row.object_entity + f" డఎ{row.object_ner}ఢ " + row.sentence[obj_end+1:]
        # print(sent)
    return sent


def clean_punc_ner(text):
    punct_mapping = {'డ':"[", "ఢ":"]", "ఎ":"/", 'ʿ': '', 'ū': 'u', 'è': 'e', 'ȳ': 'y', 'ồ': 'o', 'ề': 'e', 'â': 'a', 'æ': 'ae', 'ő': 'o', 'ῶ': 'ω', '𑀕': 'Λ', 'ß': 'β', 'ヶ': 'ケ', '‘': "'", '₹': 'e', '´': "'", '°': '', '€': 'e', '™': 'tm', '√': ' sqrt ', '×': 'x', '²': '2', '—': '-', '–': '-', '’': "'", '_': '-', '`': "'", '“': '"', '”': '"', '£': 'e', '∞': 'infinity', '÷': '/', '•': '.', 'à': 'a', '−': '-', 'Ῥ': 'Ρ', 'ầ': 'a', '́': "'", 'ò': 'o', 'Ö': 'O', 'Š': 'S', 'ệ': 'e', 'Ś': 'S', 'ē': 'e', 'ä': 'a', 'ć': 'c', 'ë': 'e', 'å': 'a', 'Ǧ': 'G', 'ạ': 'a', 'ņ': 'n', 'İ': 'I', 'ğ': 'g', 'ê': 'e', 'Č': 'C', 'ã': 'a', 'ḥ': 'h', 'ả': 'a', 'ễ': 'e', '％': '%', 'ợ': 'o', 'Ú': 'U', 'ư': 'u', 'Ž': 'Z', 'ú': 'u', 'É': 'E', 'Ó': 'O', 'ü': 'u', 'é': 'e', 'ā': 'a', 'š': 's', '𑀥': 'D', 'í': 'i', 'û': 'u', 'ý': 'y', 'ī': 'i', 'ï': 'i', 'ộ': 'o', 'ì': 'i', 'ọ': 'o', 'ş': 's', 'ó': 'o', 'ñ': 'n', 'ậ': 'a', 'Â': 'A', 'ù': 'u', 'ô': 'o', 'ố': 'o', 'Á': 'A', 'ö': 'o', 'ơ': 'o', 'ç': 'c', 'ˈ': "'", 'µ': 'μ', '／': '/', '（': '(', 'ｍ': 'm', '˘': ' ', '𑀫': 'ma', '？': '?', 'ł': 'l', 'Đ': 'D', '：': ':', '･': ',', 'Ç': 'C', 'ı': 'i', '，': ',', '𥘺': '祉', '·': ',', '＇': "'", ' ': ' ', '）': ')', '１': '1', 'ø': 'o', '～': '~', '³': '3', '(˘ ³˘)': '', '˹': '"', '｢': '"', '｣': '"', '«': '<<', '˼': '"', '»': '>>', '®': 'R'}
    text = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\[\]\<\>`\'…《》▲△〈〉]', ' ', text)
    
    for p in punct_mapping:
        text = re.sub(p, punct_mapping[p],text)
    return text



def tokenized_dataset(dataset, tokenizer):
  """ tokenizer에 따라 sentence를 tokenizing 합니다."""
  concat_entity = []
  for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
    temp = ''
    temp = e01 + '[SEP]' + e02
    concat_entity.append(temp)
  tokenized_sentences = tokenizer(
      concat_entity,
      # list(dataset['sentence']),
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=256,
      add_special_tokens=True,
      )
  return tokenized_sentences


def tokenized_dataset_ner(dataset, tokenizer):
  """ tokenizer에 따라 sentence를 tokenizing 합니다."""
  concat_entity = []
  for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
    temp = ''
    temp = e01 + '[SEP]' + e02
    concat_entity.append(temp)

  ner_tokens_dict = {"additional_special_tokens": ["[PER]", "[/PER]", "[ORG]", "[/ORG]", "[NOH]", "[/NOH]",
                                    "[POH]", "[/POH]", "[DAT]", "[/DAT]", "[LOC]", "[/LOC]",]}

  num_added_toks = tokenizer.add_special_tokens(ner_tokens_dict)
  print('We have added', num_added_toks, 'tokens')  # 12
  # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e., the length of the tokenizer.

  tokenized_sentences = tokenizer(
      concat_entity,
      list(dataset['ner_cleaned_sent']),
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=256,
      add_special_tokens=True,
      )

  return tokenized_sentences


