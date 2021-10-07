import pandas as pd
from pandas.io.parsers import read_csv
import re
import argparse

'''def clean_punc(text):
    punct_mapping = { 'ū': 'u', 'è': 'e', 'ȳ': 'y', 'ồ': 'o', 'ề': 'e', 'â': 'a', 'æ': 'ae', 'ő': 'o', 'α':'alpha','ß':'beta', 'β':'beta', 'ヶ': 'ケ', '‘': "'", '₹': 'e', '´': "'", '°': '', '€': 'euro', '™': 'tm', '√': ' sqrt ', '×': 'x', '²': '2', '—': '-', '–': '-', '’': "'", '_': '-', '`': "'", '“': '"', '”': '"', '£': 'e', '∞': 'infinity', '÷': '/', '•': '.', 'à': 'a', '−': '-', 'Ῥ': 'Ρ', 'ầ': 'a', '́': "'", 'ò': 'o', 'Ö': 'O', 'Š': 'S', 'ệ': 'e', 'Ś': 'S', 'ē': 'e', 'ä': 'a', 'ć': 'c', 'ë': 'e', 'å': 'a', 'Ǧ': 'G', 'ạ': 'a', 'ņ': 'n', 'İ': 'I', 'ğ': 'g', 'ê': 'e', 'Č': 'C', 'ã': 'a', 'ḥ': 'h', 'ả': 'a', 'ễ': 'e', '％': '%', 'ợ': 'o', 'Ú': 'U', 'ư': 'u', 'Ž': 'Z', 'ú': 'u', 'É': 'E', 'Ó': 'O', 'ü': 'u', 'é': 'e', 'ā': 'a', 'š': 's', '𑀥': 'D', 'í': 'i', 'û': 'u', 'ý': 'y', 'ī': 'i', 'ï': 'i', 'ộ': 'o', 'ì': 'i', 'ọ': 'o', 'ş': 's', 'ó': 'o', 'ñ': 'n', 'ậ': 'a', 'Â': 'A', 'ù': 'u', 'ô': 'o', 'ố': 'o', 'Á': 'A', 'ö': 'o', 'ơ': 'o', 'ç': 'c', 'ˈ': "'", 'µ': 'μ', '／': '/', '（': '(', 'ｍ': 'm', '˘': ' ', '？': '?', 'ł': 'l', 'Đ': 'D', '：': ':', '･': ',', 'Ç': 'C', 'ı': 'i', '，': ',', '𥘺': '祉', '·': ',', '＇': "'", ' ': ' ', '）': ')', '１': '1', 'ø': 'o', '～': '~', '³': '3', '(˘ ³˘)': '', '˹': '<', '｢': '<', '｣': '>', '«': '<', '˼': '>', '»': '>'}

    for p in punct_mapping:
        text=re.sub(p, punct_mapping[p],text)
    return text

vocab={}
with open('../dataset/train/vocab.txt','r') as f:
    while True:
        line=re.sub('\n','',f.readline())
        vocab[line]=0
        if not line:
            break

MODEL_NAME = "klue/bert-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
not_in_vocab=set()
data=pd.read_csv('../dataset/train/train.csv')['sentence']
a=0
for sentence in data:
    sentence=clean_punc(sentence)
    sentence = re.sub('[ぁ-ゔァ-ヴー々〆〤\\u0250-\\u02AD\\u1200-\\u137F\\u0600-\\u06FF\\u0750-\\u077F\\uFB50-\\uFDFF\\uFE70‌​-\\uFEFF\\u0900-\\u097F\\u0400-\\u04FF\\u0370-\\u03FF ]',' ',sentence)
    sentence = re.sub('\s+',' ',sentence)
    for idx,token in enumerate(tokenizer.tokenize(sentence)):
        if token not in vocab:
            not_in_vocab.add(token)'''
'''
def clean_sentence(sentence):
    punct_mapping = {'ū': 'u', 'è': 'e', 'ȳ': 'y', 'ồ': 'o', 'ề': 'e', 'â': 'a', 'æ': 'ae', 'ő': 'o', 'α': 'alpha', 'ß': 'ss', 'β': 'beta', 'ヶ': 'ケ', '₹': 'e', '°': '', '€': 'euro', '™': 'tm', '√': ' sqrt ', '–': '-', '£': 'e', '∞': 'infinity', '÷': '/', 'à': 'a', '−': '-', 'Ῥ': 'Ρ', 'ầ': 'a', '́': "'", 'ò': 'o', 'Ö': 'O', 'Š': 'S', 'ệ': 'e', 'Ś': 'S', 'ē': 'e', 'ä': 'a', 'ć': 'c', 'ë': 'e', 'å': 'a', 'Ǧ': 'G', 'ạ': 'a', 'ņ': 'n', 'İ': 'I', 'ğ': 'g', 'ê': 'e', 'Č': 'C', 'ã': 'a', 'ḥ': 'h', 'ả': 'a', 'ễ': 'e', 'ợ': 'o', 'Ú': 'U', 'ư': 'u', 'Ž': 'Z', 'ú': 'u', 'É': 'E', 'Ó': 'O', 'ü': 'u', 'ā': 'a', 'š': 's', '𑀥': 'D', 'í': 'i', 'û': 'u', 'ý': 'y', 'ī': 'i', 'ï': 'i', 'ộ': 'o', 'ì': 'i', 'ọ': 'o', 'ş': 's', 'ó': 'o', 'ñ': 'n', 'ậ': 'a', 'Â': 'A', 'ù': 'u', 'ô': 'o', 'ố': 'o', 'Á': 'A', 'ö': 'o', 'ơ': 'o', 'ç': 'c', 'ˈ': "'", 'µ': 'μ', '／': '/', '（': '(', '˘': ' ', '？': '?', 'ł': 'l', 'Đ': 'D', '･': ',', 'Ç': 'C', 'ı': 'i', '𥘺': '祉', '＇': "'", ' ': ' ', '）': ')', '１': '1', 'ø': 'o', '～': '~', '³': '3', '(˘ ³˘)': '', '˹': '<', '«': '<', '˼': '>', '»': '>'}
    sub_pat='‘｢♀▼女，◆㎜㈜+?😍●㎝『》不〔Ⅰ!〉´♡️「②＆\'=∙｣㎖㎡金•▲ｔ☆♥▷‧․ᆞ①㎏℃⑤』%Ⅱ│○】×─✔”〕_,&｜²☞→↑【#};◇━]理＝⠀😂👉⊙`(💕👍△％《▶③é:|＜；*⑦/〈😭※~―@"—≫✨[㎎⑥㏊∼ㆍ＞－^❤ℓ：>🤣★ㅤ李<ｍ·Ⅲ＋.◈㎢■…$≪㎞‥□」🏻㎾＂④・{😆“㎥’'
    for p in punct_mapping:
        sentence=re.sub(p, punct_mapping[p],sentence)
    sentence = re.sub(f'[^- ㄱ-ㅎㅏ-ㅣ가-힣0-9a-zA-Zぁ-ゔァ-ヴー々〆〤一-龥(){re.escape(sub_pat)}]',' ',sentence)
    sentence = re.sub('\s+',' ',sentence)
    sentence = re.sub('\([, ]*\)','',sentence)
    return sentence

data=pd.read_csv('../dataset/train/train.csv')
data=data.drop_duplicates(subset=['sentence','subject_entity','object_entity','label'])
data['subject_entity']=[eval(i)['word']for i in data['subject_entity']]
data['object_entity']=[eval(i)['word']for i in data['object_entity']]
data=data[data['label']=='no_relation']
print(len(data))
data.to_csv('../dataset/train/no_rel.csv',index=False)'''
'''idxs={}
for i in range(len(data)):
    if i%100==0:
        print(i)
    s=data['sentence'].iloc[i]
    se=data['subject_entity'].iloc[i]
    oe=data['object_entity'].iloc[i]
    l=data['label'].iloc[i]
    d=data[(data['sentence']==s)&(data['subject_entity']==se)&(data['object_entity']==oe)&(data['label']!=l)]
    if len(d):
        idxs[i]=d
print(idxs)'''

'''sent=list(data['sentence'])[31900:]
for _ in range(100):
    s=sent.pop(0)
    print(s)
    print(clean_sentence(s))
'''
if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--model_name', type=str, default="klue/roberta-large")
  parser.add_argument('--bsz', type=int, default=32)
  parser.add_argument('--epochs', type=int, default=5)
  parser.add_argument('--save_dir', type=str, default="")
  parser.add_argument('--dev_set', type=str, default="True")
  parser.add_argument('--filter_no_rel', type=bool, default=True)
  args = parser.parse_args()
  
  print(type(args.filter_no_rel),args.filter_no_rel)