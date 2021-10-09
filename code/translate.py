from deep_translator import GoogleTranslator

from tqdm import tqdm
import pandas as pd
from load_data import *

def google_translate(sent):
    ko_to_en = GoogleTranslator(source='auto', target='en')
    en_to_ko = GoogleTranslator(source='auto', target='ko')
    en_sent = ko_to_en.translate(sent)
    ko_sent = en_to_ko.translate(en_sent)
    return ko_sent


def double_translate(sent_list):
    new_sent = []
    for idx in tqdm(range(len(sent_list)), desc="Tons of Translations Ongoing..."):
        trans_sent = google_translate(sent_list[idx])
        new_sent.append(trans_sent)
    return new_sent


df = load_data('../dataset/train/train85.csv')
sentences = list(df['sentence'])


SEP = 50
num_iter = len(sentences)//SEP
for idx in range(num_iter+1):
    if idx != num_iter:
        sents = sentences[idx*SEP:(idx+1)*SEP]
    else:
        sents = sentences[idx*SEP:]
    new_sentences = double_translate(sents)
    if idx != num_iter:
        new_df = pd.DataFrame({'id':df['id'][idx*SEP:(idx+1)*SEP],'sentence':new_sentences,'subject_entity':df['subject_entity'][idx*SEP:(idx+1)*SEP],'object_entity':df['object_entity'][idx*SEP:(idx+1)*SEP],'label':df['label'][idx*SEP:(idx+1)*SEP],})
    else:
        new_df = pd.DataFrame({'id':df['id'][idx*SEP:],'sentence':new_sentences,'subject_entity':df['subject_entity'][idx*SEP:],'object_entity':df['object_entity'][idx*SEP:],'label':df['label'][idx*SEP:],})
    new_df.to_csv(f'../dataset/train/translate/train_trans{idx}.csv', index=False)
    print(f"train_trans{idx}.csv saved now to translate_whole!!")