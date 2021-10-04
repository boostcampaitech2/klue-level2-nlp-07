import googletrans
import kakaotrans

from tqdm import tqdm
import pandas as pd
from load_data import *

def google_translate(en1, en2, sent):
    translator = googletrans.Translator()
    ja_sent = translator.translate(sent, dest='ja')
    ja_en1 = translator.translate(en1, dest='ja')
    ja_en2 = translator.translate(en2, dest='ja')
    ko_sent = translator.translate(ja_sent.text, dest='ko')
    ko_en1 = translator.translate(ja_en1.text, dest='ko')
    ko_en2 = translator.translate(ja_en2.text, dest='ko')
    return ko_en1, ko_en2, ko_sent


def kakao_translate(en1, en2, sent):
    translator = kakaotrans.Translator()
    en_sent = translator.translate(sent, src='kr', dest='en')
    en_en1 = translator.translate(en1, src='kr', dest='en')
    en_en2 = translator.translate(en2, src='kr', dest='en')
    ko_sent = translator.translate(en_sent, src='en', dest='kr')
    ko_en1 = translator.translate(en_en1, src='en', dest='kr')
    ko_en2 = translator.translate(en_en2, src='en', dest='kr')
    return ko_en1, ko_en2, ko_sent


def double_translate(sub_list, ob_list, sent_list):
    new_sub, new_ob, new_sent = [], [], []
    for idx in tqdm(range(len(sub_list)), desc="Tons of Translations Ongoing..."):
        try:
            trans_sub, trans_ob, trans_sent = kakao_translate(sub_list[idx], ob_list[idx], sent_list[idx])
        except:
            trans_sub, trans_ob, trans_sent = google_translate(sub_list[idx], ob_list[idx], sent_list[idx])
        new_sub.append(trans_sub)
        new_ob.append(trans_ob)
        new_sent.append(trans_sent)
    return new_sub, new_ob, new_sent

df = load_data('../dataset/train/train85.csv')
sentences = list(df['sentence'])
sub_entity = [ent[1:-1] for ent in list(df['subject_entity'])]
ob_entity = [ent[1:-1] for ent in list(df['object_entity'])]

new_sentences, new_sub_entity, new_ob_entity = double_translate(sub_entity, ob_entity, sentences)
new_df = pd.DataFrame({'id':df['id'],'sentence':new_sentences,'subject_entity':new_sub_entity,'object_entity':new_ob_entity,'label':df['label'],})
new_df.to_csv('../dataset/train/train_trans.csv')