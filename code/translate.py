import googletrans
import kakaotrans

from tqdm import tqdm
import pandas as pd
from load_data import *

def google_translate(en1, en2, sent):
    translator = googletrans.Translator()
    cn_sent = translator.translate(sent, dest='zh-cn')
    cn_en1 = translator.translate(en1, dest='zh-cn')
    cn_en2 = translator.translate(en2, dest='zh-cn')
    ko_sent = translator.translate(cn_sent.text, dest='ko')
    ko_en1 = translator.translate(cn_en1.text, dest='ko')
    ko_en2 = translator.translate(cn_en2.text, dest='ko')
    return ko_en1.text, ko_en2.text, ko_sent.text


def kakao_translate(en1, en2, sent):
    translator = kakaotrans.Translator()
    en_sent = translator.translate(sent, src='kr', tgt='en')
    en_en1 = translator.translate(en1, src='kr', tgt='en')
    en_en2 = translator.translate(en2, src='kr', tgt='en')
    ko_sent = translator.translate(en_sent, src='en', tgt='kr')
    ko_en1 = translator.translate(en_en1, src='en', tgt='kr')
    ko_en2 = translator.translate(en_en2, src='en', tgt='kr')
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

### modified below ###
SEP = 100
num_iter = len(sentences)//SEP
for idx in range(num_iter+1):
    if idx == num_iter:
        subs, obs, sents = sub_entity[idx*SEP:(idx+1)*SEP], ob_entity[idx*SEP:(idx+1)*SEP], sentences[idx*SEP:(idx+1)*SEP]
    else:
        subs, obs, sents = sub_entity[idx*SEP:], ob_entity[idx*SEP:], sentences[idx*SEP:]
    new_sub_entity, new_ob_entity, new_sentences = double_translate(subs, obs, sents)
    new_df = pd.DataFrame({'id':df['id'][idx*SEP:(idx+1)*SEP],'sentence':new_sentences,'subject_entity':new_sub_entity,'object_entity':new_ob_entity,'label':df['label'][idx*SEP:(idx+1)*SEP],})
    new_df.to_csv(f'../dataset/train/translate_real/train_trans{idx}.csv', index=False)