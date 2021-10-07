import pandas as pd
from deep_translator import GoogleTranslator
from load_data import *

from tqdm import tqdm

import argparse


def google_translate(text, tgt="en"):
    try:
        if tgt == "en":
            translator = GoogleTranslator(source='auto', target="en")
            new_txt = translator.translate(text)
        elif tgt == "ko":
            translator1 = GoogleTranslator(source='auto', target="en")
            translator2 = GoogleTranslator(source='auto', target="ko")
            mid_txt = translator1.translate(text)
            new_txt = translator2.translate(mid_txt)
        if new_txt:
            # print(new_txt)
            return new_txt
        # print(text)
        return text
    except:
        print('Problem Occurred!! (Maybe Minor One, Do Not Worry)')
        return text


def check_translate(word, sent):
    if word not in sent:
        # 영어가 아니라면 = 만약 한국어라면
        if word.encode().isalpha() is False:
            new_word = google_translate(word, "en")
            # print(word, "en")
        # 영어라면
        else:
            new_word = google_translate(word, "ko")
            # print(word, "ko")
        if new_word not in sent:
            new_word = word
        return new_word
    else:
        return word

    
def modify_entity(sub, ob, sent):
    if sub[0] == "'":
        sub = sub[1:-1]
        ob = ob[1:-1]
    new_sub = check_translate(sub, sent)
    new_ob = check_translate(ob, sent)
    new_sub = "'" + new_sub + "'"
    new_ob = "'" + new_ob + "'"
    # print(new_sub, new_ob)
    return new_sub, new_ob


parser = argparse.ArgumentParser()
parser.add_argument('--FILE', type=str, default="train_trans.csv")
parser.add_argument('--SAVE', type=str, default="../dataset/train/translate/")
parser.add_argument('--START_IDX', type=int, default=0)
parser.add_argument('--END_IDX', type=int, default=5000)
args = parser.parse_args()

file_dir = "../dataset/train/" + args.FILE
cur_df = pd.read_csv(file_dir)

start_idx = args.START_IDX
end_idx = args.END_IDX if args.END_IDX < len(cur_df) else len(cur_df)
save_dir = args.SAVE

subjects = list(cur_df['subject_entity'])[start_idx:end_idx]
objects = list(cur_df['object_entity'])[start_idx:end_idx]
sentences = list(cur_df['sentence'])[start_idx:end_idx]
labels = list(cur_df['label'])[start_idx:end_idx]
ids = list(cur_df['id'])[start_idx:end_idx]

new_df = pd.DataFrame()

for i in tqdm(range(len(ids)), desc="Modifying Entities..."):
    if labels[i] != 'no_relation':
        mod_sub, mod_ob = modify_entity(subjects[i], objects[i], sentences[i])
        subjects[i], objects[i] = mod_sub, mod_ob
new_df['id'] = ids
new_df['sentence'] = sentences
new_df['subject_entity'] = subjects
new_df['object_entity'] = objects
new_df['label'] = labels
new_df.to_csv(save_dir + f'train_trans{start_idx}_{end_idx}.csv', index=False)
print(f'train_trans{start_idx}_{end_idx}.csv entities modified!!')