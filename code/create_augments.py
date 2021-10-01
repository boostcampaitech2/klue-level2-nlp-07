from koeda import EDA, AEDA
import pandas as pd
import numpy as np

from load_data import *

from tqdm import tqdm

from random import random

data = load_data("../dataset/train/train.csv")


##################### 아래 alpha, prob, ratio, punctuations 등을 필요에 맞게 수정하세요 #####################
eda = EDA(morpheme_analyzer="Mecab", alpha_sr=0.5, alpha_ri=0.2, alpha_rs=0, prob_rd=0)
aeda = AEDA(morpheme_analyzer="Mecab", punc_ratio=0.1, punctuations=["?", "!", ".", ",", "", "##", "한편"])
#####################################################################################################
FILE_NAME = 'aug_ver_4.csv'
############################# 위에서 저장할 파일명을 .csv 형식으로 설정해주세요 ################################

def right_augment(en1, en2, text):
    return en1[1:-1] in text and en2[1:-1] in text

def apply_augment(func, sentence, en1, en2):
    aug_sentence = func(sentence)
    if right_augment(en1, en2, aug_sentence):
        return aug_sentence
    return sentence
    
def augment_sent(df, my_eda, my_aeda):
    sentences = list(df['sentence'])
    entities1, entities2 = list(df['subject_entity']), list(df['object_entity'])
    original = list(map(lambda sent, ent1, ent2: (sent, ent1, ent2), sentences, entities1, entities2))
    aug_sentences = []
    for sent, ent1, ent2 in tqdm(original, desc="Count 10 Seconds ..."):
        n = random()
        if n <= 0.5:
            aug_sent = apply_augment(eda, sent, ent1, ent2)
        elif n <= 0.8:
            aug_sent = apply_augment(aeda, sent, ent1, ent2)
        else:
            aug_sent = sent
        aug_sentences.append(aug_sent)
    output = pd.DataFrame({'id':df['id'],'sentence':aug_sentences,'subject_entity':entities1,'object_entity':entities2,'label':df['label'],})
    return output


def main():
    aug_output = augment_sent(data, eda, aeda)
    aug_output.to_csv("../dataset/train/" + FILE_NAME, index=False)
    

if __name__ == '__main__':
    main()