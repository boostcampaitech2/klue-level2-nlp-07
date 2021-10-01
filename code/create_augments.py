from koeda import EDA, AEDA
import pandas as pd
import numpy as np

from load_data import *

from tqdm import tqdm

data = load_data("../dataset/train/train.csv")


##################### 아래 alpha, prob, ratio, punctuations 등을 필요에 맞게 수정하세요 #####################
eda = EDA(morpheme_analyzer="Mecab", alpha_sr=-0.2, alpha_ri=0, alpha_rs=0, prob_rd=0)
aeda = AEDA(morpheme_analyzer="Mecab", punc_ratio=0.08, punctuations=["?", "!", ".", ",", "", "##"])
#####################################################################################################
FILE_NAME = 'aug_ver_1.csv'
############################# 위에서 저장할 파일명을 .csv 형식으로 설정해주세요 ################################

def right_augment(en1, en2, text):
    if en1 in text and en2 in text:
        return True
    return False


def augment_sent(df, my_eda, my_aeda):
    sentences = list(df['sentence'])
    entities1, entities2 = list(df['object_entity']), list(df['subject_entity'])
    aug_sentences = []
    for i, sent in tqdm(enumerate(sentences)):
        aug_sent = aeda(eda(sent))
        ent1, ent2 = entities1[i], entities2[i]
        if not right_augment(ent1, ent2, aug_sent):
            aug_sent = sent
        aug_sentences.append(aug_sent)
    df['sentence'] = aug_sentences
    return df



def main():
    aug_df = augment_sent(data, eda, aeda)
    aug_df.to_csv("../dataset/train/" + FILE_NAME, index=False)
    

if __name__ == '__main__':
    main()
