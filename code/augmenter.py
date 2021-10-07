from koeda import EDA, AEDA
import pandas as pd
import numpy as np

from load_data import *

from tqdm import tqdm

data = load_data("../dataset/train/train.csv")


##################### 아래 alpha, prob, ratio, punctuations 등을 필요에 맞게 수정하세요 #####################
eda = EDA(morpheme_analyzer="Okt", alpha_sr=-0.2, alpha_ri=0, alpha_rs=0, prob_rd=0)
aeda = AEDA(morpheme_analyzer="Okt", punc_ratio=0.3, punctuations=["?", "!", ".", ",", "", "##"])
#####################################################################################################
FILE_NAME = 'without_aug.csv'
############################# 위에서 저장할 파일명을 .csv 형식으로 설정해주세요 ################################

def right_augment(en1, en2, text):
    if en1 not in text or en2 not in text:
        return True
    return False


def augment_sent(df, my_eda, my_aeda):
    sentences = list(df['sentence'])
    entities1, entities2 = list(df['object_entity']), list(df['subject_entity'])
    aug_sentences = []
    source = list(df['source'])
    for i, sent in tqdm(enumerate(sentences)):
        sent_a = sent[:len(sent)//2]
        sent_b = sent[len(sent)//2:]

        aug_sent_a = aeda(sent_a)
        aug_sent_b = aeda(sent_b)
        aug_sent = aug_sent_a+aug_sent_b

        ent1, ent2 = entities1[i], entities2[i]
        if not right_augment(ent1, ent2, aug_sent):
            aug_sent = sent
        aug_sentences.append(aug_sent)
    df['sentence'] = aug_sentences
    df['source'] = source
    return df

def pure_df_sent(df):
    sentences = list(df['sentence'])
    entities1, entities2 = list(df['object_entity']), list(df['subject_entity'])
    
    pure_sentences = []
    for i, sent in tqdm(enumerate(sentences)):
        sent = sent
        ent1, ent2 = entities1[i], entities2[i]
        pure_sentences.append(sent)
    df['sentence'] = pure_sentences
    
    return df


def main():
    aug_df = pure_df_sent(data)
    #aug_df = augment_sent(data, eda, aeda)
    aug_df.to_csv("../dataset/train/" + FILE_NAME, index=False)
    
#지금까지 python augmenter.py 를 터미널에서 찾아들어간게 아니라, 어짜피 csv 파일 만드는 코드니 그냥 돌려야겠다 싶어서 F5를 눌렀습니다 ㅋㅋㅋ
#파일 입장에서는 현재 위치가 OPT/ML인거죠! 
#근데 코드에는 상대 경로를 표시하는 ../dataset/train/가 있으니, 못찾겠다고 하는 것 같습니다 ㅎㅎ
#해결!
if __name__ == '__main__':
    main()