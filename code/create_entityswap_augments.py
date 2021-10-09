import pandas as pd
import numpy as np

from load_data import *
from tqdm import tqdm

"""
상호 대칭적인 관계에 대하여 subject, object entity를 swap하여 augmentation하는 파일입니다.
"""

TRAIN_DATA_PATH = load_data("../dataset/train/train.csv", NER_marker=True)

FILE_NAME = 'train_entity_swap.csv'
############################# 위에서 저장할 파일명을 .csv 형식으로 설정해주세요 ################################


def additional_data(TRAIN_DATA_PATH):
    config = {
        "change_entity": {
            "subject_entity": "object_entity",
            "object_entity": "subject_entity",
            "subject_ner": "object_ner",
            "object_ner": "subject_ner",
        },
        "remain_label_list": [
            # "no_relation",
            "org:members",
            # "org:alternate_names",
            # "per:children",
            # "per:alternate_names",
            "per:other_family",
            "per:colleagues",
            "per:siblings",
            # "per:spouse",
            "org:member_of",
            # "per:parents",
            # "org:top_members/employees",
        ],
        "change_values": {
            "org:member_of": "org:members",
            # "org:members": "org:member_of",
            # "per:parents": "per:children",
            # "per:children": "per:parents",
            # "org:top_members/employees": "per:employee_of",
        },
        "cols": ["id", "sentence", "subject_entity", "subject_ner", "object_entity", "object_ner", "label"],
    }

    # 훈련 데이터를 불러오고 subject_entity와 object_entity만 바꾼다.
    add_data = load_data(TRAIN_DATA_PATH).rename(columns=config["change_entity"])
    # 추가 데이터를 만들 수 있는 라벨만 남긴다
    add_data = add_data[add_data.label.isin(config["remain_label_list"])]
    # 속성 정렬을 해준다 (정렬을 안할경우 obj와 sub의 순서가 바뀌어 보기 불편함)
    add_data = add_data[config["cols"]]
    # 서로 반대되는 뜻을 가진 라벨을 바꿔준다.
    add_data = add_data.replace({"label": config["change_values"]})
    return add_data

def train_data_with_addition(TRAIN_DATA_PATH):
  added_data = load_data(TRAIN_DATA_PATH).append(additional_data(TRAIN_DATA_PATH))
  added_data = added_data.drop_duplicates(subset=['subject_entity', 'object_entity', 'sentence'], keep='first') # 중복데이터 제거, 이미 존재할 경우 기존 학습데이터를 사용
  return added_data


def main():
    aug_output = train_data_with_addition(TRAIN_DATA_PATH)
    aug_output.to_csv("../dataset/train/" + FILE_NAME, index=False)
    

if __name__ == '__main__':
    main()
