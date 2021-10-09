# Boostcamp Relation Extraction Competition
QA시스템 구축, 감정 분석, 요약 등 다양한 NLP task에서 문장 속 단어간의 관계 데이터는 정보 파악에서 중요한 역할을 합니다. 이번 대회의 목적은 문장, 단어에 대한 정보를 통해 문장 속에서 단어 사이의 관계를 추론하는 모델을 학습시키는 것 이었습니다. 결과적으로는 총 30개의 관계 클래스 중 하나를 예측한 결과와 30개의 클래스 각각에 대해서 예측한 확률을 반환하는 모델을 생성하도록 하였습니다.

대회 사이트 [AI stage](https://stages.ai/competitions/75/overview/description)

## Hardware
AI stage에서 제공한 server, GPU
- GPU: V100

## 개요
1. [Installation](#installation)
2. [Dataset](#dataset)
3. [Data Analysis](#data-analysis)
4. [Data Preprocessing](#data-preprocessing)
5. [Modeling](#modeling)
6. [Ensemble](#ensemble)
7. [References](#references)

## Installation
다음과 같은 명령어로 필요한 libraries를 다운 받습니다.
```
pip install -r requirements.txt
```

## Dataset
파일: dataset/train/train.csv, dataset/test/test.csv


## Data Analysis
파일: code/EDA.ipynb

## Data preprocessing
파일: code/preprocess_EDA.ipynb, translate.py, translate_entity.py

## Modeling
파일:

## Ensemble
파일:


## References
https://arxiv.org/pdf/1812.01187.pdf
https://kmhana.tistory.com/25
