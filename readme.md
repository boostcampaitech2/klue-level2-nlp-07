# Boostcamp Relation Extraction Competition
## **Table of contents**

1. [Introduction](#introduction)
2. [Project Outline](#project-outline)
3. [Solution](#solution)
4. [How to Use](#how-to-use)

# 1. Introduction  
<br/>
<p align="center">
   <img src="https://user-images.githubusercontent.com/62708568/136650411-a9923f11-eb89-4832-8c86-89ee48c62f69.png" style="width:800px;"/>
</p>

<br/>


## â TEAM : ì¡°ì§KLUEë
### ð Members  

ê¹ë³´ì±|ê¹ì§í|ê¹íì|ë°ì´ì­|ì´ë¤ê³¤|ì ë¯¸ì|ì ëí´
:-:|:-:|:-:|:-:|:-:|:-:|:-:
![image1][image1]|![image2][image2]|![image3][image3]|![image4][image4]|![image5][image5]|![image6][image6]|![image7][image7]
[Github](https://github.com/Barleysack)|[Github](https://github.com/JIHOO97)|[Github](https://github.com/vgptnv)|[Github](https://github.com/Tentoto)|[Github](https://github.com/DagonLee)|[Github](https://github.com/ekdub92)|[Github](https://github.com/Doohae)

### ð Contribution  
`ê¹ë³´ì±` &nbsp; Preprocessing(Data pruning â¢ clean punctuation) â¢ Ensemble(Weighted Vote) â¢ Github management  
`ê¹ì§í` &nbsp; EDA â¢ Data Augmentation(`EDA` â¢ `BackTranslation`) â¢ Binary classifier experiment  
`ê¹íì` &nbsp; Preprocessing (NER Marker) â¢ Data Augmentation(Entity Swap augmentation)  
`ë°ì´ì­` &nbsp; Preprocessing(clean punctuation â¢ special character removal) â¢ Binary classifier experiment  
`ì´ë¤ê³¤` &nbsp; Custom Token Addition â¢ Model Embedding Size Modification â¢ Vocab Modification â¢ Tokenizer Experiment  
`ì ë¯¸ì` &nbsp; Data Visualization â¢ Modeling â¢ Binary classifier experiment â¢ Ensemble  
`ì ëí´` &nbsp; EDA â¢ Data Augmentation(`EDA` â¢ `AEDA` â¢ `RandomDeletion` â¢ `BackTranslation`) â¢ Code Abstraction  

[image1]: https://avatars.githubusercontent.com/u/56079922?v=4
[image2]: https://avatars.githubusercontent.com/u/57887761?v=4
[image3]: https://avatars.githubusercontent.com/u/62708568?v=4
[image4]: https://avatars.githubusercontent.com/u/80071163?v=4
[image5]: https://avatars.githubusercontent.com/u/43575986?v=4
[image6]: https://avatars.githubusercontent.com/u/42200769?v=4
[image7]: https://avatars.githubusercontent.com/u/80743307?v=4

<br/>

# 2. Project Outline  
<p align="center">
   <img src="https://user-images.githubusercontent.com/43575986/136648106-87ba583b-61ba-43a0-a05e-95bf8a0c8d8d.png" width="500" height="300">
   <img src="https://user-images.githubusercontent.com/43575986/136648152-16d3caa3-323e-4240-8e6c-a9cd6c6279d7.png" width="500" height="300">
</p>

- Task : ë¬¸ì¥ ë´ ê°ì²´ê° ê´ê³ ì¶ì¶ (Relation Extraction)
- Date : 2021.09.27 - 2021.10.07 (2 weeks)
- Description : QA ìì¤í êµ¬ì¶, ê°ì  ë¶ì, ìì½ ë± ë¤ìí NLP taskìì ë¬¸ì¥ ì ë¨ì´ê°ì ê´ê³ ë°ì´í°ë ì ë³´ íììì ì¤ìí ì­í ì í©ëë¤. ì´ë² ëíì ëª©ì ì ë¬¸ì¥, ë¨ì´ì ëí ì ë³´ë¥¼ íµí´ ë¬¸ì¥ ììì ë¨ì´ ì¬ì´ì ê´ê³ë¥¼ ì¶ë¡ íë ëª¨ë¸ì íìµìí¤ë ê²ì´ììµëë¤. ê²°ê³¼ì ì¼ë¡ë ì´ 30ê°ì ê´ê³ í´ëì¤ ì¤ íëë¥¼ ìì¸¡í ê²°ê³¼ì 30ê°ì í´ëì¤ ê°ê°ì ëí´ì ìì¸¡í íë¥ ì ë°ííë ëª¨ë¸ì ìì±íëë¡ íììµëë¤.
- Train : 32,470ê°
- Test : 7,765ê°


### ð Final Score
![1](https://user-images.githubusercontent.com/62708568/136651953-f4c13acb-0736-4f8b-8530-b7ab2d309dd3.JPG)

ëí ì¬ì´í¸ : [AI stage](https://stages.ai/competitions/75/overview/description)

<br/>


# 3. Solution

### KEY POINT

- No-relationì ë°ì´í°ê° ìëì ì¼ë¡ ë§ììµëë¤.
- No-relation(label:0)ê³¼ Have-relation(label:1~29) ë°ì´í° ê°ì ë¶í¬ ì°¨ì´ë ì»¸ìµëë¤.
    - ì´ë¥¼ í´ê²°íê¸° ìí´ ëì ë¶ë¥íë binary classifier modelì êµ¬ííìµëë¤.
- ë°ì´í° ë¶ê· í ë¬¸ì ê° ì¬ê°íì¬ Data augmentationì ëí ì¤ìëê° í¬ë¤ê³  íë¨íìµëë¤.
    - Back translation
    - `EDA` `AEDA` `RandomDeletion`
    - ëì¹­ì±ì´ ìë relationì ê²½ì° subject, object `EntitySwap`
- Relation Extraction Taskë¥¼ ì ìííê¸° ìí fine-tuning ê¸°ë²ì¼ë¡ ë°ì´í°ì NER markerë¥¼ ì¶ê°íìµëë¤.
- Weighted ensembleì íµí ì±ë¥ í¥ìì ê¸°ëíìµëë¤.

### Checklist

- [x]  Exploratory Data Analysis
- [x]  Data Visualization
- [x]  Data Preprocessing(`special character removal`)
- [x]  Inserting NER Marker
- [x]  Transformer based model (`BERT`, `klue/RoBERTa` `XLM-RoBERTa`)
- [x]  Data Augmentation(`Back Translation`, `EDA`, `AEDA`, `Entity-Swap`)
- [x]  Model with binary classifier
- [x]  Ensemble(weighted voting)
- [x]  Experimental Logging (`WandB`, `tensorboard`)
- [ ]  Customize Model Architecture
- [ ]  Customize Loss (Focal Loss + Label Smoothing)
- [ ]  Stratified k-fold cross validation
  
### Evaluation  

ë¨ì¼ ëª¨ë¸ì Evaluation ê²°ê³¼ë ìëì ê°ìµëë¤.  
ìë ë¦¬ì¤í¸ì ëª¨ë¸ì í¹ìë¬¸ìë¥¼ ì ê±°í (`special character removal`) ë°ì´í° ì ì²ë¦¬ ê³¼ì ì ê±°ì¹ í íìµì´ ì§íëììµëë¤. 

| Method | Micro F1-score |
| --- | --- |
| `KLUE/BERT-base` | 67.602 |
| `KLUE/RoBERTa-base` | 68.064 |
| `kykim/bert-kor-base` | 68.9 |
| `KLUE/RoBERTa-large` | 71.167 |
| `KLUE/RoBERTa-large` + NER Marker(w/ adding special_token) | 69.615 |
| `KLUE/RoBERTa-large` + NER Marker(w/o adding special_token) | 70.444 |
| `KLUE/RoBERTa-large` + Entity Marker | 68.617 |
| `KLUE/RoBERTa-large` + NER Marker + Data Augmentation(`EntitySwap`) | 69.646 |
| `XLM-RoBERTa-large` + Data Augmentation (`EDA`:Original=1:1) | 68.994 |
| `KLUE/RoBERTa-large` + Data Augmentation (`RandomDeletion`:Original=1:1) | 71.167 |
| `KLUE/RoBERTa-large` + Data Augmentation (`EDA`:Original=1:1) | 72.862 |
| `KLUE/RoBERTa-large` + binary classifier + Data Augmentation (`BackTranslation`:Original=1:1) | 70.731 |
| `KLUE/RoBERTa-large` + Data Augmentation (`BackTranslation`:Original=1:1) | 72.969 |

# 4. How to Use  

## **Installation**

ë¤ìê³¼ ê°ì ëªë ¹ì´ë¡ íìí librariesë¥¼ ë¤ì´ ë°ìµëë¤.
```
pip install -r requirements.txt
```  

KoEDA ëª¨ë
```
pip install koeda  
apt-get install g++ openjdk-8-jdk python3-dev python3-pip curl  
python3 -m pip install --upgrade pip  
python3 -m pip install konlpy  
apt-get install curl git  
bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh)  
```  

Google deep_translator ëª¨ë
```
pip install -U deep-translator
```  
  
## **Dataset**

íì¼: dataset/train/train.csv, dataset/test/test_data.csv

## **Data Analysis**

íì¼: [code/EDA.ipynb](https://github.com/boostcampaitech2/klue-level2-nlp-07/blob/master/code/EDA.ipynb), [/concat.ipynb](https://github.com/boostcampaitech2/klue-level2-nlp-07/blob/master/code/concat.ipynb), [/cleanse.ipynb](https://github.com/boostcampaitech2/klue-level2-nlp-07/blob/master/code/cleanse.ipynb), [/preprocess_EDA.ipynb](https://github.com/boostcampaitech2/klue-level2-nlp-07/blob/master/code/preprocess_EDA.ipynb), [/papago.ipynb](https://github.com/boostcampaitech2/klue-level2-nlp-07/blob/master/code/papago.ipynb)

## **Data preprocessing & Data Augmentation**

íì¼: [code/preprocess_EDA.ipynb](https://github.com/boostcampaitech2/klue-level2-nlp-07/blob/master/code/preprocess_EDA.ipynb), [translate.py](https://github.com/boostcampaitech2/klue-level2-nlp-07/blob/master/code/translate.py), [translate_entity.py](https://github.com/boostcampaitech2/klue-level2-nlp-07/blob/master/code/translate_entity.py), [create_augments.py](https://github.com/boostcampaitech2/klue-level2-nlp-07/blob/master/code/create_augments.py), [create_entityswap_augments.py](https://github.com/boostcampaitech2/klue-level2-nlp-07/blob/master/code/create_entityswap_augments.py), 

## **Modeling**

íì¼: [train.py](https://github.com/boostcampaitech2/klue-level2-nlp-07/blob/master/code/train.py), [inference.py](https://github.com/boostcampaitech2/klue-level2-nlp-07/blob/master/code/inference.py), [train_binary_classifier.py](https://github.com/boostcampaitech2/klue-level2-nlp-07/blob/master/code/train_binary_classifier.py), [inference_binary_classifier.py](https://github.com/boostcampaitech2/klue-level2-nlp-07/blob/master/code/inference_binary_classifier.py)

## **Ensemble**

íì¼: [blender.py](http://blender.py), [blender.ipynb](https://github.com/boostcampaitech2/klue-level2-nlp-07/blob/master/code/blender.ipynb)

## Directory

```
.
âââklue-level2-nlp-07
|    âââcode/
|    âââdataset
â        âââ test
â        âââ train

```

- `code` íì¼ ììë ê°ê° **data preprocessing** â¢ **train** â¢ **inference**ê° ê°ë¥í ë¼ì´ë¸ë¬ë¦¬ê° ë¤ì´ììµëë¤
- ì¬ì©ìë ì ì²´ ì½ëë¥¼ ë´ë ¤ë°ì í, argument ìµìì ì§ì íì¬ ê°ë³ ë¼ì´ë¸ë¬ë¦¬ ëª¨ë¸ì íì©í  ì ììµëë¤

## **Hardware**
ë³¸ Repositoryë AI stageìì ì ê³µí server, GPUë¥¼ ê¸°ë°ì¼ë¡ ìì±ë ì½ëìëë¤.
- GPU: V100




