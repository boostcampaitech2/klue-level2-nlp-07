import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from load_data import *

class RE_binary_Dataset(torch.utils.data.Dataset):
  """ Dataset 구성을 위한 class."""
  def __init__(self, pair_dataset, labels):
    self.pair_dataset = pair_dataset
    self.labels = labels
    print('len :', len(self.pair_dataset))

  def __getitem__(self, idx):
    item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
    item['labels'] = torch.tensor(self.labels[idx])
    item['index'] = idx
    return item

  def __len__(self):
    return len(self.labels)

def load_test_dataset(dataset_dir, tokenizer, model_name):
  """
    test dataset을 불러온 후,
    tokenizing 합니다.
  """
  test_dataset = load_data(dataset_dir)
  test_label = list(map(int,test_dataset['label'].values)) # label column -> list
  # tokenizing dataset, return indexes, tokenized dataset, labels
  tokenized_test = tokenized_dataset(test_dataset, tokenizer, model_name)
  return test_dataset['id'], tokenized_test, test_label

def inference_binary_classifier(): #(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    Tokenizer_NAME = "klue/roberta-large" # args.binary_tokenizer
    MODEL_NAME = './binary_best_model/' # args.binary_save_dir
    BATCH_SIZE = 32 # args.binary_bsz

    tokenizer = AutoTokenizer.from_pretrained(Tokenizer_NAME)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.parameters
    model.to(device)

    test_dataset_dir = "/opt/ml/dataset/test/test_data.csv"
    test_id, test_dataset, test_label = load_test_dataset(test_dataset_dir, tokenizer, Tokenizer_NAME)
    RE_binary_test_dataset = RE_binary_Dataset(test_dataset, test_label)
    # print(RE_binary_test_dataset[-1])


inference_binary_classifier()