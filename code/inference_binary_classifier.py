import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from load_data import *
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np

class RE_binary_Dataset(torch.utils.data.Dataset):
  """ Dataset 구성을 위한 class."""
  def __init__(self, pair_dataset, labels, indexes):
    self.pair_dataset = pair_dataset
    self.labels = labels
    self.indexes = indexes

  def __getitem__(self, idx):
    item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
    item['labels'] = torch.tensor(self.labels[idx])
    item['index'] = self.indexes[idx]
    return item

  def __len__(self):
    return len(self.labels)

def load_test_dataset(dataset_dir, tokenizer, model_name):
  """
    test dataset을 불러온 후,
    tokenizing 합니다.
  """
  # test_dataset = load_data(dataset_dir)
  dataset = pd.read_csv(dataset_dir)
  test_dataset = preprocessing_dataset(dataset)
  test_label = list(map(int,test_dataset['label'].values)) # label column -> list
  # tokenizing dataset, return indexes, tokenized dataset, labels
  tokenized_test = tokenized_dataset(test_dataset, tokenizer, model_name)
  return test_dataset['id'], tokenized_test, test_label

def binary_num_to_label(label):
  origin_label = []
  # with open('dict_num_to_label.pkl', 'rb') as f:
  #   dict_num_to_label = pickle.load(f)
  # print(dict_num_to_label) # 0~29의 {idx:relation}
  dict_num_to_label = {0: 'no_relation', 1: 'relation'}
  for v in label:
    origin_label.append(dict_num_to_label[v])
  return origin_label

def inference_binary(model, model_name, tokenized_sent, device, batch_size):
  dataloader = DataLoader(tokenized_sent, batch_size=batch_size, shuffle=False)
  model.eval()
  output_pred = []
  output_prob = []
  output_index = []
  for i, data in enumerate(tqdm(dataloader)):
    # print(data)
    with torch.no_grad():
      if 'roberta' not in model_name:
        outputs = model(
          input_ids=data['input_ids'].to(device),
          attention_mask=data['attention_mask'].to(device),
          token_type_ids=data['token_type_ids'].to(device),
        )
      else:
        outputs = model(
          input_ids=data['input_ids'].to(device),
          attention_mask=data['attention_mask'].to(device),
        )
    logits = outputs[0]
    prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
    logits = logits.detach().cpu().numpy()
    result = np.argmax(logits, axis=-1)
    index = data['index']

    output_pred.append(result)
    output_prob.append(prob)
    output_index.append(index)
  return np.concatenate(output_pred).tolist(), np.concatenate(output_prob, axis=0).tolist(), \
    np.concatenate(output_index).tolist()

def inference_binary_classifier(args):
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  Tokenizer_NAME = args.binary_tokenizer
  MODEL_NAME = args.binary_save_dir
  BATCH_SIZE = args.binary_bsz

  tokenizer = AutoTokenizer.from_pretrained(Tokenizer_NAME)

  model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
  model.parameters
  model.to(device)

  test_dataset_dir = "/opt/ml/dataset/test/test_data.csv"
  test_id, test_dataset, test_label = load_test_dataset(test_dataset_dir, tokenizer, Tokenizer_NAME)
  
  binary_test_label = []
  for label in test_label:
    if label != 0:
      binary_test_label.extend([1])
    else:
      binary_test_label.extend([0])
  # print(len(test_dataset['ids']))
  Re_binary_test_dataset = RE_binary_Dataset(test_dataset, binary_test_label, range(0, len(binary_test_label)))
  # print(Re_binary_test_dataset[-1])
  pred_answer, output_prob, output_index = inference_binary(model, Tokenizer_NAME, Re_binary_test_dataset, device, BATCH_SIZE)
  # print(len(output_index), output_index[0]) # 7765, 0
  # pred_answer = binary_num_to_label(pred_answer)
  # print(len(pred_answer)) # 7765, [:10], output_prob[:10], output_index[:10])
  # result_dict = {index:label for index, label in zip(output_index, pred_answer)}
  result_dict = {index:label for index, label in zip(output_index, pred_answer) if label == 1}
  print(result_dict)
  return result_dict
