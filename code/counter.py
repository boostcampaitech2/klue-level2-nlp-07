import pandas as pd
from collections import Counter

test = pd.read_csv('./prediction/final_real.csv')
counter = Counter(test['pred_label'])
print(counter)