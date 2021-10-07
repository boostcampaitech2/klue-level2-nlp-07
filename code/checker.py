from collections import Counter
import pandas as pd

a = pd.read_csv('/opt/ml/blending/binary.csv')
print(Counter(a['pred_label']))
