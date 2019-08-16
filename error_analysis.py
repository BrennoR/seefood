import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import load_model
from sklearn.metrics import confusion_matrix
sns.set()

# model = load_model('food101_inceptionv3.h5')

test_results = pd.read_csv('results_inceptionv3.csv')
test_results['Class'] = test_results['Filename'].apply(lambda x: x.split('/')[0])

conf_mx = confusion_matrix(test_results['Class'], test_results['Predictions'])
row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums
np.fill_diagonal(norm_conf_mx, 0)

plt.figure(figsize=(20, 15))
sns.heatmap(norm_conf_mx)
plt.show()

print(norm_conf_mx)
