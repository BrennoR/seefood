import pandas as pd

test_results = pd.read_csv('results_inceptionv3.csv')
test_results['Class'] = test_results['Filename'].apply(lambda x: x.split('/')[0])
print("Test Accuracy: ", sum(test_results['Predictions'] == test_results['Class']) / len(test_results))
