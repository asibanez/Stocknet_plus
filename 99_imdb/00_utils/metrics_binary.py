#v1 -> Metrics adapted to scores instead of logits

# Computes metrics

#%% Imports
import os
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score, roc_curve, classification_report

#%% Function definitions
# Sigmoid
def sigmoid(z):
    return 1/(1 + np.exp(-z))

#%% Metrics 
def compute_metrics(Y_ground_truth, Y_pred_binary, Y_pred_score):
    tn, fp, fn, tp = confusion_matrix(Y_ground_truth, Y_pred_binary).ravel()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    auc = roc_auc_score(Y_ground_truth, Y_pred_score)
    
    return precision, recall, f1, auc

#%% Path definitions
base_path = 'C:/Users/siban/Dropbox/BICTOP/MyInvestor/06_model/02_NLP/06_stocknet/00_data_imbd/02_runs/00_TEST_0_no_att_masks'

#%% Global initialization
random.seed(1234)
threshold = 0.5

#%% Read data json
train_results_path = os.path.join(base_path, 'train_results.json')
test_results_path = os.path.join(base_path, 'test_results.json')

with open(train_results_path) as fr:
    train_results = json.load(fr)
    
with open(test_results_path) as fr:
    test_results = json.load(fr)
    
#%% Extract results
Y_pred_scores = test_results['Y_pred']
Y_ground_truth = test_results['Y_gr_truth']

#%% Print ground truth class balance
ratio_0 = Y_ground_truth.count(0) / len(Y_ground_truth)
ratio_1 = Y_ground_truth.count(1) / len(Y_ground_truth)

print('\nTest set distribution')
print(f'Ratio class 0 = {ratio_0 * 100:.2f}%')
print(f'Ratio class 1 = {ratio_1 * 100:.2f}%')

#%% Compute binary results
Y_pred_binary = [round(x) for x in Y_pred_scores]
value_counts = pd.value_counts(Y_pred_binary)
print('\nPrediction distribution')
print(f'Ratio class 0 = {value_counts[0] / (value_counts[0] + value_counts[1]):.2f}%')
print(f'Ratio class 1 = {value_counts[1] / (value_counts[0] + value_counts[1]):.2f}%')

#%% Generate random results
random_pred_score = []

for i in range(0, len(Y_pred_scores)):
    random_pred_score.append(random.random())
    
random_pred_binary = [1 if x >= threshold else 0 for x in random_pred_score]

#%% Print results    
print(f'\n{classification_report(Y_ground_truth, Y_pred_binary)}')

#%% Compute metrics
precision, recall, f1, auc = compute_metrics(Y_ground_truth,
                                             Y_pred_binary,
                                             Y_pred_scores)

print(f'\nPrecision =\t{precision:.2f}')
print(f'Recall =\t{recall:.2f}')
print(f'F1 =\t\t{f1:.2f}')
print(f'AUC =\t\t{auc:.2f}')

#%% Plot ROC curve
fpr_model, tpr_model, threshold_roc_model = roc_curve(Y_ground_truth, Y_pred_scores)
fpr_rand, tpr_rand, threshold_roc_rand = roc_curve(Y_ground_truth, random_pred_score)
plt.plot(fpr_model, tpr_model, linestyle='--', label='Model')
plt.plot(fpr_rand, tpr_rand, linestyle=':', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc = 'lower right')
plt.ylim(-0.05, 1.05)
plt.grid()
plt.show()    

#%% Plot Precision - recall curve
precision_model_g, recall_model_g, threshold_model_g = precision_recall_curve(Y_ground_truth, Y_pred_scores)
precision_rand_g, recall_rand_g, threshold_rand_g = precision_recall_curve(Y_ground_truth, random_pred_score)
plt.plot(recall_model_g, precision_model_g, linestyle='--', label='Model')
plt.plot(recall_rand_g, precision_rand_g, linestyle=':', label='Random')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc = 'upper right')
plt.ylim(-0.05, 1.05)
plt.grid()
plt.show()

#%% Plot learning curves
plt.plot(train_results['training_loss'], label = 'train')
plt.plot(train_results['validation_loss'], label = 'val')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.legend(loc = 'upper right')
#plt.ylim(0.05, 0.75)
plt.grid()
plt.show()

#%%
plt.plot(train_results['training_acc'], label = 'train')
plt.plot(train_results['validation_acc'], label = 'val')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc = 'upper right')
plt.grid()
plt.show()

