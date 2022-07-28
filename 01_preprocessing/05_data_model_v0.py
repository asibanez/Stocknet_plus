'''
- Removes entries within the return upper and lower thresholds
- Splits train-dev-test
- Generates labels
'''

# %% Imports
import os
import pandas as pd
import matplotlib.pyplot as plt

# %% Path definition
data_folder = 'C:/Users/siban/Dropbox/BICTOP/MyInvestor/06_model/02_NLP/06_stocknet/00_data/01_preprocessed'
input_filename = '04_BERT_encoded.pkl'
train_output_filename = 'model_train.pkl'
dev_output_filename = 'model_dev.pkl'
test_output_filename = 'model_test.pkl'

# %% Global initialization
return_high = 0.0055  # keep greater
return_low = -0.005   # keep equal or smaller

train_end_date = '2015-08-01'
dev_end_date =  '2015-10-01'
test_end_date = '2016-01-01'

# %% Load datasets
data_df = pd.read_pickle(os.path.join(data_folder, input_filename))

# %% Preprocess and EDA
_ = plt.hist(data_df.Return_n, bins = 100, range=(-0.2, 0.2))
plt.show()
print(f'Original dataset shape = {data_df.shape}')

# %% Remove entries within the thresholds
data_df = data_df[~((data_df.Return_n <= return_high) \
                    & (data_df.Return_n > return_low))]
_ = plt.hist(data_df.Return_n, bins = 100, range=(-0.2, 0.2))
plt.show()
print(f'Dataset shape after removing middle values = {data_df.shape}')

# %% Generate labels
labels = [1 if x > return_high else 0 for x in data_df.Return_n]
data_df.Return_n = labels

# %% Split into train-dev-test
data_df = data_df.sort_values(by='Date')
train_df = data_df[data_df.Date < train_end_date]
dev_df = data_df[(data_df.Date >= train_end_date) & \
                 (data_df.Date < dev_end_date)]
test_df = data_df[(data_df.Date >= dev_end_date) & \
                 (data_df.Date < test_end_date)]

print(f'Train set shape = {train_df.shape}')
print(f'Dev set shape = {dev_df.shape}')
print(f'Test set shape = {test_df.shape}')
    
# %% Save results
train_df.to_pickle(os.path.join(data_folder, train_output_filename))
dev_df.to_pickle(os.path.join(data_folder, dev_output_filename))
test_df.to_pickle(os.path.join(data_folder, test_output_filename))
