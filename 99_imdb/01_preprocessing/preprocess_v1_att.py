# %% Imports
import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

# %% Function definitions
def BERT_encode_f(bert_tokenizer, text, seq_len):
    BERT_encoding = bert_tokenizer(text,
                                   return_tensors='pt',
                                   padding='max_length',
                                   truncation=True,
                                   max_length=seq_len)

    return BERT_encoding

# %% Path definitions
input_path = 'C:/Users/siban/Dropbox/BICTOP/MyInvestor/06_model/02_NLP/06_stocknet/00_1_data_imbd/00_raw/IMDB Dataset.csv'
output_folder ='C:/Users/siban/Dropbox/BICTOP/MyInvestor/06_model/02_NLP/06_stocknet/00_1_data_imbd/02_preprocessed_att_mask'

# %% Global initialization
model_name = 'bert-base-uncased'
seq_len = 512
random_seed = 1234

# %% Data loading
data_df = pd.read_csv(input_path)

# %% EDA labels
print(pd.value_counts(data_df.sentiment))

# %% EDA reviews
tokens = [x.split(' ') for x in data_df.review]
lens = [len(x) for x in tokens]
_ = plt.hist(lens, bins=50)

# %% Preprocess labels
labels = [1 if x == 'positive' else 0 for x in data_df.sentiment]
data_df.sentiment = labels

# %% Preprocess reviews
bert_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
bert_token_ids = []
bert_att_masks = []
for review in tqdm(data_df.review, desc = 'Encoding BERT'):
    bert_token_id = BERT_encode_f(bert_tokenizer,
                                  review, seq_len)['input_ids']
    bert_att_mask = BERT_encode_f(bert_tokenizer,
                                  review, seq_len)['attention_mask']
    bert_token_ids.append(bert_token_id)
    bert_att_masks.append(bert_att_mask)

data_df['bert_token_ids'] = bert_token_ids
data_df['bert_att_mask'] = bert_att_masks

# %% Train - test split
model_train, model_test = train_test_split(data_df,
                                           test_size=0.2,
                                           shuffle=False,
                                           random_state=random_seed)

# %% Verify dataset sizes
print(f'\nShape full set = {data_df.shape}')
print(f'Shape train set = {model_train.shape}\t{len(model_train)/len(data_df)*100}%')
print(f'Shape test set = {model_test.shape}\t\t{len(model_test)/len(data_df)*100}%')

# %% Save results
if not(os.path.isdir(output_folder)):
    os.mkdir(output_folder)
    print(f'Created folder: {output_folder}')

model_train.to_pickle(os.path.join(output_folder, 'model_train.pkl'))
model_test.to_pickle(os.path.join(output_folder, 'model_test.pkl'))
print(f'Outputs saved to: {output_folder}')