'''
XXXXX
'''

# Imports
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoModel


# %% DataClass definition
class stocknet_dataset(Dataset):
    def __init__(self, data_df):
        self.X = torch.stack(list(data_df.bert_token_ids)).squeeze(1)
        self.Y = torch.tensor(data_df.sentiment.values).float()

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        X = self.X[idx]
        Y = self.Y[idx]

        return X, Y


# %% Model definition
class stocknet_model(nn.Module):

    def __init__(self, args):
        super(stocknet_model, self).__init__()

        self.h_dim = args.hidden_dim
        self.dropout = args.dropout

        # Bert layer
        self.bert_model = AutoModel.from_pretrained(args.model_name)
        # Freeze bert parameters
        if eval(args.freeze_BERT) == True:
            for parameter in self.bert_model.parameters():
                parameter.requires_grad = False

        # Fully connected output
        self.fc_out = nn.Linear(in_features=self.h_dim, out_features=1)

        # Sigmoid
        self.sigmoid = nn.Sigmoid()

        # Dropout
        self.drops = nn.Dropout(self.dropout)

        # Batch normalization
        self.bn1 = nn.BatchNorm1d(self.h_dim)

    def forward(self, X):

        # BERT encoder
        bert_input = {'input_ids': X}
        x = self.bert_model(**bert_input, output_hidden_states=True)     # XXXXX
        x = x['pooler_output']                                           # batch_size x h_dim

        # Binary classifier
        x = self.bn1(x)                                                  # batch_size x h_dim
        x = self.fc_out(x)                                               # batch_size x 1
        x = self.sigmoid(x)                                              # batch_size x 1
        x = x.squeeze(1)                                                 # batch_size

        return x
