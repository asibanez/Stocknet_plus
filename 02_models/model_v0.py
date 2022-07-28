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
        X_nm1 = torch.stack(list(data_df['Text_n_m_1'])).squeeze(1)
        X_nm2 = torch.stack(list(data_df['Text_n_m_2'])).squeeze(1)
        X_nm3 = torch.stack(list(data_df['Text_n_m_3'])).squeeze(1)
        X_nm4 = torch.stack(list(data_df['Text_n_m_4'])).squeeze(1)
        X_nm5 = torch.stack(list(data_df['Text_n_m_5'])).squeeze(1)

        self.X = torch.cat((X_nm1, X_nm2, X_nm3, X_nm4, X_nm5), dim=1)
        self.Y = torch.tensor(data_df['Return_n'].values).float()

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

        self.lookback_days = args.lookback_days
        self.h_dim = args.hidden_dim
        self.n_heads = args.n_heads
        self.seq_len = args.seq_len
        self.dropout = args.dropout

        # Bert layer
        self.bert_model = AutoModel.from_pretrained(args.model_name)
        # Freeze bert parameters
        # for parameter in self.bert_model.parameters():
        #    parameter.requires_grad = False

        # Transformer layer
        self.transf_enc = nn.TransformerEncoderLayer(d_model=self.h_dim,
                                                     nhead=self.n_heads,
                                                     batch_first=True)

        # Fully connected output
        self.fc_out = nn.Linear(in_features=self.lookback_days*self.h_dim,
                                out_features=1)

        # Sigmoid
        self.sigmoid = nn.Sigmoid()

        # Dropout
        self.drops = nn.Dropout(self.dropout)

        # Batch normalization
        self.bn1 = nn.BatchNorm1d(self.lookback_days*self.h_dim)

    def forward(self, X):
        batch_size = X.size()[0]
        device = X.get_device() if X.get_device() >= 0 else 'cpu'

        empty_par_ids = torch.cat([torch.tensor([101, 102]),
                                   torch.zeros(self.seq_len-2)]).long()  # seq_len
        empty_par_ids = empty_par_ids.repeat(batch_size, 1).to(device)   # batch_size x seq_len

        # Encode paragraphs - BERT & generate transfomers masks
        bert_out = {}
        transf_mask = torch.zeros((batch_size,
                                   self.lookback_days),
                                  dtype=torch.bool)                      # batch_size x lookback_days

        for idx in range(0, self.lookback_days):
            span_b = self.seq_len * idx
            span_e = self.seq_len * (idx + 1)

            # Slice sequence
            X_aux = X[:, span_b:span_e]                                  # batch_size x seq_len

            # Generate masks for transformer
            equiv = torch.eq(X_aux, empty_par_ids)                       # batch_size x seq_len
            equiv = equiv.all(dim=1)                                     # batch_size
            transf_mask[:, idx] = equiv                                  # batch_size

            # Generate input dict to bert model
            bert_input = {'input_ids': X_aux}                            # .long()

            # Compute bert output
            output = self.bert_model(**bert_input,
                                     output_hidden_states=True)
            bert_out[idx] = output['pooler_output'].unsqueeze(1)         # batch_size x 1 x h_dim

        x = torch.cat(list(bert_out.values()), dim=1)                    # batch_size x lookback_days x h_dim

        # Encode document - Transformer
        transf_mask = transf_mask.to(device)                             # batch_size x lookback_days
        x = self.transf_enc(x, src_key_padding_mask=transf_mask)         # batch_size x lookback_days x h_dim
        x = self.drops(x)                                                # batch_size x lookback_days x h_dim

        # Binary classifier
        x = x.reshape(-1, self.lookback_days*self.h_dim)                 # batch_size x (lookback_days x h_dim)
        x = self.bn1(x)                                                  # batch_size x (lookback_days x h_dim)
        x = self.fc_out(x)                                               # batch_size x 1
        x = self.sigmoid(x)                                              # batch_size x 1
        x = x.squeeze(1)                                                 # batch_size

        return x
