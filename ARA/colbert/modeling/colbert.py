import string
import torch
import torch.nn as nn

from transformers import BertPreTrainedModel, BertModel, BertTokenizerFast
from colbert.parameters import DEVICE


class ColBERT(BertPreTrainedModel):
    def __init__(self, config, query_maxlen, doc_maxlen, mask_punctuation, dim=128, similarity_metric='cosine'):

        super(ColBERT, self).__init__(config)

        self.query_maxlen = query_maxlen
        self.doc_maxlen = doc_maxlen
        self.similarity_metric = similarity_metric
        self.dim = dim

        self.mask_punctuation = mask_punctuation
        self.skiplist = {}

        if self.mask_punctuation:
            self.tokenizer = BertTokenizerFast.from_pretrained('./bert-base-uncased')
            self.skiplist = {w: True
                             for symbol in string.punctuation
                             for w in [symbol, self.tokenizer.encode(symbol, add_special_tokens=False)[0]]}

        self.bert = BertModel(config)
        self.linear = nn.Linear(config.hidden_size, dim, bias=False)
        self.gate = nn.Linear(dim, 1, bias=True)

        self.init_weights()
        

    def forward(self, Q, D, keyword=None, trianing=False):

        D_ = self.doc(*D)
        
        assert self.similarity_metric == 'l2'
        
        keyword = keyword.to(DEVICE)
        scores, kw_scores = self.score(self.query(*Q), D_, keyword=keyword, trianing=trianing)
        return scores, kw_scores


    def query(self, input_ids, attention_mask):
        input_ids, attention_mask = input_ids.to(DEVICE), attention_mask.to(DEVICE)
        Q = self.bert(input_ids, attention_mask=attention_mask)[0]
        Q = self.linear(Q)

        return torch.nn.functional.normalize(Q, p=2, dim=2)

    def doc(self, input_ids, attention_mask, keep_dims=True):
        input_ids, attention_mask = input_ids.to(DEVICE), attention_mask.to(DEVICE)
        D = self.bert(input_ids, attention_mask=attention_mask)[0]
        D = self.linear(D)

        mask = torch.tensor(self.mask(input_ids), device=DEVICE).unsqueeze(2).float()
        D = D * mask

        D = torch.nn.functional.normalize(D, p=2, dim=2)

        if not keep_dims:
            D, mask = D.cpu().to(dtype=torch.float16), mask.cpu().bool().squeeze(-1)
            D = [d[mask[idx]] for idx, d in enumerate(D)]

        return D

    def score(self, Q, D, keyword=None, trianing=False):
        if self.similarity_metric == 'cosine':
            q_token = (Q @ D.permute(0, 2, 1)).max(2).values
            q_gate = torch.sigmoid(self.gate(Q).squeeze())
            q_token_norm = q_gate * q_token
            if trianing: return q_token_norm.sum(1) * 16, q_gate
            return q_token_norm.sum(1) * 16

        assert self.similarity_metric == 'l2'
        
        q_token = (-1.0 * ((Q.unsqueeze(2) - D.unsqueeze(1))**2).sum(-1)).max(-1).values
        
        if keyword is None:
            q_gate = torch.sigmoid(self.gate(Q).squeeze())
            q_token_norm = q_gate * q_token
            if trianing: return q_token_norm.sum(-1) * 16, q_gate
            return q_token_norm.sum(-1) * 16
    
        q_token_norm = q_token * keyword
        return q_token.sum(-1), q_token_norm.sum(-1)


    def mask(self, input_ids):
        mask = [[(x not in self.skiplist) and (x != 0) for x in d] for d in input_ids.cpu().tolist()]
        return mask
