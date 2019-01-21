import torch
import torch.nn.functional as F
import numpy as np

TABLE_SIZE = 1e8

def create_sample_table(word_count):
    """ Create negative sample table for vocabulary, words with
        higher frequency will have higher occurrences in table.
    """
    table = []
    frequency = np.power(np.array(word_count), 0.75)
    sum_frequency = sum(frequency)
    ratio = frequency / sum_frequency
    count = np.round(ratio * TABLE_SIZE)
    for word_idx, c in enumerate(count):
        table += [word_idx] * int(c)
    return np.array(table)

class SkipGramModel(torch.nn.Module):
    """ Center word as input, context words as target.
        Objective is to maximize the score of map from input to target.
    """
    def __init__(self, device, vocabulary_size, embedding_dim, neg_num=0, word_count=[]):
        super(SkipGramModel, self).__init__()
        self.device = device
        self.neg_num = neg_num
        self.embeddings = torch.nn.Embedding(vocabulary_size, embedding_dim)
        initrange = 0.5 / embedding_dim
        self.embeddings.weight.data.uniform_(-initrange, initrange)
        if self.neg_num > 0:
            self.table = create_sample_table(word_count)

    def forward(self, centers, contexts):
        batch_size = len(centers)
        u_embeds = self.embeddings(centers).view(batch_size,1,-1)
        v_embeds = self.embeddings(contexts).view(batch_size,1,-1)
        score  = torch.bmm(u_embeds, v_embeds.transpose(1,2)).squeeze()
        loss = F.logsigmoid(score).squeeze()
        if self.neg_num > 0:
            neg_contexts = torch.LongTensor(np.random.choice(self.table, size=(batch_size, self.neg_num))).to(self.device)
            neg_v_embeds = self.embeddings(neg_contexts)
            neg_score = torch.bmm(u_embeds, neg_v_embeds.transpose(1,2)).squeeze()
            neg_score = torch.sum(neg_score, dim=1)
            neg_score = F.logsigmoid(-1*neg_score).squeeze()
            loss += neg_score
        return -1 * loss.sum()

    def get_embeddings(self):
        return self.embeddings.weight.data

class CBOWModel(torch.nn.Module):
    """ Context words as input, returns possiblity distribution
        prediction of center word (target).
    """
    def __init__(self, device, vocabulary_size, embedding_dim):
        super(CBOWModel, self).__init__()
        self.device = device
        self.embeddings = torch.nn.Embedding(vocabulary_size, embedding_dim)
        initrange = 0.5 / embedding_dim
        self.embeddings.weight.data.uniform_(-initrange, initrange)
        self.linear1 = torch.nn.Linear(embedding_dim, vocabulary_size)

    def forward(self, contexts):
        # input
        embeds = self.embeddings(contexts)
        # projection
        add_embeds = torch.sum(embeds, dim=1)
        # output
        out = self.linear1(add_embeds)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

    def get_embeddings(self):
        return self.embeddings.weight.data
