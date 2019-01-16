import torch
from torch.autograd import Variable
import torch.nn.functional as F

class SkipGramModel(torch.nn.Module):
    def __init__(self, vocabulary_size, embedding_dim):
        super(SkipGramModel, self).__init__()
        self.vocabulary_size = vocabulary_size
        self.embedding_dim = embedding_dim
        self.embeddings = torch.nn.Embedding(vocabulary_size, embedding_dim)
        initrange = 0.5 / embedding_dim
        self.embeddings.weight.data.uniform_(-initrange, initrange)

    def forward(self, centers, contexts):
        u_embeds = self.embeddings(centers).view(len(centers),1,-1)
        v_embeds = self.embeddings(contexts).view(len(contexts),1,-1)
        score  = torch.bmm(u_embeds, v_embeds.transpose(1,2)).squeeze()
        score = F.logsigmoid(score).squeeze()
        return -1 * score.sum()

    def get_embeddings(self):
        return self.embeddings.weight.data

class CBOWModel(torch.nn.Module):
    def __init__(self, vocabulary_size, embedding_dim):
        super(CBOWModel, self).__init__()
        self.vocabulary_size = vocabulary_size
        self.embedding_dim = embedding_dim
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
