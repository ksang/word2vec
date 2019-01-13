import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

class SkipGramModel(torch.nn.Module):
    def __init__(self, vocabulary_size, embedding_dims):
        """
        u_embeddings - center words
        v_embeddings - context words
        """
        super(SkipGramModel, self).__init__()
        self.vocabulary_size = vocabulary_size
        self.embedding_dims = embedding_dims
        self.u_embeddings = torch.nn.Embedding(vocabulary_size, embedding_dims, sparse=True)
        self.v_embeddings = torch.nn.Embedding(vocabulary_size, embedding_dims, sparse=True)
        initrange = 0.5 / embedding_dims
        self.u_embeddings.weight.data.uniform_(-initrange, initrange)
        self.v_embeddings.weight.data.uniform_(-initrange, -initrange)
        self.W1 = torch.nn.Embedding(vocabulary_size, embedding_dims, sparse=True)
        self.W1 = Variable(torch.randn(embedding_dims, vocabulary_size).float(), requires_grad=True)
        self.W2 = Variable(torch.randn(vocabulary_size, embedding_dims).float(), requires_grad=True)
        """
    def forward(self, batch, labels):
        emb_u = self.u_embeddings(batch)
        emb_v = self.v_embeddings(labels)
        score = torch.mul(emb_u, emb_v).squeeze()
        score = torch.sum(score, dim=1)
        score = F.logsigmoid(score)
        return -1 * torch.sum(score)
        """

    def forward(self, batch, labels):
        x = torch.zeros(self.vocabulary_size).float()
        x[data] = 1.0
        h1 = torch.matmul(self.W1, x)
        y = torch.matmul(self.W2, h1)
        y_pred = F.log_softmax(y, dim=0)
        y_true = Variable(torch.from_numpy(np.array(label)).long())
        return F.nll_loss(y_pred.view(1,-1), y_true)

    def get_embeddings(self):
        #return self.u_embeddings.weight.data
        return self.W1.t().data

class CBOWModel(torch.nn.Module):
    def __init__(self, vocabulary_size, embedding_dim):
        super(CBOWModel, self).__init__()
        self.vocabulary_size = vocabulary_size
        self.embedding_dim = embedding_dim
        self.embeddings = torch.nn.Embedding(vocabulary_size, embedding_dim)
        self.linear1 = torch.nn.Linear(embedding_dim, vocabulary_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        add_embeds = torch.sum(embeds, dim=1)
        out = self.linear1(add_embeds)
        log_probs = F.log_softmax(out)
        return log_probs

    def get_embeddings(self):
        return self.embeddings.weight.data
