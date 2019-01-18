import pickle

def save_embeddings(filename, embeddings, dictionary):
    """Embeddings and reverse dictionary serialization and dump to a file."""
    data = {
        'emb':  embeddings,
        'dict': dictionary
    }
    file = open(filename, 'wb')
    print("Saving embeddings to file:", filename)
    pickle.dump(data, file)

class Word2Vec(object):
    """Inference interface of Word2Vec embeddings
        Before inference the embdedding result of a word, data need to be initialized
        by calling method from_file or from_object.
    """
    def __init__(self):
        self.embeddings = None
        self.dictionary = None

    def from_file(self, filename):
        file = open(filename, 'rb')
        data = pickle.load(file)
        self.embeddings = data['emb']
        self.dictionary = data['dict']

    def from_object(self, embeddings, dictionary):
        self.embeddings = embeddings
        self.dictionary = dictionary

    def inference(self, word):
        assert self.embeddings is not None and self.dictionary is not None,\
            'Embeddings not initialized, use from_file or from_object to load data.'
        word_idx = self.dictionary.get(word)
        # Unknown word returns UNK's word_idx
        if word_idx is None:
            word_idx = 0
        return self.embeddings[word_idx]
