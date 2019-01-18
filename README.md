# word2vec

### About

Word2vec is a group of related models that are used to produce word embeddings. You may find original paper [here](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf).

The motivation of this project is to provide meaningful semantic and syntactic information as reinforcement learning observations when playing text based games.  In this project, two models: CBOW and Skipgram are implemented using pytorch.

![model archtecture](/imgs/model.jpg "model")

[figure source](http://www.cs.nthu.edu.tw/~shwu/courses/ml/labs/10_Keras_Word2Vec/10_Keras_Word2Vec.html)

Training module also provides t-SNE plotting of a subset of vocabulary embeddings, an example of plotting:

![tsne](/imgs/tsne.png "tsne")

### Usage

##### Training

    usage: word2vec.py [-h] [-d DATA] [-o OUT] [-p PLOT] [-pn PLOT_NUM] [-s SIZE]
                       [-m {CBOW,skipgram}] [-bs BATCH_SIZE] [-ns NUM_SKIPS]
                       [-sw SKIP_WINDOW] [-ed EMBEDDING_DIM] [-lr LEARNING_RATE]
                       [-i NUM_STEPS] [-n NUM_SAMPLED]

    optional arguments:
      -h, --help            show this help message and exit
      -d DATA, --data DATA  Data file for word2vec training.
      -o OUT, --out OUT     Output filename.
      -p PLOT, --plot PLOT  Plotting output filename.
      -pn PLOT_NUM, --plot_num PLOT_NUM
                            Plotting data number.
      -s SIZE, --size SIZE  Vocabulary size.
      -m {CBOW,skipgram}, --mode {CBOW,skipgram}
                            Training model.
      -bs BATCH_SIZE, --batch_size BATCH_SIZE
                            Training batch size.
      -ns NUM_SKIPS, --num_skips NUM_SKIPS
                            How many times to reuse an input to generate a label.
      -sw SKIP_WINDOW, --skip_window SKIP_WINDOW
                            How many words to consider left and right.
      -ed EMBEDDING_DIM, --embedding_dim EMBEDDING_DIM
                            Dimension of the embedding vector.
      -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                            Learning rate
      -i NUM_STEPS, --num_steps NUM_STEPS
                            Number of steps to run.
