# word2vec

### About

Word2vec is a group of related models that are used to produce word embeddings. You may find original paper [here](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf).

The motivation of this project is to provide meaningful semantic and syntactic information as reinforcement learning observations when playing text based games.  

In this project, two models: CBOW and Skipgram are implemented using pytorch.

![model archtecture](/imgs/model.jpg "model")

[figure source](http://www.cs.nthu.edu.tw/~shwu/courses/ml/labs/10_Keras_Word2Vec/10_Keras_Word2Vec.html)

Training module also provides t-SNE plotting of a subset of vocabulary embeddings, an example of plotting:

![tsne](/imgs/tsne.png "tsne")

### Usage

##### Training

    usage: word2vec.py [-h] [-d DATA] [-o OUTPUT] [-p PLOT] [-pn PLOT_NUM]
                       [-s SIZE] [-m {CBOW,skipgram}] [-bs BATCH_SIZE]
                       [-ns NUM_SKIPS] [-sw SKIP_WINDOW] [-ed EMBEDDING_DIM]
                       [-lr LEARNING_RATE] [-i NUM_STEPS] [-ne NEGATIVE_EXAMPLE]
                       [-dc]

    optional arguments:
      -h, --help            show this help message and exit
      -d DATA, --data DATA  Data file for word2vec training.
      -o OUTPUT, --output OUTPUT
                            Output embeddings filename.
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
      -ne NEGATIVE_EXAMPLE, --negative_example NEGATIVE_EXAMPLE
                            Number of negative examples.
      -c CLIP, --clip CLIP  Clip gradient norm value.
      -dc, --disable_cuda   Explicitly disable cuda and GPU.

##### Inference

    Python 3.6.5 (default, Jun 17 2018, 12:13:06)
    [GCC 4.2.1 Compatible Apple LLVM 9.1.0 (clang-902.0.39.2)] on darwin
    Type "help", "copyright", "credits" or "license" for more information.
    >>> from inference import Word2Vec
    >>> wv = Word2Vec()
    >>> wv.from_file('embeddings.bin')
    >>> wv.inference('is')
    tensor([-0.8292,  0.3417, -0.3445, -0.6491, -0.3434,  1.0455,  0.1523, -1.1553])
    >>> wv.inference('of')
    tensor([-0.7774,  1.6229, -0.6826, -1.4431, -0.2579,  0.4803,  0.1727, -0.7641])
    >>>
