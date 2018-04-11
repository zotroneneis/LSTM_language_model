You can train two different kind of word embeddings:
1. word2vec (skip-gram or CBOW)
2. fasttext

The corresponding scripts can be run with

*python word2vec.py*

*python fasttext.py*

Note: If you want to initialize a model with pre-trained embeddings, you will have to transform the data to word ID's using the word2vec/fastText vocabulary.
To transform the data, use the functions provided in the file *src/data/preprocess_ptb.py*
