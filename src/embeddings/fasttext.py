import os
import pickle
import ipdb
import logging
from gensim.models.fasttext import FastText
from word2vec import prepareEmbeddings

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO, format=' %(asctime)s - %(levelname)s - %(message)s')

def fastText(root, sentences, embedding_size=200, model_type=0, min_count=1, workers=4):
    """
    Trains and saves a fastText model on a given list of sentences

    Params:
        sentences: list of tokenized sentences
        embedding_size: size of the embeddings
        model_type: int, 0: CBOW, 1: skip-gram
        min_count:
        worker: number of workers for parallelization

    The saved file stores the model in a format compatible with the original word2vec implementation.

    https://groups.google.com/forum/#!topic/gensim/RhiwLU0vP1A

    """
    logger.info("Train the fasttext model")
    model = FastText(sentences, min_count=min_count, workers=workers, sg=model_type, size=embedding_size)

    logger.info("Save the trained model")
    model.save(root+'fasttext_model')

    logger.info("Save the model in word2vec format")
    fname = 'fasttext_word2vecFormat'
    model.wv.save_word2vec_format(root+fname)

if __name__=='__main__':

    root = os.path.expanduser('~/lstmLanguageModel/data/ptb/embeddings/fastText/')
    tokens = os.path.expanduser('~/lstmLanguageModel/data/ptb/embeddings/tokenized_sentences')

    logger.info("Read in sentences")
    with open(tokens, 'rb') as f:
        tokenized_sentences = pickle.load(f)

    ipdb.set_trace()

    fastText(root, tokenized_sentences, embedding_size=200)
    fastText_model = os.path.join(root, 'fasttext_word2vecFormat')

    vocab_dict, embedding_index, embedding_matrix = prepareEmbeddings(fastText_model, embedding_size=200)

    logger.info("Save word dictionary and embedding matrix to file")
    save_vocab = os.path.join(root, 'fastText_vocab_dict')
    with open(save_vocab, 'wb') as f:
        pickle.dump(vocab_dict, f)

    save_embeddings = os.path.join(root, 'fastText_embeddingMatrix')
    with open(save_embeddings, 'wb') as f:
        pickle.dump(embedding_matrix, f)

    logger.info("End of program")
