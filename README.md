LSTM-based language model
==============================
***This repository is still under active development***

This repository contains all code and resources related to my master thesis on the topic

"Recurrent Neural Language Modeling - Using Transfer Learning to Perform Radiological Sentence Completion"

Abstract:
Motivated by the potential benefits of a system that accelerates the process of writing radiological reports, we present a Recurrent Neural Network Language Model for modeling radiological language. We show that recurrent neural language models can be used to produce convincing radiological reports and investigate how their performance can be improved by using advanced regularization and initialization techniques. Furthermore, we study the use of transfer learning to create topic-specific language models.

Data
==============================
The original data used in the thesis is confidential. Therefore, this repository features a version of the code that runs on the Penn Treebank dataset available [here](http:/www.fit.vutbr.cz/~imikolov/rnnlm/).

To run the code you will have to preprocess the data first. Further details on this can be found in the folder *src/data/*

Testable Features
==============================
By adapting the config file, the following features can be tested:
- Variational dropout of hidden layers
- Weight tying
- Embedding dropout
- Pre-trained embeddings

Thesis and Presentation
==============================
The thesis and slides can be found in the *reports* folder 


Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README 
    ├── data
    │   ├── raw            <- Original PTB files 
    │   ├── training_files <- Preprocessed PTB word ids
    │   └── embeddings     <- word embeddings
    │       │                 
    │       ├── fasttext
    │       └── word2vec
    │
    ├── models             <- Trained and serialized models
    │   ├── checkpoints    <- Model checkpoints
    │   └── tensorboard    <- Tensorboard logs
    │
    ├── reports            <- Thesis and presentation slides
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment
    │
    ├── src                <- Source code for use in this project.
    │   │
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── main.py        <- main file for training, testing, etc.
    │   │
    │   ├── config_ptb.yamp  <- config file, specifying model params
    │   │
    │   ├── data           <- scripts to preprocess data
    │   │   │                 
    │   │   ├── README.md
    │   │   ├── preprocess_ptb.py
    │   │   └── inverse_dict.py
    │   │
    │   ├── embeddings     <- scripts to train word embeddings
    │   │   │                 
    │   │   ├── README.md
    │   │   ├── fasttext.py
    │   │   └── word2vec.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make predictions
    │   │   │                 
    │   │   └── ptb_basic_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       │                 
    │       ├── README.md
    │       └── visualize_embeddings.py
    │
    └── 


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
