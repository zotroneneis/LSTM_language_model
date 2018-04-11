"""
File: inverse_dict.py
Author: Anna-Lena Popkes
Email: popkes@gmx.net
Github: https://github.com/zotroneneis
Description: Script for creating dictionary with inverse mapping
"""
import pickle
import ipdb
import os

dict_path = os.path.expanduser('~/lstmLanguageModel/data/ptb/raw/vocab_dict')
inverse_path = os.path.expanduser('~/lstmLanguageModel/data/ptb/raw/inverse_vocab_dict')

with open(dict_path, "rb") as f:
    vocab_dict = pickle.load(f)

inverse_dict = {v:k for k,v in vocab_dict.items()}

with open(inverse_path, "wb") as f:
    pickle.dump(inverse_dict, f)
