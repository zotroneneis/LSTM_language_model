import json
import os
import argparse

import tensorflow as tf
import yaml
import logging
logger = logging.getLogger()

logging.basicConfig(filename='log_languageModel.txt', level=logging.DEBUG, format=' %(asctime)s - %(levelname)s - %(message)s')

from models import make_model

def main():
    number_suggestions = args.number_suggestions
    length_suggestions = args.length_suggestions

    with open("config_ptb.yaml", 'r') as configfile:
        config = yaml.load(configfile)

    tf.logging.set_verbosity(config['general']['logging'])

    model_name = config['general']['model_name']
    tf.logging.info('Initializing the model: {}'.format(model_name))

    model = make_model(config)

    if config['train']:
        tf.logging.info('Training {}'.format(model_name))
        model.train()

    if config['predict']:
        primer_words = args.primer_words.lower().split()
        assert primer_words, 'At least one primer word must be provided!'
        prediction = model.predict(primer_words, number_suggestions, length_suggestions)

        for i in range(len(prediction)):
            tf.logging.info('Predicted sentence {}: {}'.format(i+1, prediction[i]))

    if config['test']:
        tf.logging.info('Testing')
        model.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Language Model")

    parser.add_argument('-p', '--primer_words', required=False, help='Wörter, mit denen das Modell gestartet wird')
    parser.add_argument('-n', '--number_suggestions', required=False, nargs='?', const=5, type=int, default=5, help='Anzahl der vorgeschlagenen Wörter')
    parser.add_argument('-l', '--length_suggestions', required=False, nargs='?', const=3, type=int, default=3, help='Länge der Vorschläge')

    args = parser.parse_args()
    main()
