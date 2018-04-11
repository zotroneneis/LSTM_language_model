from models.ptb_basic_model import BasicLanguageModel

__all__ = [
    "BasicLanguageModel",
]

def make_model(config):
    if config['general']['model_name'] in __all__:
        return globals()[config['general']['model_name']](config)
    else:
        raise Exception('The model name %s does not exist' % config['general']['model_name'])


def get_model_class(config):
    if config['general']['model_name'] in __all__:
        return globals()[config['general']['model_name']]
    else:
        raise Exception('The model name %s does not exist' % config['general']['model_name'])
