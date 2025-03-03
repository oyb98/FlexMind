import os

import yaml
from pathlib import Path
# get absolute path of the current file
# __file__ is the path of the current file
# os.path.dirname(__file__) is the directory containing the current file
# os.path.abspath(os.path.dirname(__file__)) is the absolute path of the directory containing the current file

path = Path(os.path.abspath(os.path.dirname(__file__)))
def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        return config


def set_environment(config: dict) -> None:
    os.environ['model'] = config['llm']['model']
    os.environ['api_key'] = default(config['llm']['api_key'], '')
    os.environ['base_url'] = default(config['llm']['endpoint'], '')
    os.environ['openai'] = default(config['llm']['openai'], 'openai')



config = load_config( path.parent / 'env.yaml')
llm_config = {"config_list": [{"model": config['llm']['model'],
                               "api_key": 'sk-ypDOrct5sWOZEVCUYWmNqK8rsGXcJ6svajXmNrPdEeU5tshv',
                               'base_url': config['llm']['base_url'],
                               }]}

# MODEL ='gpt-3.5-turbo'  #config['llm']['model']
MODEL =  config['llm']['model']
# BASE_URL = "https://api.chatanywhere.tech" #"http://113.31.110.212:11002/v1" # config['llm']['base_url']
BASE_URL =  config['llm']['base_url']
API_KEY = "sk-SQ4abgeK6rbIXJYauqKtMO7FcSZ4azd0Un4LcXpupmx8uenw" #"XXXXX"



if __name__ == '__main__':
    print(config)
    print(llm_config)
