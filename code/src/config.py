import os

PATH_ROOT = os.path.abspath(os.path.join(os.path.split(__file__)[0], '../../'))
PATH_DATASETS = os.path.join(PATH_ROOT, 'datasets')
PATH_SAVE_MODELS = os.path.join(PATH_ROOT, 'models_data')
PATH_LOGS = os.path.join(PATH_ROOT, 'tensorboard_log')
