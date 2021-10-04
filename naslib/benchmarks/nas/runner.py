import logging
import sys
#from nasbench import api

from naslib.defaults.trainer import Trainer
from naslib.optimizers import RandomSearch, Npenas, \
RegularizedEvolution, LocalSearch, Bananas, BasePredictor, DARTSOptimizer, DrNASOptimizer, GDASOptimizer

from naslib.search_spaces import NasBench101SearchSpace, NasBench201SearchSpace, \
DartsSearchSpace, NasBenchNLPSearchSpace, TransBench101SearchSpaceMicro, TransBench101SearchSpaceMacro
from naslib.utils import utils, setup_logger, get_dataset_api

from naslib.search_spaces.transbench101.loss import SoftmaxCrossEntropyWithLogits

config = utils.get_config_from_args(config_type='nas')

logger = setup_logger(config.save + "/log.log")
logger.setLevel(logging.INFO)

utils.log_args(config)

supported_optimizers = {
    'rs': RandomSearch(config),
    're': RegularizedEvolution(config),
    'bananas': Bananas(config),
    'npenas': Npenas(config),
    'ls': LocalSearch(config),
    'darts': DARTSOptimizer(config),
    'drnas': DrNASOptimizer(config),
    'gdas': GDASOptimizer(config),
}

supported_search_spaces = {
    'nasbench101': NasBench101SearchSpace(),
    'nasbench201': NasBench201SearchSpace(),
    'darts': DartsSearchSpace(),
    'nlp': NasBenchNLPSearchSpace(),
    'transbench101_micro': TransBench101SearchSpaceMicro(config.dataset),
    'transbench101_macro': TransBench101SearchSpaceMacro()
}

dataset_api = get_dataset_api(config.search_space, config.dataset)
utils.set_seed(config.seed)

search_space = supported_search_spaces[config.search_space]

optimizer = supported_optimizers[config.optimizer]
optimizer.adapt_search_space(search_space, dataset_api=dataset_api)
# optimizer.adapt_search_space(search_space)
 
import torch

if config.dataset in ['class_object', 'class_scene']:
    optimizer.loss = SoftmaxCrossEntropyWithLogits()
elif config.dataset == 'autoencoder':
    optimizer.loss = torch.nn.L1Loss()
    

trainer = Trainer(optimizer, config, lightweight_output=True)

trainer.search(resume_from="")
trainer.evaluate(resume_from="", dataset_api=dataset_api)
# trainer.evaluate(resume_from="")
