import logging

from naslib.defaults.trainer import Trainer
from naslib.optimizers import (
    RandomSearch,
    Npenas,
    RegularizedEvolution,
    LocalSearch,
    Bananas
)

from naslib.search_spaces.core.query_metrics import Metric
from naslib.search_spaces import (
    NasBench101SearchSpace,
    NasBench201SearchSpace,
    NasBench301SearchSpace,
    NasBenchNLPSearchSpace,
    TransBench101SearchSpaceMicro,
    TransBench101SearchSpaceMacro,
    NasBenchASRSearchSpace
)

from naslib import utils
from naslib.utils import setup_logger, get_dataset_api, get_zc_benchmark_api

from torch.utils.tensorboard import SummaryWriter

from fvcore.common.config import CfgNode

# config_dict = {
#     'search_space': 'nasbench201',
#     'dataset': 'cifar10',
#     'seed': 0,
#     'save': 'test',
#     'search': {
#         'seed': 0,
#         'acq_fn_optimization': 'random_sampling',
#         'use_zc_api': True,
#         'fidelity': 108,
#         'epochs': 10,
#         'sample_size': 10,
#         'population_size': 30,
#         'k': 10,
#         'num_init': 10,
#         'num_ensemble': 3,
#         'predictor_type': 'var_sparse_gp',
#         'acq_fn_type': 'its',
#         'encoding_type': 'adjacency_one_hot',
#         'num_arches_to_mutate': 1 ,
#         'max_mutations': 1,
#         'num_candidates': 50,
#         'checkpoint_freq': 1000
#     },
#     'optimizer': 'npenas'
# }

# config = CfgNode(config_dict)

config = utils.get_config_from_args(config_type="zc")

zc_api = get_zc_benchmark_api(config.search_space, config.dataset)

supported_optimizers = {
    'rs': RandomSearch(config),
    're': RegularizedEvolution(config),
    'bananas': Bananas(config, zc_api=zc_api),
    'npenas': Npenas(config, zc_api=zc_api),
    'ls': LocalSearch(config),
}

supported_search_spaces = {
    'nasbench101': NasBench101SearchSpace(),
    'nasbench201': NasBench201SearchSpace(),
    'nasbench301': NasBench301SearchSpace(),
    'nlp': NasBenchNLPSearchSpace(),
    'transbench101_micro': TransBench101SearchSpaceMicro(config.dataset),
    'transbench101_macro': TransBench101SearchSpaceMacro(),
    'asr': NasBenchASRSearchSpace()
}

dataset_api = get_dataset_api(config.search_space, config.dataset)

search_space = supported_search_spaces[config.search_space]

search_space.labeled_archs = [eval(arch) for arch in zc_api.keys()]

if config.search.acq_fn_optimization == 'random_sampling' and config.search.use_zc_api == True:
    search_space.instantiate_model = False

if config.search.acq_fn_optimization != 'random_sampling' and config.search.use_zc_api == True:
    logging.warning("Using ZC API with a acquisition optimization strategy that is not Random Sampling")

metric = Metric.VAL_ACCURACY if config.search_space == 'nasbench301' else None


optimizer = supported_optimizers[config.optimizer]
optimizer.adapt_search_space(search_space, dataset_api=dataset_api)

trainer = Trainer(optimizer, config, lightweight_output=True)

trainer.search(resume_from="", summary_writer=None, report_incumbent=False)
trainer.evaluate(resume_from="", dataset_api=dataset_api, metric=metric)
