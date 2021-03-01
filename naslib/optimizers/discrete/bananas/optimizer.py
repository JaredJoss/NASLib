import collections
import logging
import torch
import copy
import numpy as np

from naslib.optimizers.core.metaclasses import MetaOptimizer
from naslib.optimizers.discrete.bananas.acquisition_functions import acquisition_function

from naslib.predictors.ensemble import Ensemble
from naslib.predictors.zerocost_estimators import ZeroCostEstimators

from naslib.search_spaces.core.query_metrics import Metric

from naslib.utils.utils import AttrDict, count_parameters_in_MB, get_train_val_loaders
from naslib.utils.logging import log_every_n_seconds


logger = logging.getLogger(__name__)


class Bananas(MetaOptimizer):
    
    # training the models is not implemented
    using_step_function = False
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.epochs = config.search.epochs

        self.performance_metric = Metric.VAL_ACCURACY
        self.dataset = config.dataset

        self.k = config.search.k
        self.num_init = config.search.num_init
        self.num_ensemble = config.search.num_ensemble
        self.predictor_type = config.search.predictor_type
        self.acq_fn_type = config.search.acq_fn_type
        self.acq_fn_optimization = config.search.acq_fn_optimization
        self.encoding_type = config.search.encoding_type # currently not implemented
        self.num_arches_to_mutate = config.search.num_arches_to_mutate
        self.max_mutations = config.search.max_mutations
        self.num_candidates = config.search.num_candidates
        self.max_zerocost = 100

        self.train_data = []
        self.next_batch = []
        self.history = torch.nn.ModuleList()

        self.zc = ('omni' in self.predictor_type)

    def adapt_search_space(self, search_space, scope=None, dataset_api=None):
        assert search_space.QUERYABLE, "Bananas is currently only implemented for benchmarks."
        
        self.search_space = search_space.clone()
        self.scope = scope if scope else search_space.OPTIMIZER_SCOPE      
        self.dataset_api = dataset_api
        if self.zc:
            self.train_loader, _, _, _, _ = get_train_val_loaders(self.config, mode='train')
    
    def new_epoch(self, epoch):

        if epoch < self.num_init:
            # randomly sample initial architectures 
            model = torch.nn.Module()   # hacky way to get arch and accuracy checkpointable
            model.arch = self.search_space.clone()
            model.arch.sample_random_architecture(dataset_api=self.dataset_api)
            model.accuracy = model.arch.query(self.performance_metric, self.dataset, dataset_api=self.dataset_api)
            if self.zc and len(self.train_data) <= self.max_zerocost:                
                zc_method = ZeroCostEstimators(self.config, batch_size=64, method_type='jacov')
                zc_method.train_loader = copy.deepcopy(self.train_loader)
                score = zc_method.query([model.arch])
                model.zc_score = np.squeeze(score)
            
            self.train_data.append(model)
            self._update_history(model)

        else:
            if len(self.next_batch) == 0:
                # train a neural predictor
                xtrain = [m.arch for m in self.train_data]
                ytrain = [m.accuracy for m in self.train_data]
                ensemble = Ensemble(num_ensemble=self.num_ensemble,
                                    ss_type=self.search_space.get_type(),
                                    predictor_type=self.predictor_type)
                zc_scores = None
                if self.zc and len(self.train_data) <= self.max_zerocost:
                    zc_scores = [m.zc_score for m in self.train_data]
                train_error = ensemble.fit(xtrain, ytrain, train_info=zc_scores)

                # define an acquisition function
                acq_fn = acquisition_function(ensemble=ensemble, 
                                              ytrain=ytrain,
                                              acq_fn_type=self.acq_fn_type)
                
                # optimize the acquisition function to output k new architectures
                candidates = []
                zc_scores = []
                if self.acq_fn_optimization == 'random_sampling':

                    for _ in range(self.num_candidates):
                        arch = self.search_space.clone()
                        arch.sample_random_architecture(dataset_api=self.dataset_api)
                        candidates.append(arch)
                    
                elif self.acq_fn_optimization == 'mutation':
                    # mutate the k best architectures by x
                    best_arch_indices = np.argsort(ytrain)[-self.num_arches_to_mutate:]
                    best_arches = [self.train_data[i].arch for i in best_arch_indices]
                    candidates = []
                    for arch in best_arches:
                        for _ in range(int(self.num_candidates / len(best_arches) / self.max_mutations)):
                            candidate = arch.clone()
                            for edit in range(int(self.max_mutations)):
                                arch_ = self.search_space.clone()
                                arch_.mutate(candidate, dataset_api=self.dataset_api)
                                candidate = arch_
                            candidates.append(candidate)

                else:
                    logger.info('{} is not yet supported as a acq fn optimizer'.format(encoding_type))
                    raise NotImplementedError()

                if self.zc and len(self.train_data) <= self.max_zerocost:
                    zc_method = ZeroCostEstimators(self.config, batch_size=64, method_type='jacov')
                    zc_method.train_loader = copy.deepcopy(self.train_loader)
                    zc_scores = zc_method.query(candidates)
                    values = [acq_fn(enc, score) for enc, score in zip(candidates, zc_scores)]
                else:
                    values = [acq_fn(encoding) for encoding in candidates]
                sorted_indices = np.argsort(values)
                choices = [candidates[i] for i in sorted_indices[-self.k:]]
                self.next_batch = [*choices]

            # train the next architecture chosen by the neural predictor
            model = torch.nn.Module()   # hacky way to get arch and accuracy checkpointable            
            model.arch = self.next_batch.pop()
            model.accuracy = model.arch.query(self.performance_metric, self.dataset, dataset_api=self.dataset_api)
            if self.zc and len(self.train_data) <= self.max_zerocost:                
                zc_method = ZeroCostEstimators(self.config, batch_size=64, method_type='jacov')
                zc_method.train_loader = copy.deepcopy(self.train_loader)
                score = zc_method.query([model.arch])
                model.zc_score = np.squeeze(score)

            self._update_history(model)
            self.train_data.append(model)
        
    def _update_history(self, child):
        if len(self.history) < 100:
            self.history.append(child)
        else:
            for i, p in enumerate(self.history):
                if child.accuracy > p.accuracy:
                    self.history[i] = child
                    break

    def train_statistics(self):
        best_arch = self.get_final_architecture()
        return (
            best_arch.query(Metric.TRAIN_ACCURACY, self.dataset, dataset_api=self.dataset_api), 
            best_arch.query(Metric.VAL_ACCURACY, self.dataset, dataset_api=self.dataset_api), 
            best_arch.query(Metric.TEST_ACCURACY, self.dataset, dataset_api=self.dataset_api), 
            best_arch.query(Metric.TRAIN_TIME, self.dataset, dataset_api=self.dataset_api), 
        )

    def test_statistics(self):
        best_arch = self.get_final_architecture()
        return best_arch.query(Metric.RAW, self.dataset, dataset_api=self.dataset_api)


    def get_final_architecture(self):
        return max(self.history, key=lambda x: x.accuracy).arch
    
    
    def get_op_optimizer(self):
        raise NotImplementedError()

    
    def get_checkpointables(self):
        return {'model': self.history}
    

    def get_model_size(self):
        return count_parameters_in_MB(self.history)
