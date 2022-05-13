import codecs
import os
import json
import tracemalloc
import torch
import numpy as np
import logging
import timeit

from guppy import hpy
from naslib.predictors.zerocost import ZeroCost
from naslib.search_spaces.core.query_metrics import Metric

from naslib.utils import compute_scores

logger = logging.getLogger(__name__)

class ZCEnsembleEvaluator(object):
    def __init__(self, n_train, n_test, zc_names):
        self.n_train = n_train
        self.n_test = n_test
        self.zc_names = zc_names
        self.performance_metric = Metric.VAL_ACCURACY

    def _compute_zc_scores(self, model, predictors, train_loader):
        zc_scores = {}
        zc_times = {}
        for predictor in predictors:
            starttime = timeit.default_timer()
            score = predictor.query(model, train_loader)
            endtime = timeit.default_timer()
            zc_times[predictor.method_type] = endtime - starttime
            zc_scores[predictor.method_type] = score

        logger.info(zc_times)

        return zc_scores

    def _sample_new_model(self):
        model = torch.nn.Module()
        model.arch = self.search_space.clone()
        model.arch.sample_random_architecture(dataset_api=self.dataset_api)
        model.arch.parse()
        model.accuracy = model.arch.query(self.performance_metric, self.dataset, dataset_api=self.dataset_api)

        return model

    def _log_to_json(self, results, filepath):
        """log statistics to json file"""
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        with codecs.open(
            os.path.join(filepath, "scores.json"), "w", encoding="utf-8"
        ) as file:
            for res in results:
                for key, value in res.items():
                    if type(value) == np.int32 or type(value) == np.int64:
                        res[key] = int(value)
                    if type(value) == np.float32 or type(value) == np.float64:
                        res[key] = float(value)

            json.dump(results, file, separators=(",", ":"))

    def adapt_search_space(self, search_space, dataset, dataset_api, config):
        self.search_space = search_space.clone()
        self.dataset = dataset
        self.dataset_api = dataset_api
        self.config = config

    def sample_random_models(self, n):
        tracemalloc.start()
        models = [self._sample_new_model() for _ in range(n)]
        current, peak = tracemalloc.get_traced_memory()
        logger.info(f"sample_random_models:: Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
        tracemalloc.stop()
        return models

    def compute_zc_scores(self, models, zc_predictors, train_loader):
        for model in models:
            model.zc_scores = self._compute_zc_scores(model.arch, zc_predictors, train_loader)

    def evaluate(self, ensemble, train_loader):
        # Load models to train
        starttime = timeit.default_timer()
        train_models = self.sample_random_models(self.n_train)
        endtime = timeit.default_timer()
        logger.info(f'Time to sample and query train models: {endtime - starttime}s')

        # Get their ZC scores
        zc_predictors = [ZeroCost(method_type=zc_name) for zc_name in self.zc_names]

        starttime = timeit.default_timer()

        tracemalloc.start()
        self.compute_zc_scores(train_models, zc_predictors, train_loader)
        current, peak = tracemalloc.get_traced_memory()
        logger.info(f"compute_zc_scores:: Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
        tracemalloc.stop()

        endtime = timeit.default_timer()
        logger.info(f'Time to compute zc scores of train models: {endtime - starttime}s')

        # Set ZC results as precomputations and fit the ensemble
        train_info = {'zero_cost_scores': [m.zc_scores for m in train_models]}
        ensemble.set_pre_computations(xtrain_zc_info=train_info)

        xtrain = [m.arch for m in train_models]
        ytrain = [m.accuracy for m in train_models]

        starttime = timeit.default_timer()
        ensemble.fit(xtrain, ytrain)
        endtime = timeit.default_timer()
        logger.info(f'Time to fit xgboost: {endtime - starttime}s')

        # Get the feature importance
        # self.ensemble[0].feature_importance

        # Sample test models, query zc scores
        starttime = timeit.default_timer()
        test_models = self.sample_random_models(self.n_test)
        endtime = timeit.default_timer()
        logger.info(f'Time to sample and query test models: {endtime - starttime}s')

        starttime = timeit.default_timer()
        self.compute_zc_scores(test_models, zc_predictors, train_loader)
        endtime = timeit.default_timer()
        logger.info(f'Time to compute zc scores of test models: {endtime - starttime}s')

        # Query the ensemble for the predicted accuracy
        x_test = [m.arch for m in test_models]
        test_info = [{'zero_cost_scores': m.zc_scores} for m in test_models]
        preds = np.mean(ensemble.query(x_test, test_info), axis=0)

        # Compute scores
        ground_truths = [m.accuracy for m in test_models]
        scores = compute_scores(ground_truths, preds)

        model = ensemble.ensemble[0].model
        feature_importances = model.get_fscore()
        feature_mapping = ensemble.ensemble[0].zc_to_features_map

        zc_feature_importances = {zc_name: 0 for zc_name in self.zc_names}
        for zc_name, feature_name in feature_mapping.items():
            if feature_name in feature_importances:
                zc_feature_importances[zc_name] = feature_importances[feature_name]

        scores['zc_feature_importances'] = zc_feature_importances
        scores['feature_importances'] = feature_importances

        logger.info(f'ZC feature importances: {zc_feature_importances}')
        self._log_to_json([self.config, scores], self.config.save)
