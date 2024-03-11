from naslib.search_spaces import NasBench201SearchSpace
from naslib.utils import get_dataset_api
# from naslib.search_spaces.core import Metric

from naslib.predictors import ZeroCost
from naslib.utils import get_train_val_loaders, get_project_root
from fvcore.common.config import CfgNode
from tqdm import tqdm
from naslib.utils.encodings import EncodingType

import random
from tqdm import tqdm
from naslib.search_spaces.core.query_metrics import Metric
from naslib.search_spaces.nasbench201.conversions import convert_naslib_to_str

import seaborn as sns
import matplotlib.pyplot as plt
from naslib.utils import compute_scores  # computes more metrics than just correlation
from scipy.stats import kendalltau, spearmanr


graph = NasBench201SearchSpace(n_classes=10)

graph.sample_random_architecture()

# Parse the sampled architecture into a PyTorch model - this model can now be trained like a regular PyTorch model
# (This step is optional because we're not training the model from scratch)
graph.parse()

# Get the NASLib representation of the sampled model
graph.get_hash()

print(graph.get_hash())

dataset_apis = {}
dataset_apis["NASBench201-cifar10"] = get_dataset_api(search_space='nasbench201', dataset='cifar10')

train_acc_parent = graph.query(metric=Metric.TRAIN_ACCURACY, dataset='cifar10', dataset_api=dataset_apis["NASBench201-cifar10"])
val_acc_parent = graph.query(metric=Metric.VAL_ACCURACY, dataset='cifar10', dataset_api=dataset_apis["NASBench201-cifar10"])

print('Performance of parent model')
print(f'Train accuracy: {train_acc_parent:.2f}%')
print(f'Validation accuracy: {val_acc_parent:.2f}%')

# Create configs required for get_train_val_loaders
config_dict = {
    'dataset': 'cifar10', # Dataset to loader: can be cifar10, cifar100, ImageNet16-120
    'data': str(get_project_root()) + '/data', # path to naslib/data where cifar is saved
    'search': {
        'seed': 9001, # Seed to use in the train, validation and test dataloaders
        'train_portion': 0.7, # Portion of train dataset to use as train dataset. The rest is used as validation dataset.
        'batch_size': 32, # batch size of the dataloaders
    }
}
config = CfgNode(config_dict)

# Get the dataloaders
train_loader, val_loader, test_loader, train_transform, valid_transform = get_train_val_loaders(config)

# Sample a random NB201 graph and instantiate it
graph = NasBench201SearchSpace()
graph.sample_random_architecture()
graph.parse()

# Instantiate the ZeroCost predictor
# The Zero Cost predictors can be any of the following:
# {'epe_nas', 'fisher', 'grad_norm', 'grasp', 'jacov', 'l2_norm', 'nwot', 'plain', 'snip', 'synflow', 'zen', 'flops', 'params'}

zc_proxy = 'plain'
zc_predictor = ZeroCost(method_type=zc_proxy)
score = zc_predictor.query(graph=graph, dataloader=train_loader)

print(f'Score of model for Zero Cost predictor {zc_proxy}: {score}')


# Instantiate the ZeroCost predictor
# The Zero Cost predictors can be any of the following:
# zc_proxies = ['epe_nas', 'fisher', 'grad_norm', 'grasp', 'jacov', 'l2_norm', 'nwot', 'plain', 'snip', 'synflow', 'zen', 'flops', 'params']

# print('Scores of the model: ')
# for zc_proxy in zc_proxies:
#     zc_predictor = ZeroCost(method_type=zc_proxy)
#     score = zc_predictor.query(graph=graph, dataloader=train_loader)
#     print(f'{zc_proxy}: {score:3f}')


## Zero cost proxies for performance prediction
def evaluate_predictions(y_true, y_pred, plot=False, plot_func=None, title=None):
  res = {}
  res['kendalltau'] = kendalltau(y_true, y_pred)[0]
  res['spearmanr'] = spearmanr(y_true, y_pred)[0]

  if plot:
    plt.figure()
    plt.scatter(y_true, y_pred, marker='.')

    # additional graph aesthetics
    if plot_func is not None:
      plot_func(res)

    tautitle = f"tau: {res['kendalltau'].round(2)}"
    if title is not None:
      plt.title(title + f" - {tautitle}")
    else:
      plt.title(tautitle)

    plt.show()

  return res

def iterate_whole_searchspace(search_space, dataset_api, seed=None, shuffle=False):
  # Note - for nb301, this method only returns the training set architectures
  arch_iter = search_space.get_arch_iterator(dataset_api)
  if shuffle:
    arch_iter = list(arch_iter)
    rng = random if seed is None else random.Random(seed)
    rng.shuffle(arch_iter)

  for arch_str in arch_iter:
    yield arch_str


def sample_arch_dataset(search_space, dataset, dataset_api, data_size=None, arch_hashes=None, seed=None, shuffle=False):
  xdata = []
  ydata = []
  train_times = []
  arch_hashes = arch_hashes if arch_hashes is not None else set()

  # Cloning NASLib objects takes some time - this is a hack-... speedup so that
  # we can quickly get all architecture hashes and accuracies in a searchspace.
  # However, not all methods are available - e.g. you can't encode the architecture
  search_space = search_space.clone()
  search_space.instantiate_model = False
  arch_iterator = iterate_whole_searchspace(search_space, dataset_api, shuffle=shuffle, seed=seed)

  # iterate over architecture hashes
  for arch in tqdm(arch_iterator):
      if data_size is not None and len(xdata) >= data_size:
        break

      if arch in arch_hashes:
          continue

      arch_hashes.add(arch)
      search_space.set_spec(arch)

      # query metric for the current architecture hash
      accuracy = search_space.query(metric=Metric.TRAIN_ACCURACY, dataset=dataset, dataset_api=dataset_api)
      train_time = search_space.query(metric=Metric.TRAIN_TIME, dataset=dataset, dataset_api=dataset_api)

      xdata.append(arch)
      ydata.append(accuracy)
      train_times.append(train_time)

  return [xdata, ydata, train_times], arch_hashes

def encode_archs(search_space, arch_ops, encoding=None, verbose=True):
    encoded = []

    for arch_str in tqdm(arch_ops, disable=not verbose):
        arch = search_space.clone()
        arch.set_spec(arch_str)

        arch = arch.encode(encoding) if encoding is not None else arch
        encoded.append(arch)

    return encoded

def eval_zcp(model, zc_name, data_loader):
    model = encode_archs(NasBench201SearchSpace(), [model], verbose=False)[0]
    model.parse()
    zc_pred = ZeroCost(method_type=zc_name)
    res = zc_pred.query(graph=model, dataloader=data_loader)

    return {zc_name: res}


seed = 2
pred_dataset = 'cifar10'
pred_api = dataset_apis['NASBench201-cifar10']
train_size = 10
test_size = 10

train_sample, train_hashes = sample_arch_dataset(NasBench201SearchSpace(), pred_dataset, pred_api, data_size=train_size, shuffle=True, seed=seed)
test_sample, _ = sample_arch_dataset(NasBench201SearchSpace(), pred_dataset, pred_api, arch_hashes=train_hashes, data_size=test_size, shuffle=True, seed=seed + 1)

# xtrain, ytrain, _ = train_sample
xtest, ytest, _ = test_sample

# zc_proxies = ['epe_nas', 'fisher', 'grad_norm', 'grasp', 'jacov', 'l2_norm', 'nwot', 'plain', 'snip', 'synflow', 'zen', 'flops', 'params']
zc_proxies = ['epe_nas', 'grasp', 'jacov']
# zcp_name = 'synflow'
# zc_only = False

spearman_metrics = {}
for zcp_name in zc_proxies:
  # train and query expect different ZCP formats for some reason
  # zcp_train = {'zero_cost_scores': [eval_zcp(t_arch, zcp_name, train_loader) for t_arch in tqdm(xtrain)]}
  zcp_test = [{'zero_cost_scores': eval_zcp(t_arch, zcp_name, train_loader)} for t_arch in tqdm(xtest)]

  zcp_pred = [s['zero_cost_scores'][zcp_name] for s in zcp_test]
  metrics = evaluate_predictions(ytest, zcp_pred, plot=False,
                                title=f"NB201 accuracies vs {zcp_name}")

  # print("zcp_test:  ", zcp_test)
  # print("zcp_pred:  ", zcp_pred)
  # print("\n", metrics)
  spearman_metrics[zcp_name] = metrics['spearmanr']

print("spearman_metrics:  ", spearman_metrics)


# # ensemble
# weights = {
#    'epe_nas': 40, 
#    'grasp': 20, 
#    'jacov': 40
# }

# zcp_preds = {}
# for zcp_name in zc_proxies:
#   zcp_test = [{'zero_cost_scores': eval_zcp(t_arch, zcp_name, train_loader)} for t_arch in tqdm(xtest)]
#   zcp_pred = [s['zero_cost_scores'][zcp_name] * weights[zcp_name] for s in zcp_test]
#   zcp_preds[zcp_name] = zcp_pred

# ensemble_preds = []
# for i in range(len(zcp_preds['epe_nas'])):
#   ensemble_preds[i] = sum([zcp_preds[zcp_name][i] for zcp_name in zc_proxies])/sum(weights.values())

# ens_metrics = evaluate_predictions(ytest, ensemble_preds, plot=False,
#                                 title=f"NB201 accuracies vs {zcp_name}")

# print("ens_metrics:  ", ens_metrics)
