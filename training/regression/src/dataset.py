from torch.utils.data import Dataset
import torch
import os
from torch.nn import functional as F

class Hemopi_Regression_Dataset(Dataset):
    def __init__(self, dataset_config,headers, phc50_values, prob_features, merci_features, mode='train'):
        self.feat_paths = []
        self.dataset_config = dataset_config
        self.embedding_types = 'mean_representations' if dataset_config.mean else 'representations'
        self.max_length = dataset_config.max_length
        self.headers = headers
        self.phc50_values = phc50_values
        self.mode = mode
        self.prob_features = prob_features
        self.merci_features = merci_features
        if mode == 'test':
            for feature_name in dataset_config.feature_list:
                self.feat_paths.append(
                    os.path.join(dataset_config.test_data_root, feature_name)
                )
        else:
            for feature_name in dataset_config.feature_list:
                self.feat_paths.append(
                    os.path.join(dataset_config.train_data_root, feature_name)
                )
        self._setup_keys()
        self._preload_data()

    def get_pep_keys(self):
        return self.pep_keys

    def _setup_keys(self):
        self.pep_keys = [i for i in self.headers]

    def _preload_data(self):
        self.data = {}
        self.data_conv = {}
        self.data_merci = {}

        for key in self.pep_keys:
            self.data[key] = [
                torch.load(os.path.join(path, f'{key}.pt'))
                for path in self.feat_paths
            ]
            self.data_conv[key] = self.prob_features[key]
            self.data_merci[key] = self.merci_features[key]

    def __len__(self):
        return len(self.pep_keys)

    def __getitem__(self, idx):
        pep_key = self.pep_keys[idx]
        X = self.data[pep_key]
        X_probs = self.data_conv[pep_key]
        X_merci = self.data_merci[pep_key]

        len_tokens = [x.shape[0] if not self.dataset_config.mean else 1 for x in X]

        masks_X = [None for l in len_tokens]

        X = [x.detach() for x in X]

        if not self.dataset_config.mean:
            for i in range(len(X)):
                X[i] = F.pad(X[i], (0, 0, 0, self.max_length - X[i].size(0)), value=0)
                masks_X[i] = torch.ones(self.max_length, dtype=torch.bool)
                masks_X[i][:len_tokens[i]] = False

        return X, X_probs, X_merci, masks_X, self.phc50_values[idx]
