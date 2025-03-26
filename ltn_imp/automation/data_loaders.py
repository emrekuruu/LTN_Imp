from itertools import cycle
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class DynamicDataset(Dataset):
    def __init__(self, config, features, device=None, scalers=None, type = None):
        self.config = config
        self.features = features
        self.device = device  # Store the device
        self.data = pd.read_csv(config["path"], index_col=0)
        self.batch_size = config["batch_size"]
        
        self.instance_features = {instance: [str(feat) for feat in features[instance]] for instance in config["instances"]}
        self.target_features = {target: [str(feat) for feat in features[target]] for target in config["targets"]}

        if any(self.data.index.name in self.instance_features[instance] for instance in self.config["instances"]):
            self.data = pd.read_csv(config["path"])

        if type == "standard":
            scaler = StandardScaler

        if type == "minmax":
            scaler = MinMaxScaler

        if scalers is None and type is not None:
            self.scalers = {}
            for instance in self.config["instances"]:
                self.scalers[instance] = {}
                for feature in self.instance_features[instance]:
                    self.scalers[instance][feature] = scaler()
                    self.data[feature] = self.scalers[instance][feature].fit_transform(self.data[feature].values.reshape(-1, 1))

        elif scalers:
            self.scalers = scalers
            for instance in self.config["instances"]:
                for feature in self.instance_features[instance]:
                    self.data[feature] = self.scalers[instance][feature].fit_transform(self.data[feature].values.reshape(-1, 1))

        elif type is None:
            self.scalers = {}
            return 


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        batch = []

        # Get instance data
        for instance in self.config["instances"]:
            instance_data = self.data[self.instance_features[instance]].iloc[idx]
            batch.append(instance_data.values)

        # Get target data
        for target in self.config["targets"]:
            target_data = self.data[self.target_features[target]].iloc[idx]
            batch.append(target_data.values)

        # Convert to tensors and move to the specified device
        batch = [torch.tensor(item, dtype=torch.float32).to(self.device) for item in batch]
        
        return tuple(batch)
class LoaderWrapper:
    def __init__(self, config, features, device=None, scalers = None, type = None, shuffle = True):
        self.variables = config["instances"]
        self.targets = config["targets"]
        self.loader = DataLoader(DynamicDataset(config, features, device=device, scalers=scalers, type = type), batch_size=config["batch_size"], shuffle=shuffle)
        self.scalers = self.loader.dataset.scalers

    def __iter__(self):
        self.iter_loader = iter(self.loader)
        return self
    
    def __next__(self):
        return next(self.iter_loader)
    
    def __len__(self):
        return len(self.loader)
    
    def __repr__(self) -> str:
        return f"<Loader>:({self.variables} -> {self.targets})"
    
class CombinedDataLoader:
    def __init__(self, loaders):
        self.loaders = loaders
        self.iters = {loader: cycle(loader) for loader in loaders}
        self.current_batches = {loader: None for loader in loaders}
        self.step()

        if loaders != []:
            self.max_length = max(len(loader) for loader in loaders)
        else:
            self.max_length = 1

    def __iter__(self):
        return self

    def __len__(self):
        return self.max_length
    
    def __next__(self):
        return self.current_batches

    def step(self):
        for loader in self.loaders:
            self.current_batches[loader] = next(self.iters[loader])


