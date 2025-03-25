import torch 
from ltn_imp.fuzzy_operators.aggregators import SatAgg
from ltn_imp.automation.data_loaders import CombinedDataLoader, LoaderWrapper
from ltn_imp.parsing.expressions import LessThanExpression, MoreThanExpression, EqualityExpression
from ltn_imp.parsing.parser import LTNConverter
from ltn_imp.parsing.ancillary_modules import ModuleFactory
from ltn_imp.automation.network_factory import NNFactory
import yaml

sat_agg_op = SatAgg()

class KnowledgeBase:
    def __init__(self, yaml_file, device = "cpu"):

        if device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif device == "mps" and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
    
        print(f"Using device: {self.device}")
        
        with open(yaml_file, "r") as file:
            config = yaml.safe_load(file)
            
        self.config = config
        self.factory = NNFactory()
        self.scalers = {}

        self.set_loaders()
        self.set_val_loaders()
        self.set_test_loaders()

        self.constant_mapping = self.set_constant_mapping()
        self.set_predicates()
        self.set_converter()
        self.set_ancillary_rules()
        self.set_rules()
        self.set_rule_weights()
        self.set_rule_to_data_loader_mapping()

    def set_converter(self):
        self.converter = LTNConverter(yaml=self.config, predicates=self.predicates, device=self.device, scalers=self.scalers) 

    def set_constant_mapping(self):
        constants = self.config.get("constants", [])
        constant_mapping = {}
        for const in constants:
            for name, value in const.items():
                if isinstance(value, list):
                    constant_mapping[name] = torch.tensor(value, dtype=torch.float32)
                else:
                    constant_mapping[name] = torch.tensor([value], dtype=torch.float32)

        return constant_mapping

    def evaluate_layer_size(self, layer_size, features_dict, instance_names):
        in_size_str, out_size_str = layer_size
        for instance_name in instance_names:
            feature_count = len(features_dict[instance_name])
            if isinstance(in_size_str, str):
                in_size_str = in_size_str.replace(instance_name, str(feature_count))
            if isinstance(out_size_str, str):
                out_size_str = out_size_str.replace(instance_name, str(feature_count))

        in_size = eval(in_size_str) if isinstance(in_size_str, str) else in_size_str
        out_size = eval(out_size_str) if isinstance(out_size_str, str) else out_size_str
        return in_size, out_size

    def set_predicates(self):
        features = self.config["features"]
        self.predicates = {}

        for predicate_name, predicate_info in self.config["predicates"].items():
            args = predicate_info["args"]
            structure = predicate_info["structure"]
            
            layers = []
            activations = []
            regularizations = []

            for layer in structure['layers']:
                layer_type = list(layer.keys())[0]
                layer_size = layer[layer_type]
                activation = layer.get('activation', None)
                regularization = layer.get('regularization', [])
                
                in_size, out_size = self.evaluate_layer_size(layer_size, features, args)
                layers.append((in_size, out_size))
                activations.append(activation)
                regularizations.append(regularization)

            arguments = predicate_info["args"]

            network = self.factory(
                arguments=arguments,
                layers=layers,
                activations=activations,
                regularizations=regularizations
            )

            self.predicates[predicate_name] = network.float()

        for predicate in self.predicates.values():
            predicate.to(self.device)

    def set_rules(self):
        self.rules = [self.converter(rule["rule"]) for rule in self.config["constraints"]]

    def set_rule_weights(self):
        self.rule_weights = [int(rule.get("weight", 1)) for rule in self.config["constraints"]]

    def set_ancillary_rules(self):
        if "knowledge" not in self.config:
            return
        for anchor in self.config["knowledge"]:
            name = anchor["rule"]
            params = anchor["args"]
            functionality = anchor["clause"]
            ModuleFactory(self.converter).create_module(name, params, functionality)

    def set_loaders(self):
        self.loaders = []
        features = self.config["features"]
        for dataset in self.config["train"]:
            dataset = self.config["train"][dataset]
            type = dataset["scaler"] if "scaler" in dataset else None
            loader = LoaderWrapper(dataset, features, device=self.device, type = type, shuffle = True)
            self.loaders.append(loader)
            self.scalers.update(loader.scalers)

    def set_val_loaders(self):
        if "validation" not in self.config:
            self.val_loaders = None
            return
        self.val_loaders = []
        features = self.config["features"]
        for dataset in self.config["validation"]:
            dataset = self.config["validation"][dataset]
            loader = LoaderWrapper(dataset, features, device=self.device, scalers=self.scalers, shuffle = True)
            self.val_loaders.append(loader)

    def set_test_loaders(self):
        if "test" not in self.config:
            self.test_loaders = None
            return
        self.test_loaders = []
        features = self.config["features"]
        for dataset in self.config["test"]:
            dataset = self.config["test"][dataset]
            loader = LoaderWrapper(dataset, features, device=self.device, scalers=self.scalers, shuffle = False)
            self.test_loaders.append(loader)
        
    def set_rule_to_data_loader_mapping(self):
        rule_to_loader_mapping = {}

        for rule in self.rules:
            variables = rule.variables()
            for v in variables:
                for loader in self.loaders:
                    if str(v) in loader.variables or str(v) in loader.targets:
                        if rule in rule_to_loader_mapping:
                            rule_to_loader_mapping[rule].append(loader)
                        else:
                            rule_to_loader_mapping[rule] = [loader]

        self.rule_to_data_loader_mapping = rule_to_loader_mapping
    
    def loss(self, rule_outputs):
        input = []
        for weight, rule_output in zip(self.rule_weights, rule_outputs):
            for _ in range(weight):
                input.append(rule_output)
                                
        sat_agg_value = sat_agg_op(
            *input,
        )
        loss = 1.0 - sat_agg_value

        assert torch.isfinite(loss).all(), f"Loss contains invalid values: {loss}"

        return loss
        
    def parameters(self):
        params = []
        for model in self.predicates.values():
            if hasattr(model, 'parameters'):
                params += list(model.parameters())
        return params
    
    def partition_data(self, var_mapping, batch, loader):
        for k, v in self.constant_mapping.items(): 
            var_mapping[k] = v.to(self.device)

        *batch, = batch 

        for i, var in enumerate(loader.variables):
            var_mapping[var] = batch[i].to(self.device)

        for i, target in enumerate(loader.targets):
            var_mapping[target] = batch[i + len(loader.variables)].to(self.device)
            
    def compute_validation_loss(self):
        with torch.no_grad():
            rule_outputs = []
            for rule in self.rules:
                rule_output = []
                var_mapping = {}
                if self.val_loaders:
                    for loader in self.val_loaders:
                        for batch in loader:
                            self.partition_data(var_mapping, batch, loader)
                            rule_output.append(rule(var_mapping))
                else:
                    rule_output.append(rule(var_mapping))
                rule_outputs.append(torch.mean(torch.stack(rule_output)))
            validation_loss = self.loss(rule_outputs)
        return validation_loss
    
    def compute_test_loss(self):
        with torch.no_grad():
            rule_outputs = []
            for rule in self.rules:
                rule_output = []
                var_mapping = {}
                if self.test_loaders:
                    for loader in self.test_loaders:
                        for batch in loader:
                            self.partition_data(var_mapping, batch, loader)
                            rule_output.append(rule(var_mapping))
                else:
                    rule_output.append(rule(var_mapping))
                rule_outputs.append(torch.mean(torch.stack(rule_output)))
            test_loss = self.loss(rule_outputs)
        return test_loss

    def optimize(self, num_epochs=10, log_steps=10, lr=0.001, early_stopping=False, patience=5, min_delta=0.0, weight_decay=0.0, verbose = True):
    
        try:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        except Exception as e:
            print(e)
            print("No parameters to optimize")
            return

        all_loaders = set(loader for loaders in self.rule_to_data_loader_mapping.values() if loaders is not None for loader in loaders if loader is not None)
        combined_loader = CombinedDataLoader([loader for loader in all_loaders if loader is not None])
        best_val_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(num_epochs):
            for _ in range(len(combined_loader)):
                rule_outputs = []
                current_batches = next(combined_loader)

                for rule in self.rules:
                    loaders = self.rule_to_data_loader_mapping[rule]
                    var_mapping = {}
                
                    if loaders == None:
                        rule_outputs.append(rule(var_mapping))
                        continue
                        
                    for loader in loaders:
                        batch = current_batches[loader]
                        self.partition_data(var_mapping, batch, loader)
                    
                    rule_output = rule(var_mapping)
                    rule_outputs.append(rule_output)
                            
                self.optimizer.zero_grad()
                loss = self.loss(rule_outputs)
                loss.backward()
                self.optimizer.step()

            if epoch % log_steps == 0 and verbose:
                validation_loss = self.compute_validation_loss() if self.val_loaders else None

                for rule, outcome in zip(self.rules, rule_outputs):
                    print(f"Rule: {rule}, Outcome: {outcome}")
                
                if validation_loss is not None:
                    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {loss.item()}, Validation Loss: {validation_loss.item()}")
                else:
                    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {loss.item()}")
                print()

            # Early Stopping Logic
            if early_stopping and self.val_loaders and verbose:
                validation_loss = self.compute_validation_loss()
                if validation_loss + min_delta < best_val_loss:
                    best_val_loss = validation_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= patience:
                    for rule, outcome in zip(self.rules, rule_outputs):
                        print(f"Rule: {rule}, Outcome: {outcome}")
                    print(f"Early stopping at Epoch {epoch + 1}/{num_epochs}, Train Loss: {loss.item()}, Validation Loss: {validation_loss.item()}")
                    break
