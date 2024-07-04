from abc import ABC, abstractmethod
import torch

# Base class for all connectives
class Connective(ABC):
    @abstractmethod
    def forward(self, *args):
        pass

    def __call__(self, *args):
        return self.forward(*args)

# Base class for binary connectives
class BinaryConnective(Connective):
    def __init__(self, implementation):
        self.implementation = implementation

    def forward(self, a, b):
        return self.implementation(a, b)

# Base class for unary operations
class UnaryConnective(Connective):
    def __init__(self, implementation):
        self.implementation = implementation

    def forward(self, a):
        return self.implementation(a)

# FIXME: the "implementations" dictionary is a trick to escape good OOP. let's rething that asap

class AndConnective(BinaryConnective):
    def __init__(self, implementation_name = "min", stable=True):
        implementations = {
            "min": self.and_min,
            "prod": lambda a, b: self.and_prod(a, b, stable=stable),
            "luk": self.and_luk
        }

        if implementation_name not in implementations:
            raise ValueError(f"Unknown implementation: {implementation_name}")

        super().__init__(implementations[implementation_name])

    def and_min(self, a, b):
        return torch.minimum(a, b)

    def and_prod(self, a, b, stable=True):
        eps = 1e-4
        if stable:
            a = (1 - eps) * a + eps
            b = (1 - eps) * b + eps
        return torch.mul(a, b)

    def and_luk(self, a, b):
        return torch.maximum(a + b - 1, torch.zeros_like(a))

class OrConnective(BinaryConnective):
    def __init__(self, implementation_name="max", stable=True):
        implementations = {
            "max": self.or_max,
            "prob_sum": lambda a, b: self.or_prob_sum(a, b, stable=stable),
            "luk": self.or_luk
        }
        if implementation_name not in implementations:
            raise ValueError(f"Unknown implementation: {implementation_name}")
        super().__init__(implementations[implementation_name])

    def or_max(self, a, b):
        return torch.maximum(a, b)

    def or_prob_sum(self, a, b, stable=True):
        eps = 1e-4
        if stable:
            a = (1 - eps) * a
            b = (1 - eps) * b
        return a + b - a * b

    def or_luk(self, a, b):
        return torch.minimum(a + b, torch.ones_like(a))

class ImpliesConnective(BinaryConnective):
    def __init__(self, implementation_name="kleene_dienes", stable=True):
        implementations = {
            "kleene_dienes": self.implies_kleene_dienes,
            "godel": self.implies_godel,
            "reichenbach": lambda a, b: self.implies_reichenbach(a, b, stable=stable),
            "goguen": lambda a, b: self.implies_goguen(a, b, stable=stable),
            "luk": self.implies_luk
        }

        if implementation_name not in implementations:
            raise ValueError(f"Unknown implementation: {implementation_name}")

        super().__init__(implementations[implementation_name])

    def implies_kleene_dienes(self, a, b):
        return torch.maximum(1. - a, b)

    def implies_godel(self, a, b):
        return torch.where(a <= b, torch.ones_like(a), b)

    def implies_reichenbach(self, a, b, stable=True):
        eps = 1e-4
        if stable:
            a = (1 - eps) * a + eps
            b = (1 - eps) * b
        return 1. - a + a * b

    def implies_goguen(self, a, b, stable=True):
        eps = 1e-4
        if stable:
            a = (1 - eps) * a + eps
        return torch.where(a <= b, torch.ones_like(a), b / a)

    def implies_luk(self, a, b):
        return torch.minimum( 1.0 - a + b, torch.ones_like(a))



class IffConnective(BinaryConnective):
    def __init__(self, implementation_name="default"):
        implementations = {
            "default": self.iff_default
        }

        if implementation_name not in implementations:
            raise ValueError(f"Unknown implementation: {implementation_name}")

        super().__init__(implementations[implementation_name])

    def iff_default(self, a, b):
        return 1 - torch.abs(a - b)

class NotConnective(UnaryConnective):
    def __init__(self, implementation_name="standard"):
        implementations = {
            "standard": self.not_standard,
            "godel": self.not_godel
        }

        if implementation_name not in implementations:
            raise ValueError(f"Unknown implementation: {implementation_name}")

        super().__init__(implementations[implementation_name])

    def not_standard(self, a):
        return 1 - a

    def not_godel(self, a):
        return torch.eq(a, 0.).float()
