import unittest
import torch
from ltn_imp.fuzzy_operators.quantifiers import ForallQuantifier, ExistsQuantifier

class TestQuantifiers(unittest.TestCase):

    # FIXME: tests with bare float numbers are very fragile, use a tolerance

    def _test_quantifier(self, quantifier, truth_values, expected, dim=0):
        result = quantifier(truth_values, dim=dim)
        self.assertTrue(torch.allclose(result, expected, atol=0.1), f"{result} not close to {expected}")

    def test_forall_quantifier(self):
        methods = {
            "min": torch.tensor([0.5]),
            "pmean": torch.tensor([0.7416]),
            "pmean_error": torch.tensor([0.6928])
        }
        truth_values = torch.tensor([[0.5], [0.7], [0.9]])
        for method, expected in methods.items():
            with self.subTest(method=method):
                forall = ForallQuantifier(method=method)
                self._test_quantifier(forall, truth_values, expected, dim=0)

    def test_exists_quantifier(self):
        methods = {
            "min": torch.tensor([0.5]),
            "pmean": torch.tensor([0.7416]),
            "pmean_error": torch.tensor([0.6928])
        }
        truth_values = torch.tensor([[0.5], [0.7], [0.9]])
        for method, expected in methods.items():
            with self.subTest(method=method):
                exists = ExistsQuantifier(method=method)
                self._test_quantifier(exists, truth_values, expected, dim=0)

if __name__ == '__main__':
    unittest.main()
