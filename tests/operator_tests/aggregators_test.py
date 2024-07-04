import unittest
import torch
from ltn_imp.fuzzy_operators.aggregators import AggregMin, AggregPMean, AggregPMeanError

class TestAggregators(unittest.TestCase):

    # FIXME: tests with bare float numbers are very fragile, use a tolerance

    def _test_aggregator(self, agg_op,  expected, **kwargs):
        xs = torch.tensor([[0.5, 0.7, 0.3], [0.2, 0.9, 0.4]])
        result = agg_op(xs, **kwargs)
        self.assertTrue(torch.allclose(result, expected, atol=0.1), f"{result} not close to {expected}")

    def test_aggreg_min(self):
        agg_min = AggregMin()
        expected = torch.tensor([0.3, 0.2])
        self._test_aggregator(agg_min,  expected, dim=1)

    def test_aggreg_pmean(self):
        agg_pmean = AggregPMean(p=2)
        expected = torch.tensor([0.5745, 0.5568])
        self._test_aggregator(agg_pmean,  expected, dim=1)

    def test_aggreg_pmean_error(self):
        agg_pmean_error = AggregPMeanError(p=2)
        expected = torch.tensor([0.5547, 0.4564])
        self._test_aggregator(agg_pmean_error, expected, dim=1)

if __name__ == '__main__':
    unittest.main()
