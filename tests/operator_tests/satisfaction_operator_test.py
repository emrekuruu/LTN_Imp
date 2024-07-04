import unittest
import torch
from ltn_imp.fuzzy_operators.aggregators import AggregMin, AggregPMean, AggregPMeanError, SatAgg

class TestSatAgg(unittest.TestCase):

    # FIXME: tests with bare float numbers are very fragile, use a tolerance

    def _test_sat_agg(self, sat_agg_op, expected, formulas):
        result = sat_agg_op(*formulas)
        self.assertTrue(torch.allclose(result, expected, atol=0.1), f"{result} not close to {expected}")

    def test_sat_agg_min(self):
        formulas = [torch.tensor(0.5), torch.tensor(0.7), torch.tensor(0.3)]
        sat_agg_min = SatAgg(AggregMin())
        expected = torch.tensor(0.3)
        self._test_sat_agg(sat_agg_min, expected, formulas)

    def test_sat_agg_pmean(self):
        formulas = [torch.tensor(0.5), torch.tensor(0.7), torch.tensor(0.3)]
        sat_agg_pmean = SatAgg(AggregPMean(p=2))
        expected = torch.tensor(0.5745)  # Adjusted based on p-mean calculation
        self._test_sat_agg(sat_agg_pmean, expected, formulas)

    def test_sat_agg_pmean_error(self):
        formulas = [torch.tensor(0.5), torch.tensor(0.7), torch.tensor(0.3)]
        sat_agg_pmean_error = SatAgg(AggregPMeanError(p=2))
        expected = torch.tensor(0.5547)  # Adjusted based on p-mean error calculation
        self._test_sat_agg(sat_agg_pmean_error, expected, formulas)

if __name__ == '__main__':
    unittest.main()
