import unittest
import torch
from ltn_imp.fuzzy_operators.connectives import AndConnective, OrConnective, NotConnective, ImpliesConnective, IffConnective

class TestConnectives(unittest.TestCase):

    # FIXME: tests with bare float numbers are very fragile, use a tolerance

    def _test_connective(self, connective, a, b=None, expected=None):
        if b is not None:
            result = connective(a, b)
        else:
            result = connective(a)
        self.assertTrue(torch.allclose(result, expected, atol=1e-4), f"{result} not close to {expected}")

    def test_and_min(self):
        and_conn = AndConnective(implementation_name="min")
        a = torch.tensor([0.5])
        b = torch.tensor([0.7])
        expected = torch.tensor([0.5])
        self._test_connective(and_conn, a, b, expected)

    def test_and_prod(self):
        and_conn = AndConnective(implementation_name="prod")
        a = torch.tensor([0.5])
        b = torch.tensor([0.7])
        expected = torch.tensor([0.35])
        self._test_connective(and_conn, a, b, expected)

    def test_and_luk(self):
        and_conn = AndConnective(implementation_name="luk")
        a = torch.tensor([0.5])
        b = torch.tensor([0.7])
        expected = torch.tensor([0.2])
        self._test_connective(and_conn, a, b, expected)

    def test_or_max(self):
        or_conn = OrConnective(implementation_name="max")
        a = torch.tensor([0.5])
        b = torch.tensor([0.7])
        expected = torch.tensor([0.7])
        self._test_connective(or_conn, a, b, expected)

    def test_or_prob_sum(self):
        or_conn = OrConnective(implementation_name="prob_sum")
        a = torch.tensor([0.5])
        b = torch.tensor([0.7])
        expected = torch.tensor([0.85])
        self._test_connective(or_conn, a, b, expected)

    def test_or_luk(self):
        or_conn = OrConnective(implementation_name="luk")
        a = torch.tensor([0.5])
        b = torch.tensor([0.7])
        expected = torch.tensor([1.0])
        self._test_connective(or_conn, a, b, expected)

    def test_not_standard(self):
        not_conn = NotConnective(implementation_name="standard")
        a = torch.tensor([0.5])
        expected = torch.tensor([0.5])
        self._test_connective(not_conn, a, expected=expected)

    def test_not_godel(self):
        not_conn = NotConnective(implementation_name="godel")
        a = torch.tensor([0.5])
        expected = torch.tensor([0.0])
        self._test_connective(not_conn, a, expected=expected)

    def test_implies_kleene_dienes(self):
        implies_conn = ImpliesConnective(implementation_name="kleene_dienes")
        a = torch.tensor([0.5])
        b = torch.tensor([0.7])
        expected = torch.tensor([0.7])
        self._test_connective(implies_conn, a, b, expected)

    def test_implies_godel(self):
        implies_conn = ImpliesConnective(implementation_name="godel")
        a = torch.tensor([0.5])
        b = torch.tensor([0.7])
        expected = torch.tensor([1.0])
        self._test_connective(implies_conn, a, b, expected)

    def test_implies_reichenbach(self):
        implies_conn = ImpliesConnective(implementation_name="reichenbach")
        a = torch.tensor([0.5])
        b = torch.tensor([0.7])
        expected = torch.tensor([0.85])
        self._test_connective(implies_conn, a, b, expected)

    def test_implies_goguen(self):
        implies_conn = ImpliesConnective(implementation_name="goguen")
        a = torch.tensor([0.5])
        b = torch.tensor([0.7])
        expected = torch.tensor([1.0])
        self._test_connective(implies_conn, a, b, expected)

    def test_implies_luk(self):
        implies_conn = ImpliesConnective(implementation_name="luk")
        a = torch.tensor([0.7])
        b = torch.tensor([0.5])
        expected = torch.tensor([0.8])
        self._test_connective(implies_conn, a, b, expected)

    def test_iff(self):
        iff_conn = IffConnective()
        a = torch.tensor([0.5])
        b = torch.tensor([0.7])
        expected = torch.tensor([0.8])
        self._test_connective(iff_conn, a, b, expected)

if __name__ == '__main__':
    unittest.main()
