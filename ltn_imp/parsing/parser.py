from nltk.sem import logic
from ltn_imp.fuzzy_operators.connectives import AndConnective, OrConnective, NotConnective, ImpliesConnective, IffConnective
from ltn_imp.fuzzy_operators.predicates import Predicate 
from ltn_imp.fuzzy_operators.quantifiers import ForallQuantifier, ExistsQuantifier
from ltn_imp.fuzzy_operators.functions import Function
from nltk.sem.logic import Expression

class ExpressionVisitor:

    def __init__(self, predicates, functions, connective_impls=None, quantifier_impls=None):

        connective_impls = connective_impls or {}
        quantifier_impls = quantifier_impls or {}

        self.predicates = predicates
        self.functions = functions

        And = AndConnective(implementation_name=connective_impls.get('and', 'min'))
        Or = OrConnective(implementation_name=connective_impls.get('or', 'max'))
        Not = NotConnective(implementation_name=connective_impls.get('not', 'standard'))
        Implies = ImpliesConnective(implementation_name=connective_impls.get('implies', 'kleene_dienes'))
        Equiv = IffConnective(implementation_name=connective_impls.get('iff', 'default'))

        Exists = ExistsQuantifier(method=quantifier_impls.get('exists', 'max'))
        Forall = ForallQuantifier(method=quantifier_impls.get('forall', 'min'))

        self.connective_map = {
            logic.AndExpression: And,
            logic.OrExpression: Or,
            logic.ImpExpression: Implies,
            logic.IffExpression: Equiv,
            logic.NegatedExpression : Not
        }
        self.quantifier_map = {
            logic.ExistsExpression: Exists,
            logic.AllExpression: Forall
        }

    def visit(self, expression):
        if isinstance(expression, logic.ApplicationExpression):
            return self.visit_ApplicationExpression(expression)
        elif isinstance(expression, logic.BinaryExpression):
            return self.visit_BinaryExpression(expression)
        elif isinstance(expression, logic.NegatedExpression):
            return self.visit_NegatedExpression(expression)
        elif isinstance(expression, logic.QuantifiedExpression):
            return self.visit_QuantifiedExpression(expression)
        else:
            return self.generic_visit(expression)

    def generic_visit(self, expression):
        raise NotImplementedError(f"No visit method for {expression.__class__.__name__}")

    def visit_ApplicationExpression(self, expression):
        functor = expression.function
        while hasattr(functor, 'function'):
            functor = functor.function
        functor = str(functor)
        if functor in self.predicates:
            return lambda var_mapping: Predicate(self.predicates[functor])(*[var_mapping[str(var)] for var in expression.variables()])
        elif functor in self.functions:
            return lambda var_mapping: Function(self.functions[functor])(*[var_mapping[str(var)] for var in expression.variables()])
        else:
            raise ValueError(f"Unknown functor: {type(functor)}")

    def visit_BinaryExpression(self, expression):
        connective = self.connective_map.get(type(expression))
        if connective:
            left = self.visit(expression.first)
            right = self.visit(expression.second)
            return lambda var_mapping: connective(left(var_mapping), right(var_mapping))
        else:
            raise NotImplementedError(f"Unsupported binary expression type: {type(expression)}")

    def visit_NegatedExpression(self, expression):
        connective = self.connective_map.get(type(expression))
        term = self.visit(expression.term)
        return lambda var_mapping: connective(term(var_mapping))

    def visit_QuantifiedExpression(self, expression):
        quantifier = self.quantifier_map.get(type(expression))
        if quantifier:
            term = self.visit(expression.term)
            return lambda variable_mapping: quantifier([], term(variable_mapping))
        else:
            raise NotImplementedError(f"Unsupported quantifier expression type: {type(expression)}")

# Add the accept method dynamically to Expression and all its subclasses
def add_accept_method(cls):
    cls.accept = lambda self, visitor: visitor.visit(self)

def convert_to_ltn(expression, predicates, functions, connective_impls=None, quantifier_impls=None):

    for subclass in logic.Expression.__subclasses__():
        add_accept_method(subclass)
        for subsubclass in subclass.__subclasses__():
            add_accept_method(subsubclass)

    expression = Expression.fromstring(expression)
    visitor = ExpressionVisitor(predicates, functions, connective_impls = connective_impls, quantifier_impls = quantifier_impls )
    return expression.accept(visitor)