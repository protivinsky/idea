from abc import ABC, abstractmethod


class ExpressionPrintingVisitor():

    def print_literal(self, literal) -> None:
        print(literal)

    def print_addition(self, addition) -> None:
        left_value = addition.left.get_value()
        right_value = addition.right.get_value()
        sum = addition.get_value()
        print(f'{left_value} + {right_value} = {sum}')


class Expression(ABC):

    @abstractmethod
    def accept(visitor: ExpressionPrintingVisitor) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_value():
        raise NotImplementedError

    
class Literal(Expression):
    
    def __init__(self, value):
        self.value = value

    def accept(self, visitor: ExpressionPrintingVisitor) -> None:
        visitor.print_literal(self.value)

    def get_value(self):
        return self.value


class Addition(Expression):

    def __init__(self, left: Expression, right: Expression):
        self.left = left
        self.right = right

    def accept(self, visitor: ExpressionPrintingVisitor) -> None:
        self.left.accept(visitor)
        self.right.accept(visitor)
        visitor.print_addition(self)

    def get_value(self):
        return self.left.get_value() + self.right.get_value()


# 1 + 2 + 3
e = Addition(Addition(Literal(1), Literal(2)), Literal(3))

v = ExpressionPrintingVisitor()

e.accept(v)





    
