class Exp:
    """
    Class representing a statistic expression, using operator overloads to create compound expressions
    """
    def __init__(self, key, operator=None, lhs=None, rhs=None, name=None):
        """
        Initializes an expression instance with the key to an NFL statistic
        :param key: Key for an NFL statistic
        :param operator: Operator for compound expressions (optional)
        :param lhs: Left-hand side operands for compound expressions (optional)
        :param rhs: Right-hand side operands for compound expressions (optional)
        :param name: Explicit name for expression (optional)
        """
        self.key = key
        self.operator = operator if operator is not None else "terminal"
        self.lhs = lhs
        self.rhs = rhs
        self.name = name

    def __mul__(self, other):
        return Exp(None, operator='*', lhs=self, rhs=other)

    def __truediv__(self, other):
        return Exp(None, operator="/", lhs=self, rhs=other)

    def __add__(self, other):
        return Exp(None, operator='+', lhs=self, rhs=other)

    def __sub__(self, other):
        return Exp(None, operator='-', lhs=self, rhs=other)

    def __floordiv__(self, other: str):
        # Floor division, performed by //, renames an expression
        return Exp(self.key, self.operator, self.lhs, self.rhs, other)

    def __str__(self):
        # Default naming implementation
        if self.name is not None:
            return self.name
        if self.operator == 'terminal':
            return self.key
        elif self.operator == '*':
            return self.lhs + "_x_" + self.rhs
        elif self.operator == '/':
            return self.lhs + "_per_" + self.rhs
        elif self.operator == '+':
            return self.lhs + "_plus_" + self.rhs
        elif self.operator == '-':
            return self.lhs + "_minus_" + self.rhs

    def eval(self, data, drives, plays):
        """
        Evaluates a statistic expression from kay-value data and drive/play counts
        :param data: Dictionary mapping stat keys to values
        :param drives: Number of drives represented in data
        :param plays: Number of plays represented in data
        :return: Statistic expression value for given data
        """

        # Recursively evaluate expressions based on operator
        if self.operator == 'terminal':
            if self.key == 'drives':
                return drives
            elif self.key == 'plays':
                return plays
            return data[self.key]

        elif self.operator == '*':
            return self.lhs.eval(data, drives, plays) * self.rhs.eval(data, drives, plays)

        elif self.operator == '/':
            # Prevent zero-division errors
            if self.rhs.eval(data, drives, plays) == 0:
                return 0
            return self.lhs.eval(data, drives, plays) / self.rhs.eval(data, drives, plays)

        elif self.operator == '+':
            return self.lhs.eval(data, drives, plays) + self.rhs.eval(data, drives, plays)

        elif self.operator == '-':
            return self.lhs.eval(data, drives, plays) - self.rhs.eval(data, drives, plays)

    def keys(self):
        """
        Finds all keys required by the expression
        :return: List of unique stat keys required by the expression
        """
        if self.operator == 'terminal':
            if self.key == "drives":
                return ["fixed_drive"]
            if self.key == "plays":
                return ["fixed_drive"]
            return [self.key, "fixed_drive"]
        else:
            return list(set(self.lhs.keys() + self.rhs.keys()))
