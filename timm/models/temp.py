import numpy as np

class classA_base():
    def __init__(self, a):
        self.a = a
        print('In init A base', self.a)

class classA(classA_base):
    def __init__(self, a):
        super(classA, self).__init__(a-1)
        self.a = a
        print('In init A', self.a)

def init(self, b):
    super(self.__class__, self).__init__(b-2)
    self.a = b + 1
    print('In init B', self.a)

classA.__init__ = init

A = classA(1)