
class M:

    def __init__(self):

        self.x = 2

    def foo(self, a, b=None):
        b = self.x if b == None else b
        return a + b


a = M()
print(a.foo(1))
print(a.foo(1, 2))
print(a.foo(1, 3))
print(a.foo(1, b=3))
