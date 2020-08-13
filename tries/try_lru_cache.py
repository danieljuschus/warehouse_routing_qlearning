import time


class Foo:
    @cache
    def foo(self, x):
        print("executing foo with {}".format(x))
        time.sleep(0.1)


f = Foo()
s = time.time()
[f.foo(i) for i in [1, 2, 1, 4, 1]]
print(time.time()-s)
