import time
import ray
import pylru


@ray.remote
class Foo:
    def __init__(self):
        self.cache = pylru.lrucache(5)

    def get_cache(self):
        return list(self.cache.items())

    def foo(self, x):
        def get_value(x):
            print("Getting " + str(x))
            time.sleep(0.1)
            return x+1
        if str(x) not in self.cache:
            self.cache[str(x)] = get_value(x)
        return self.cache[str(x)]


ray.init()
f = Foo.remote()
s = time.time()
ray.get([f.foo.remote(x=i) for i in [1, 2, 1]])
print(time.time()-s)
c = ray.get(f.get_cache.remote())
ray.shutdown()
