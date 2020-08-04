import time
import ray


@ray.remote
class Main:
    def __init__(self):
        self.attr = "b"

    def send_attr(self, i=1):
        print("Sending {}".format(self.attr*i))
        return self.attr*i

    def train(self, slaves):
        ray.get([slave.run.remote() for slave in slaves])


@ray.remote
class Worker:
    def __init__(self, main, id):
        self.main = main
        self.id = id

    def receive_attr(self):
        print("runnning submethod...")
        time.sleep(self.id)
        res = ray.get(self.main.send_attr.remote(i=self.id))
        print("{} received {}".format(self.id, res))
        # return res

    def run(self):
        print("running...")
        self.receive_attr()


ray.init()
main = Main.remote()
workers = [Worker.remote(main=main, id=i) for i in range(5)]
s = time.time()
res = ray.get([worker.run.remote() for worker in workers])
e = time.time() - s
# ray.shutdown()
