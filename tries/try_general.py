import inspect


def foo():
    print("Remote foo was called from {}".format(inspect.stack()[1].filename.split("/")[-1]))
