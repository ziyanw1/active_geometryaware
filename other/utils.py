import os
import copy
import psutil
import numpy as np

def nyi():
    raise Exception, 'not yet implemented'


def ensure(path):
    if not os.path.exists(path):
        os.makedirs(path)

def exchange_scope(name, scope, oldscope):
    head, tail = name.split(scope)
    assert head == ''
    return oldscope + tail


def onmatrix():
    return 'Linux compute' in os.popen('uname -a').read()


def iscontainer(obj):
    return isinstance(obj, list) or isinstance(obj, dict) or isinstance(obj, tuple)


def strip_container(container, fn=lambda x: None):
    assert iscontainer(container), 'not a container'

    if isinstance(container, list) or isinstance(container, tuple):
        return [(strip_container(obj, fn) if iscontainer(obj) else fn(obj))
                for obj in container]
    else:
        return {k: (strip_container(v, fn) if iscontainer(v) else fn(v))
                for (k, v) in container.items()}


def memory_consumption():
    #print map(lambda x: x/1000000000.0, list(psutil.Process(os.getpid()).memory_info()))
    return psutil.Process(os.getpid()).memory_info().rss / (1024.0**3)
    #return psutil.virtual_memory().used / 1000000000.0 #oof


 

def check_numerics(stuff):
    if isinstance(stuff, dict):
        for k in stuff:
            if not check_numerics(stuff[k]):
                raise Exception, 'not finite %s', k
        return True
    elif isinstance(stuff, list) or isinstance(stuff, tuple):
        for x in stuff:
            check_numerics(x)
    else:
        return np.isfinite(stuff).all()
