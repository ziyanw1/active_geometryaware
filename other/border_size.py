#!/usr/bin/env python3
import numpy as np

def should_merge(x1, x2, ratio):
    b_size = border_size(x1, x2)
    min_volume = min(volume(x1), volume(x2))
    #print('actual ratio is', b_size / min_volume)
    return b_size / min_volume > ratio

def volume(x):
    return np.sum(x.astype(np.float32))

def border_size(x1, x2):
    #assume both are S^3
    ex1 = expand(x1, range(3))
    ex2 = expand(x2, range(3))
    intersection = np.minimum(ex1, ex2).astype(np.float32)
    return np.sum(intersection)

def expand(x, axes):
    xs = [x]
    for a in axes: 
        for d in [-1, 1]:
            xs.append(shift(x, a, d))
    xs = np.stack(xs, axis = 0)
    return np.max(xs, axis = 0)

def shift(x, axis, direction):
    x = np.swapaxes(x, axis, 0)
    if direction == -1:
        x = np.flip(x, 0)

    x_trim = x[1:]
    tail = np.zeros_like(x[0:1])
    x = np.concatenate([x_trim, tail], axis = 0)
        
    if direction == -1:
        x = np.flip(x, 0)
    x = np.swapaxes(x, axis, 0)

    return x
