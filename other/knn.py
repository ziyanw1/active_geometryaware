import tensorflow as tf
import random
import tfpy

def batch_knn(points, iters = 2, k = 3, mask = None, tries = 1):
    if mask is None:
        fn = lambda (x, m): repeat_knn(x, iters = iters, k = k, tries = tries)
        args = points
    else:
        fn = lambda (x, m): repeat_knn(x, iters = iters, k = k, mask = m, tries = tries)
        args = [points, mask]
    return tf.map_fn(fn, args, dtype = [tf.float32]*k)

def repeat_knn(points, iters = 2, k = 3, mask = None, tries = 1):
    centerss = []
    costs = []
    for t in range(tries):
        centers, cost = knn(points, iters = iters, k = k, mask = mask, retcost = True)
        centerss.append(centers)
        costs.append(cost)
    cost = tf.stack(costs, axis = 0) #T
    best_idx = tf.argmin(cost)
    centers = [tf.stack(c, axis = 0) for c in zip(*centerss)] #k of T x D
    centers = [tf.gather(c, best_idx) for c in centers] #k of D
    return centers

def knn(points, iters = 2, k = 3, mask = None, retcost = False):
    #this is a pretty crude implementation
    #which creates new tensors for every additional iteration
    #should work ok for small #iters

    if mask is not None:
        points = tf.boolean_mask(points, mask)

    def dist_mat(pts1, pts2):
        #(x-y)^2 = x^2 + y^2 - 2xy
        xy = tf.matmul(pts1, tf.transpose(pts2))  # n by m
        xsq = tf.expand_dims(tf.reduce_sum(tf.square(pts1), axis=1), axis=1)  # n by 1
        ysq = tf.expand_dims(tf.reduce_sum(tf.square(pts2), axis=1), axis=0)  # 1 by m
        return xsq + ysq - 2 * xy

    if True:
        def find_center(pts): #K centers
            return (tf.reduce_max(pts, axis = 0) + tf.reduce_min(pts, axis = 0)) / 2.0
    else:
        def find_center(pts): #K means
            return tf.reduce_mean(pts, axis = 0)
    
    #initialization
    idx = tf.random_uniform(shape = (), maxval = tf.shape(points)[0], dtype = tf.int32)
    centers = [tf.gather(points, idx)]
    for j in range(k):
        dists = dist_mat(centers, points)
        idx = tf.argmax(tf.reduce_min(dists, axis = 0))
        centers.append(tf.gather(points, idx))
        if j == 0:
            centers.pop(0)

    #iteration
    for i in range(iters):
        dists = dist_mat(centers, points)
        labels = tf.cast(tf.argmin(dists, axis = 0), tf.int32)
        #if i == iters-1:
        #    labels = tfpy.print_val(labels, 'labels')
        clusters = tf.dynamic_partition(points, labels, k)
        centers = map(find_center, clusters)

    cost = tf.reduce_max(tf.reduce_min(dists, axis = 0))
    if retcost:
        return centers, cost
    return centers
