import tensorflow as tf

def batch_knn(points, iters = 2, k = 3, mask = None):
    if mask is None:
        return tf.map_fn(
            lambda (x, m): knn(x, iters = iters, k = k),
            points,
            dtype = [tf.float32]*k,
        )
    else:
        return tf.map_fn(
            lambda (x, m): knn(x, iters = iters, k = k, mask = m),
            [points, mask],
            dtype = [tf.float32]*k,
        )

def knn(points, iters = 2, k = 3, mask = None):
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

    def find_center(pts):
        return (tf.reduce_max(pts, axis = 0) + tf.reduce_min(pts, axis = 0)) / 2.0
    
    #initialization
    idx = tf.random_uniform(shape = (), maxval = k-1, dtype = tf.int32)
    centers = [tf.gather(points, idx)]
    for j in range(k-1):
        dists = dist_mat(centers, points)
        idx = tf.argmax(tf.reduce_min(dists, axis = 0))
        centers.append(tf.gather(points, idx))

    #iteration
    for i in range(iters):
        dists = dist_mat(centers, points)
        labels = tf.cast(tf.argmin(dists, axis = 0), tf.int32)
        clusters = tf.dynamic_partition(points, labels, k)
        centers = map(find_center, clusters)

    #dists = dist_mat(centers, points)
    #labels = tf.argmin(dists, axis = 0)
    return centers
