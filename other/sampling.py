import tensorflow as tf
import constants as const
import tfpy

def constant_boolean_mask_single((tensor, mask), k):
    tensor = tf.boolean_mask(tensor, mask)
    N = tf.shape(tensor)[0]
    indices = sample_indices(k, N)
    out = tf.gather(tensor, indices)
    return out

def constant_boolean_mask(tensor, mask, k, bs):
    return tf.map_fn(
        lambda x: constant_boolean_mask_single(x, k),
        [tensor, mask],
        dtype = tf.float32,
        parallel_iterations = bs
    )

def sample_with_mask_reshape(tensor, mask, sample_count, bs = None):
    if bs is None:
        bs = const.BS
    D = tensor.shape[-1]
    tensor = tf.reshape(tensor, (bs, -1, D))
    mask = tf.reshape(mask, (bs, -1))
    return sample_with_mask(tensor, mask, sample_count, bs = bs)

def sample_with_mask(tensor, mask, sample_count, bs = None):
    
    #tensor is (BS x N x D), mask is (BS x N)
    if bs is None:
        bs = const.BS
    hard_mask = mask > 0.5
    hard_float_mask = tf.cast(hard_mask, dtype = tf.float32)
    k = tf.minimum(tf.cast(tf.reduce_min(tf.reduce_sum(hard_float_mask, axis = 1)), tf.int32), sample_count)
    feats = constant_boolean_mask(tensor, hard_mask, k, bs)
    return feats
        
def sample_indices(k, N):
    randoms = tf.random_uniform(shape = tf.stack([N]))
    topk = tf.nn.top_k(randoms, k = k, sorted = False)
    return topk.indices
