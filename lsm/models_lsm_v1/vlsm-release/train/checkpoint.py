import os
import numpy as np
from tensorflow.python import pywrap_tensorflow
import tensorflow as tf

def rename(ckp_path, replace_from, replace_to):
    checkpoint = tf.train.get_checkpoint_state(ckp_path)
    with tf.Session() as sess:
        for var_name, _ in tf.contrib.framework.list_variables(ckp_path):
            var = tf.contrib.framework.load_variable(ckp_path, var_name)

            new_name = var_name.replace(replace_from, replace_to)
            print('%s would be renamed to %s.' % (var_name, new_name))
            print var.shape
            if len(var.shape)==1 and var.shape[0] == 3:
                print var
            if len(var.shape) > 0 and var.shape[0] == 1:
                var = np.squeeze(var)
            else:
                pass

            #if 'fc' in new_name and 'bias' not in new_name:
            #    var = np.reshape(var, (-1, var.shape[-1]))
        
            var = tf.Variable(var, name=new_name)
            print var
        #saver = tf.train.Saver()
        #sess.run(tf.global_variables_initializer())
        #saver.save(sess, ckp_path)

model_dir = './'
checkpoint_path = os.path.join(model_dir, "mvnet-100000.data-00000-of-00001")
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
for key in var_to_shape_map:
    print("tensor_name: ", key)
    #print(reader.get_tensor(key)) # Remove this is you want to print only variable names

#rename('./vgg_16_copy.ckpt', 'biases', 'bias')
