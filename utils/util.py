import numpy as np
import tensorflow as tf
import os
import termcolor

# create directory
def mkdir(path):
	if not os.path.exists(path): os.mkdir(path)

# convert to colored strings
def toRed(content): return termcolor.colored(content,"red",attrs=["reverse", "blink"])
def toGreen(content): return termcolor.colored(content,"green",attrs=["reverse", "blink"])
def toBlue(content): return termcolor.colored(content,"blue",attrs=["reverse", "blink"])
def toCyan(content): return termcolor.colored(content,"cyan",attrs=["bold"])
def toYellow(content): return termcolor.colored(content,"yellow",attrs=["bold"])
def toMagenta(content): return termcolor.colored(content,"magenta",attrs=["bold"])

# make image summary from image batch
def imageSummary(tag,image,p,H,W):
	blockSize = p.visBlockSize
	imageOne = tf.batch_to_space(image[:blockSize**2],crops=[[0,0],[0,0]],block_size=blockSize)
	imagePermute = tf.reshape(imageOne,[H,blockSize,W,blockSize,-1])
	imageTransp = tf.transpose(imagePermute,[1,0,3,2,4])
	imageBlocks = tf.reshape(imageTransp,[1,H*blockSize,W*blockSize,-1])
	summary = tf.summary.image(tag,imageBlocks)
	return summary

# create TF saver
def tfSaver(name,maxKeep=1):
	saver = tf.train.Saver(var_list=[v for v in tf.global_variables() if name in v.name],max_to_keep=maxKeep)
	return saver

# restore model
def restoreModel(p,sess,saver,name,i=None):
	if i is None:
		saver.restore(sess,"{0}_{1}.ckpt".format(p.load,name))
	else:
		saver.restore(sess,"models_{2}/{0}_it{1}k_{3}.ckpt".format(p.model,i//1000,p.group,name))

# save model
def saveModel(p,sess,saver,name,i=None,interm=False):
	if i is None:
		saver.save(sess,"models_{1}/final/{0}_{2}.ckpt".format(p.model,p.group,name))
	elif interm:
		saver.save(sess,"models_{2}/interm/{0}_it{1}k_{3}.ckpt".format(p.model,(i+1)//1000,p.group,name))
	else:
		saver.save(sess,"models_{2}/{0}_it{1}k_{3}.ckpt".format(p.model,(i+1)//1000,p.group,name))


