import numpy as np 
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import rnn  # rnn stuff temporarily in contrib, moving back to code in TF 1.1
import os

import text_utils as txt
import base_paths as bp
import time
import math


# reading the data files
codetext = txt.read_data_files("./data/*.txt")


### defining parameters
SEQLEN = 30
BATCHSIZE = 200
ALPHASIZE = txt.ALPHASIZE
INTERNALSIZE = 512
NLAYERS = 3
learning_rate = 0.001  # fixed learning rate
dropout_pkeep = 0.8    # some dropout


### defining the tensorflow graph

lr = tf.placeholder(tf.float32, name='lr')  # learning rate
pkeep = tf.placeholder(tf.float32, name='pkeep')  # dropout parameter
batchsize = tf.placeholder(tf.int32, name='batchsize')

# defining the input and output placeholder
X = tf.placeholder(dtype = tf.uint8 , shape=[None,None],name='X') # [ BATCHSIZE, SEQLEN ]
Y_ = tf.placeholder(dtype = tf.uint8 , shape=[None,None],name='Y_') # [ BATCHSIZE, SEQLEN ]

#converting to one_hot encoding
Xo = tf.one_hot(X, depth=ALPHASIZE ,axis=-1,name='Xo') # [ BATCHSIZE, SEQLEN, ALPHASIZE ]
Yo_ = tf.one_hot(Y_, depth=ALPHASIZE ,axis=-1,name='Yo_') # [ BATCHSIZE, SEQLEN, ALPHASIZE ]

### defining the rnn cell

# input state
Hin = tf.placeholder(tf.float32, [None, INTERNALSIZE*NLAYERS], name='Hin')  # [ BATCHSIZE, INTERNALSIZE * NLAYERS]

# using a NLAYERS=3 layers of GRU cells, unrolled SEQLEN=30 times
# dynamic_rnn infers SEQLEN from the size of the inputs Xo

# How to properly apply dropout in RNNs: see README.md
cells = [rnn.GRUCell(INTERNALSIZE) for _ in range(NLAYERS)]
# "naive dropout" implementation
dropcells = [rnn.DropoutWrapper(cell,input_keep_prob=pkeep) for cell in cells]
multicell = rnn.MultiRNNCell(dropcells, state_is_tuple=False)
multicell = rnn.DropoutWrapper(multicell, output_keep_prob=pkeep)  # dropout for the softmax layer

Yr, H = tf.nn.dynamic_rnn(multicell, Xo, dtype=tf.float32, initial_state=Hin)

# Yr: [ BATCHSIZE, SEQLEN, INTERNALSIZE ]
# H:  [ BATCHSIZE, INTERNALSIZE*NLAYERS ] # this is the last state in the sequence

H = tf.identity(H, name='H')  # just to give it a name

# defing the final output layer with softmax
Yflat = tf.reshape(Yr, [-1, INTERNALSIZE])	# [BATCHSIZE x SEQLEN , INTERNALSIZE]
Ylogits = tf.layers.dense(Yflat,ALPHASIZE)	# [BATCHSIZE x SEQLEN , ALPHASIZE]
Yo_flat = tf.reshape(Yo_,[-1, ALPHASIZE]) 	# [BATCHSIZE x SEQLEN , ALPHASIZE]

# predicting the output of the network
Yo = tf.nn.softmax(Ylogits, name='Yo')        # [ BATCHSIZE x SEQLEN, ALPHASIZE ]
Y = tf.argmax(Yo, 1)                          # [ BATCHSIZE x SEQLEN ]
Y = tf.reshape(Y, [batchsize, -1], name="Y")  # [ BATCHSIZE, SEQLEN ]

# defining loss and train_op
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Yo_flat))
train_step  = tf.train.AdamOptimizer(lr).minimize(loss)

# Init for saving models. They will be saved into a directory named 'checkpoints'.
# Only the last checkpoint is kept.
timestamp = str(math.trunc(time.time()))

if not os.path.exists("checkpoints"):
	os.mkdir("checkpoints")
saver = tf.train.Saver(max_to_keep=1000)


# for display: init the progress bar
DISPLAY_FREQ = 50
_50_BATCHES = DISPLAY_FREQ * BATCHSIZE * SEQLEN

# init
istate = np.zeros([BATCHSIZE, INTERNALSIZE*NLAYERS])  # initial zero input state
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
step = 0

print ("Starting training ... ")

### training loop
for x, y_, epoch in txt.rnn_minibatch_sequencer(codetext, BATCHSIZE, SEQLEN, nb_epochs=10):
	
	# train on one minibatch
	feed_dict = {X: x, Y_: y_, Hin: istate, lr: learning_rate, pkeep: dropout_pkeep, batchsize: BATCHSIZE}
	_, loss_value, y ,ostate = sess.run([train_step,loss,Y,H], feed_dict=feed_dict)

	# showing loss value
	if step % _50_BATCHES == 0:
		print ("STEP : {} , EPOCH : {} , loss : {}".format(step,epoch+1,loss_value))

	# display a short text generated with the current weights and biases (every 150 batches)
	if step // 3 % _50_BATCHES == 0:
		txt.print_text_generation_header()
		ry = np.array([[txt.convert_from_alphabet(ord("K"))]])
		rh = np.zeros([1, INTERNALSIZE * NLAYERS])
		for k in range(1000):
			ryo, rh = sess.run([Yo, H], feed_dict={X: ry, pkeep: 1.0, Hin: rh, batchsize: 1})
			rc = txt.sample_from_probabilities(ryo, topn=10 if epoch <= 1 else 2)
			print(chr(txt.convert_to_alphabet(rc)), end="")
			ry = np.array([[rc]])
		txt.print_text_generation_footer()

	# save a checkpoint (every 500 batches)
	if step // 10 % _50_BATCHES == 0:
		saved_file = saver.save(sess, 'checkpoints/rnn_train_' + timestamp, global_step=step)
		print("Saved file: " + saved_file)

	
	# loop state around
	istate = ostate
	step += BATCHSIZE * SEQLEN






