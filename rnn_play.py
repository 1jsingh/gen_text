import tensorflow as tf
import numpy as np
import text_utils as txt
import argparse

# these must match what was saved !
ALPHASIZE = txt.ALPHASIZE
NLAYERS = 3
INTERNALSIZE = 512

author = "checkpoints/rnn_train_1517966804-51000000"

ncnt = 0
with tf.Session() as sess:
	new_saver = tf.train.import_meta_graph('checkpoints/rnn_train_1517966804-0.meta')
	new_saver.restore(sess, author)

	print ("Loaded models")
	x = txt.convert_from_alphabet(ord("L"))
	x = np.array([[x]])  # shape [BATCHSIZE, SEQLEN] with BATCHSIZE=1 and SEQLEN=1

	# initial values
	y = x
	h = np.zeros([1, INTERNALSIZE * NLAYERS], dtype=np.float32)  # [ BATCHSIZE, INTERNALSIZE * NLAYERS]

	for i in range(1000000000):
		yo, h = sess.run(['Yo:0', 'H:0'], feed_dict={'X:0': y, 'pkeep:0': 1., 'Hin:0': h, 'batchsize:0': 1})

		# If sampling is be done from the topn most likely characters, the generated text
		# is more credible and more "english". If topn is not set, it defaults to the full
		# distribution (ALPHASIZE)

		# Recommended: topn = 10 for intermediate checkpoints, topn=2 or 3 for fully trained checkpoints

		c = txt.sample_from_probabilities(yo, topn=2)
		y = np.array([[c]])  # shape [BATCHSIZE, SEQLEN] with BATCHSIZE=1 and SEQLEN=1
		c = chr(txt.convert_to_alphabet(c))
		print(c, end="")

		# shifting the output to the next line if num_char in the line == 100
		if c == '\n':
			ncnt = 0
		else:
			ncnt += 1
		if ncnt == 100:
			print("")
			ncnt = 0

