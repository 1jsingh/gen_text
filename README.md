# gen_text

Tensorflow based implementation for char based text generation

usage: rnn_train.py [-h] [-s SEQLEN] [-b BATCHSIZE] [-hs HIDDENSIZE]
                    [-lr LEARNING_RATE] [-dr DROPOUT] [-nl {2,3,4}]

train character level GRU text_generation model

optional arguments:
  -h, --help            show this help message and exit
  -s SEQLEN, --seqlen SEQLEN
                        sequence length
  -b BATCHSIZE, --batchsize BATCHSIZE
                        batch size
  -hs HIDDENSIZE, --hiddensize HIDDENSIZE
                        hidden state size of the GRU
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                        learning rate
  -dr DROPOUT, --dropout DROPOUT
                        dropout value used in the GRU
  -nl {2,3,4}, --num_layers {2,3,4}
                        number of GRU layers
