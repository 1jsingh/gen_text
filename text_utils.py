# encoding: UTF-8
# Copyright 2017 Google.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import glob
import sys

# size of the alphabet that we work with
ALPHASIZE = 98


# Specification of the supported alphabet (subset of ASCII-7)
# 10 line feed LF
# 32-64 numbers and punctuation
# 65-90 upper-case letters
# 91-97 more punctuation
# 97-122 lower-case letters
# 123-126 more punctuation
def convert_from_alphabet(a):
    """Encode a character
    :param a: one character
    :return: the encoded value
    """
    if a == 9:
        return 1
    if a == 10:
        return 127 - 30  # LF
    elif 32 <= a <= 126:
        return a - 30
    else:
        return 0  # unknown


# encoded values:
# unknown = 0
# tab = 1
# space = 2
# all chars from 32 to 126 = c-30
# LF mapped to 127-30
def convert_to_alphabet(c, avoid_tab_and_lf=False):
    """Decode a code point
    :param c: code point
    :param avoid_tab_and_lf: if True, tab and line feed characters are replaced by '\'
    :return: decoded character
    """
    if c == 1:
        return 32 if avoid_tab_and_lf else 9  # space instead of TAB
    if c == 127 - 30:
        return 92 if avoid_tab_and_lf else 10  # \ instead of LF
    if 32 <= c + 30 <= 126:
        return c + 30
    else:
        return 0  # unknown


def encode_text(s):
    """Encode a string.
    :param s: a text string
    :return: encoded list of code points
    """
    return list(map(lambda a: convert_from_alphabet(ord(a)), s))


def decode_to_text(c, avoid_tab_and_lf=False):
    """Decode an encoded string.
    :param c: encoded list of code points
    :param avoid_tab_and_lf: if True, tab and line feed characters are replaced by '\'
    :return:
    """
    return "".join(map(lambda a: chr(convert_to_alphabet(a, avoid_tab_and_lf)), c))



def read_data_files(data_path):
    fulltext = ""
    filelist = glob.glob(data_path, recursive=True)
    for filename in filelist:
        file = open(filename,'r')
        print ("Loading {}".format(filename))
        filetext = file.read()
        fulltext += filetext
    codetext = encode_text(fulltext)
    print ("\n Loaded {} files from {} with {} characters".format(len(filelist),data_path,len(codetext)))
    return codetext

def rnn_minibatch_sequencer(raw_data, batch_size, sequence_size, nb_epochs):
    """
    Divides the data into batches of sequences so that all the sequences in one batch
    continue in the next batch. This is a generator that will keep returning batches
    until the input data has been seen nb_epochs times. Sequences are continued even
    between epochs, apart from one, the one corresponding to the end of raw_data.
    The remainder at the end of raw_data that does not fit in an full batch is ignored.
    :param raw_data: the training text
    :param batch_size: the size of a training minibatch
    :param sequence_size: the unroll size of the RNN
    :param nb_epochs: number of epochs to train on
    :return:
        x: one batch of training sequences
        y: on batch of target sequences, i.e. training sequences shifted by 1
        epoch: the current epoch number (starting at 0)
    """
    data = np.array(raw_data)
    data_len = data.shape[0]
    # using (data_len-1) because we must provide for the sequence shifted by 1 too
    nb_batches = (data_len - 1) // (batch_size * sequence_size)
    assert nb_batches > 0, "Not enough data, even for a single batch. Try using a smaller batch_size."
    rounded_data_len = nb_batches * batch_size * sequence_size
    xdata = np.reshape(data[0:rounded_data_len], [batch_size, nb_batches * sequence_size])
    ydata = np.reshape(data[1:rounded_data_len + 1], [batch_size, nb_batches * sequence_size])

    for epoch in range(nb_epochs):
        for batch in range(nb_batches):
            x = xdata[:, batch * sequence_size:(batch + 1) * sequence_size]
            y = ydata[:, batch * sequence_size:(batch + 1) * sequence_size]
            x = np.roll(x, -epoch, axis=0)  # to continue the text from epoch to epoch (do not reset rnn state!)
            y = np.roll(y, -epoch, axis=0)
            yield x, y, epoch

def print_text_generation_header():
    print()
    print("┌{:─^111}┐".format('Generating random text from learned state'))


def print_text_generation_footer():
    print()
    print("└{:─^111}┘".format('End of generation'))

def sample_from_probabilities(probabilities, topn=ALPHASIZE):
    """Roll the dice to produce a random integer in the [0..ALPHASIZE] range,
    according to the provided probabilities. If topn is specified, only the
    topn highest probabilities are taken into account.
    :param probabilities: a list of size ALPHASIZE with individual probabilities
    :param topn: the number of highest probabilities to consider. Defaults to all of them.
    :return: a random integer
    """
    p = np.squeeze(probabilities)
    p[np.argsort(p)[:-topn]] = 0
    p = p / np.sum(p)
    return np.random.choice(ALPHASIZE, 1, p=p)[0]



            