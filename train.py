import utils
import argparse
import numpy as np
import URLNet as unet

parser = argparse.ArgumentParser("Train model")

# Data args
default_max_len_words = 200
parser.add_argument('--data.max_len_words', type=int, default=default_max_len_words, metavar="MLW",
                    help="maximum length of url in words (default: {})".format(default_max_len_words))
default_max_len_chars = 200
parser.add_argument('--data.max_len_chars', type=int, default=default_max_len_chars, metavar="MLC",
                    help="maximum length of url in characters (default: {})".format(default_max_len_chars))
default_max_len_subwords = 20
parser.add_argument('--data.max_len_subwords', type=int, default=default_max_len_subwords, metavar="MLSW",
                    help="maximum length of word in sub-words/ characters (default: {})".format(
                        default_max_len_subwords))
default_min_word_freq = 1
parser.add_argument('--data.min_word_freq', type=int, default=default_min_word_freq, metavar="MWF",
                    help="minimum frequency of word in training population to build vocabulary (default: {})".format(
                        default_min_word_freq))
default_dev_pct = 0.001
parser.add_argument('--data.dev_pct', type=float, default=default_dev_pct, metavar="DEVPCT",
                    help="percentage of training set used for dev (default: {})".format(default_dev_pct))
default_malicious_data = "IntegratedData/malicious.txt"
parser.add_argument('--data.malicious_data', type=str, default=default_malicious_data, metavar="MD",
                    help="location of malicious data file")
default_benign_data = "IntegratedData/benign.txt"
parser.add_argument('--data.benign_data', type=str, default=default_benign_data, metavar="BD",
                    help="location of benign data file")
default_delimit_mode = 1
parser.add_argument("--data.delimit_mode", type=int, default=default_delimit_mode, metavar="DLMODE",
                    help="0: delimit by special chars, 1: delimit by special chars + each char as a word (default: {})".format(
                        default_delimit_mode))

# Model args
default_emb_dim = 32
parser.add_argument('--model.emb_dim', type=int, default=default_emb_dim, metavar="EMBDIM",
                    help="embedding dimension size (default: {})".format(default_emb_dim))
default_filter_sizes = "3,4,5,6"
parser.add_argument('--model.filter_sizes', type=str, default=default_filter_sizes, metavar="FILTERSIZES",
                    help="filter sizes of the convolution layer (default: {})".format(default_filter_sizes))
default_emb_mode = 1
parser.add_argument('--model.emb_mode', type=int, default=default_emb_mode, metavar="EMBMODE",
                    help="1: charCNN, 2: wordCNN, 3: char + wordCNN, 4: char-level wordCNN, 5: char + char-level wordCNN (default: {})".format(
                        default_emb_mode))

# Train args
default_nb_epochs = 5
parser.add_argument('--train.nb_epochs', type=int, default=default_nb_epochs, metavar="NEPOCHS",
                    help="number of training epochs (default: {})".format(default_nb_epochs))
default_batch_size = 128
parser.add_argument('--train.batch_size', type=int, default=default_batch_size, metavar="BATCHSIZE",
                    help="Size of each training batch (default: {})".format(default_batch_size))
parser.add_argument('--train.l2_reg_lambda', type=float, default=0.0, metavar="L2LREGLAMBDA",
                    help="l2 lambda for regularization (default: 0.0)")
default_lr = 0.001
parser.add_argument('--train.lr', type=float, default=default_lr, metavar="LR",
                    help="learning rate for optimizer (default: {})".format(default_lr))

# Log args
parser.add_argument('--log.output_dir', type=str, default="Model/", metavar="OUTPUTDIR",
                    help="directory of the output model")
parser.add_argument('--log.print_every', type=int, default=50, metavar="PRINTEVERY",
                    help="print training result every this number of steps (default: 50)")
parser.add_argument('--log.eval_every', type=int, default=500, metavar="EVALEVERY",
                    help="evaluate the model every this number of steps (default: 500)")
parser.add_argument('--log.checkpoint_every', type=int, default=500, metavar="CHECKPOINTEVERY",
                    help="save a model every this number of steps (default: 500)")

FLAGS = vars(parser.parse_args())

for key, val in FLAGS.items():
    print("{}={}".format(key, val))

urls, labels = utils.load_data(FLAGS["data.malicious_data"], FLAGS["data.benign_data"])

high_freq_words = None
if FLAGS["data.min_word_freq"] > 0:
    x1, word_reverse_dict = utils.get_vocabulary(urls, FLAGS["data.max_len_words"], FLAGS["data.min_word_freq"])
    high_freq_words = sorted(list(word_reverse_dict.values()))
    print("Number of words with freq >={}: {}".format(FLAGS["data.min_word_freq"], len(high_freq_words)))

x, word_reverse_dict = utils.get_vocabulary(urls, FLAGS["data.max_len_words"])
word_x = utils.get_words(x, word_reverse_dict, FLAGS["data.delimit_mode"], urls)
ngramed_id_x, ngrams_dict, worded_id_x, words_dict = utils.ngram_id_x(word_x, FLAGS["data.max_len_subwords"], high_freq_words)

chars_dict = ngrams_dict
chared_id_x = utils.char_id_x(urls, chars_dict, FLAGS["data.max_len_chars"])

pos_x = []
neg_x = []
for i in range(len(labels)):
    label = labels[i]
    if label == 1:
        pos_x.append(i)
    else:
        neg_x.append(i)
print("Overall Mal/Ben split: {}/{}".format(len(pos_x), len(neg_x)))
pos_x = np.array(pos_x)
neg_x = np.array(neg_x)

x_train, y_train, x_test, y_test = utils.prep_train_test(pos_x, neg_x, FLAGS["data.dev_pct"])

x_train_char = utils.get_ngramed_id_x(x_train, ngramed_id_x)
x_test_char = utils.get_ngramed_id_x(x_test, ngramed_id_x)

x_train_word = utils.get_ngramed_id_x(x_train, worded_id_x)
x_test_word = utils.get_ngramed_id_x(x_test, worded_id_x)

x_train_char_seq = utils.get_ngramed_id_x(x_train, chared_id_x)
x_test_char_seq = utils.get_ngramed_id_x(x_test, chared_id_x)

# Training

