import re
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from bisect import bisect_left


def split_url(url):
    if url.startswith("https://"):
        url = url[8:]
    if url.startswith("http://"):
        url = url[7:]
    if url.startswith("www."):
        url = url[4:]
    
    processed_url = re.sub("[\|\\/.=?&:]", " ", url)
    return processed_url


def load_data(malicious_path, benign_path):
    urls = []
    labels = []

    with open(malicious_path) as f:
        for line in f.readlines():
            url = split_url(line)
            urls.append(url)
            labels.append(1)

    with open(benign_path) as f:
        for line in f.readlines():
            url = split_url(line)
            urls.append(url)
            labels.append(0)

    return urls, labels


def load_test_data(path):
    urls = []
    labels = []

    test_df = pd.read_csv(path)

    for _, row in test_df.iterrows():
        url = split_url(row["url"])
        urls.append(url)
        if row["label"] == "benign":
            labels.append(0)
        else:
            labels.append(1)
    
    return urls, labels


def get_word_vocabulary(urls, max_tokens, max_length_words):
    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens = max_tokens,
        standardize = "lower",
        output_mode = "int",
        output_sequence_length = max_length_words
    )
    start = time.time()
    vectorizer.adapt(urls)
    x = vectorizer(urls)
    print("Finished build vocabulary and mapping to x in {}".format(time.time() - start))

    # Get vocabulary list from vectorizer
    # Then turn into a reversed vocabulary dict, which sets id as key and word as value
    vocab_list = vectorizer.get_vocabulary()
    reverse_dict = dict()
    for i in range(len(vocab_list)):
        reverse_dict.setdefault(i, vocab_list[i])
    print("Size of word vocabulary: {}".format(len(reverse_dict)))
    return x, reverse_dict

def get_words(x, reverse_dict, urls=None):
    processed_x = []
    for url in x:
        words = []
        for word_id in url:
            if word_id != 0:
                words.append(reverse_dict[int(word_id)])
            else:
                break
        processed_x.append(words)
    return processed_x


def get_char_ngrams(ngram_len, word):
    word = "<" + word + ">"
    chars = list(word)
    begin_idx = 0
    ngrams = []
    while (begin_idx + ngram_len) <= len(chars):
        end_idx = begin_idx + ngram_len
        ngrams.append("".join(chars[begin_idx:end_idx]))
        begin_idx += 1
    return ngrams


def char_id_x(urls, char_dict, max_len_chars):
    chared_id_x = []
    for url in urls:
        url = list(url)
        url_in_char_id = []
        lenth = min(len(url), max_len_chars)
        for i in range(lenth):
            char = url[i]
            try:
                char_id = char_dict[char]
            except KeyError:
                char_id = 0
            url_in_char_id.append(char_id)
        chared_id_x.append(url_in_char_id)
    return chared_id_x


def is_in(str, x):
    index = bisect_left(str, x)
    if index != len(str) and str[index] == x:
        return True
    else:
        return False


def ngram_id_x(word_x, max_len_subwords):
    char_ngram_len = 1
    all_ngrams = set()
    ngramed_x = []
    all_words = set()
    worded_x = []
    counter = 0
    for url in word_x:
        if counter % 100000 == 0:
            print("Processing #url {}".format(counter))
        counter += 1
        url_in_ngrams = []
        url_in_words = []
        words = url
        for word in words:
            ngrams = get_char_ngrams(char_ngram_len, word)
            if (len(ngrams) > max_len_subwords):
                all_ngrams.update(ngrams[:max_len_subwords])
                url_in_ngrams.append(ngrams[:max_len_subwords])
                all_words.add("<UNK>")
                url_in_words.append("<UNK>")
            else:
                all_ngrams.update(ngrams)
                url_in_ngrams.append(ngrams)
                all_words.add(word)
                url_in_words.append(word)
        ngramed_x.append(url_in_ngrams)
        worded_x.append(url_in_words)
    
    all_ngrams = list(all_ngrams)
    ngrams_dict = dict()
    for i in range(len(all_ngrams)):
        ngrams_dict[all_ngrams[i]] = i + 1  # ngram id = 0 is for padding ngram
    print("Size of ngram vocabulary: {}".format(len(ngrams_dict))) 
    all_words = list(all_words) 
    words_dict = dict() 
    for i in range(len(all_words)): 
        words_dict[all_words[i]] = i + 1  # word id = 0 is for padding word 
    print("Size of word vocabulary: {}".format(len(words_dict)))
    print("Index of <UNK> word: {}".format(words_dict["<UNK>"]))

    ngramed_id_x = []
    for ngramed_url in ngramed_x: 
        url_in_ngrams = []
        for ngramed_word in ngramed_url: 
            ngram_ids = [ngrams_dict[x] for x in ngramed_word] 
            url_in_ngrams.append(ngram_ids) 
        ngramed_id_x.append(url_in_ngrams)  
    worded_id_x = []
    for worded_url in worded_x: 
        word_ids = [words_dict[x] for x in worded_url]
        worded_id_x.append(word_ids) 
    
    return ngramed_id_x, ngrams_dict, worded_id_x, words_dict


def ngram_id_x_from_dict(word_x, max_len_subwords, ngram_dict, word_dict = None): 
    char_ngram_len = 1
    print("Index of <UNK> word: {}".format(word_dict["<UNK>"]))
    ngramed_id_x = [] 
    worded_id_x = [] 
    counter = 0
    if word_dict:
        word_vocab = sorted(list(word_dict.keys()))
    for url in word_x:
        if counter % 100000 == 0: 
            print("Processing url #{}".format(counter))
        counter += 1  
        url_in_ngrams = [] 
        url_in_words = [] 
        words = url
        for word in words:
            ngrams = get_char_ngrams(char_ngram_len, word) 
            if len(ngrams) > max_len_subwords:
                word = "<UNK>"  
            ngrams_id = [] 
            for ngram in ngrams: 
                if ngram in ngram_dict: 
                    ngrams_id.append(ngram_dict[ngram]) 
                else: 
                    ngrams_id.append(0) 
            url_in_ngrams.append(ngrams_id)
            if is_in(word_vocab, word): 
                word_id = word_dict[word]
            else: 
                word_id = word_dict["<UNK>"] 
            url_in_words.append(word_id)
        ngramed_id_x.append(url_in_ngrams)
        worded_id_x.append(url_in_words)
    
    return ngramed_id_x, worded_id_x


def prep_train_test(pos_x, neg_x, dev_pct):
    np.random.seed(10)
    shuffle_indices=np.random.permutation(np.arange(len(pos_x)))
    pos_x_shuffled = pos_x[shuffle_indices]
    dev_idx = -1 * int(dev_pct * float(len(pos_x)))
    pos_train = pos_x_shuffled[:dev_idx]
    pos_test = pos_x_shuffled[dev_idx:]

    np.random.seed(10)
    shuffle_indices=np.random.permutation(np.arange(len(neg_x)))
    neg_x_shuffled = neg_x[shuffle_indices]
    dev_idx = -1 * int(dev_pct * float(len(neg_x)))
    neg_train = neg_x_shuffled[:dev_idx]
    neg_test = neg_x_shuffled[dev_idx:] 

    x_train = np.array(list(pos_train) + list(neg_train))
    y_train = len(pos_train)*[1] + len(neg_train)*[0]
    x_test = np.array(list(pos_test) + list(neg_test))
    y_test = len(pos_test)*[1] + len(neg_test)*[0]

    y_train = tf.keras.utils.to_categorical(y_train, num_classes=2)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=2)

    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(x_train)))
    x_train = x_train[shuffle_indices]
    y_train = y_train[shuffle_indices]

    np.random.seed(10) 
    shuffle_indices = np.random.permutation(np.arange(len(x_test)))
    x_test = x_test[shuffle_indices]
    y_test = y_test[shuffle_indices] 
    
    print("Train Malicious/Benign split: {}/{}".format(len(pos_train), len(neg_train)))
    print("Test Malicious/Benign split: {}/{}".format(len(pos_test), len(neg_test)))
    print("Train/Test split: {}/{}".format(len(y_train), len(y_test)))
    print("Train/Test split: {}/{}".format(len(x_train), len(x_test)))

    return x_train, y_train, x_test, y_test


def get_ngramed_id_x(x_idxs, ngramed_id_x): 
    output_ngramed_id_x = [] 
    for idx in x_idxs:  
        output_ngramed_id_x.append(ngramed_id_x[idx])
    return output_ngramed_id_x


def pad_seq(urls, max_d1=0, max_d2=0, embedding_size=128): 
    if max_d1 == 0 and max_d2 == 0: 
        for url in urls: 
            if len(url) > max_d1: 
                max_d1 = len(url) 
            for word in url: 
                if len(word) > max_d2: 
                    max_d2 = len(word) 
    pad_idx = np.zeros((len(urls), max_d1, max_d2, embedding_size))
    pad_urls = np.zeros((len(urls), max_d1, max_d2))
    pad_vec = [1 for i in range(embedding_size)]
    for d0 in range(len(urls)): 
        url = urls[d0]
        for d1 in range(len(url)): 
            if d1 < max_d1: 
                word = url[d1]
                for d2 in range(len(word)): 
                    if d2 < max_d2: 
                        pad_urls[d0,d1,d2] = word[d2]
                        pad_idx[d0,d1,d2] = pad_vec
    return pad_urls, pad_idx


def pad_seq_in_word(urls, max_d1=0, embedding_size=128):
    if max_d1 == 0: 
        url_lens = [len(url) for url in urls]
        max_d1 = max(url_lens)
    pad_urls = np.zeros((len(urls), max_d1))
    for d0 in range(len(urls)): 
        url = urls[d0]
        for d1 in range(len(url)): 
            if d1 < max_d1: 
                pad_urls[d0,d1] = url[d1]
    return pad_urls 


def softmax(x): 
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum() 


def batch_iter(data, batch_size, num_epochs, shuffle=True): 
    data = np.array(data, dtype=object)
    data_size = len(data) 
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1 
    for epoch in range(num_epochs): 
        if shuffle: 
            shuffle_indices = np.random.permutation(np.arange(data_size)) 
            shuffled_data = data[shuffle_indices]
        else: 
            shuffled_data = data 
        for batch_num in range(num_batches_per_epoch): 
            start_idx = batch_num * batch_size 
            end_idx = min((batch_num+1) * batch_size, data_size)
            yield shuffled_data[start_idx:end_idx]


def save_test_result(labels, all_predictions, all_scores, output_dir): 
    output_labels = []
    for i in labels: 
        if i == 1: 
            output_labels.append(i) 
        else: 
            output_labels.append(0) 
    output_preds = [] 
    for i in all_predictions: 
        if i == 1: 
            output_preds.append(i) 
        else: 
            output_preds.append(0) 
    softmax_scores = [softmax(i) for i in all_scores]
    with open(output_dir, "w") as file: 
        output = "label\tpredict\tscore\n"
        file.write(output)
        for i in range(len(output_labels)): 
            output = str(int(output_labels[i])) + '\t' + str(int(output_preds[i])) + '\t' + str(softmax_scores[i][1]) + '\n' 
            file.write(output)
