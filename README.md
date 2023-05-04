**Detect Malicious Urls via Deep Learning Network**
===================================================

Introduction
------------
This is an application of URLNet - Learning a URL Representation with Deep Learning for Malicious URL Detection https://arxiv.org/abs/1802.03162.

URLNet is a convolutional neural network (CNN) based model used to detect malicious URLs. The model exploits features of URL text string at both character and word levels.

Requirements
------------
 - tensorflow 2.12.0

Usage
-----

In training datasets, malicious and benign data store in seperated files, each line includes the URL text string.

In test dataset, all urls store in a csv file with url strings and labels [malicious / benign].

The model can be trained by executing the following command:

```bash
python train.py --data.malicious_data <malicious_data_file_directory> --data.benign_data <benign_data_file_directory> \
--data.dev_pct 0.2 --data.delimit_mode <url_delimit_mode> --data.min_word_freq <url_min_word_freq> \
--model.emb_mode <embedding_mode> --model.emb_dim <size_of_embedding_dimensions> --model.filter_sizes <convolutional_filter_sizes_separated_by_comma> \
--train.nb_epochs <nb_of_training_epochs> --train.batch_size <nb_of_urls_per_batch> \
--log.print_every <print_acc_after_this_nb_steps> --log.eval_every <evaluate_on_dev_set_after_this_nb_steps> --log.checkpoint_every <checkpoint_model_after_this_nb_steps> --log.output_dir <model_output_folder_directory>
```

The training will save all the related word and character dictionaries into an
output folder, and the model checkpoints are saved into `checkpoints/` folder.

The model can be tested by running the following command:

```bash
python test.py --data.data_dir <test_data_file_directory> \ 
--data.word_dict_dir <word_dictionary_directory> --data.subword_dict_dir <character_in_word_dictionary_directory> --data.char_dict_dir <character_dictionary_directory> \
--log.checkpoint_dir <model_checkpoint_directory> --log.output_dir <test_result_directory> \
--model.emb_mode <embedding_mode> --model.emb_dim <nb_of_embedding_dimensions> \
--test.batch_size <nb_of_urls_per_batch>
```

The test will save the test results for each URL, including 3 columns: `label`
(original label), `predict` (prediction label), and `score` (softmax score). The
orders of test results from top to bottom are the same as their orders in the
test dataset. If the score is more than 0.5, prediction label is 1 (Malicious).
Else, the prediction is 0 (Benign).

**Example:**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
label predict score
1 1 0.884
0 0 0.359
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Parameters
----------

Training parameters include:

|**Parameter**|**Description**|**Default**|
|-----------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------|
|data.max_len_words|The maximum number of words in a URL. The URL is either truncated or padded with a `<PADDING>` token to reach this length.|200|
|data.max_len_chars|The maximum number of characters in a URL. The URL is either truncted or padded with a `<PADDING>` character to reach this length.|200|
|data.max_len_subwords|The maximum number of characters in each word in a URL. Each word is either truncated or padded with a `<PADDING>` character to reach this length.|20|
|data.max_tokens|The maximum number of tokens in the vocabulary|100000|
|data.dev_pct|Percentage of training data used for validation|0.001|
|data.malicious_data| Directory of the malicious training dataset|IntegratedData/malicious_train.txt|
|data.benign_data|Directory of the benign training dataset|IntegratedData/benign_train.txt|
|model.emb_dim|Dimension size of word and character embedding.| 32              |
|model.emb_mode|1: only character-based CNN, 2: only word-based CNN, 3: character and word CNN, 4: character-level word CNN, 5: character and character-level word CNN|1|
|model.filter_sizes|Sizes of convolutional filters. If more than one branches of CNN, all will have the same set of filter sizes. Separate the filter sizes by comma.|3,4,5,6|
|train.batch_size|Number of URLs in each training batch|64|
|train.nb_epochs|Number of training epochs|5|
|train.lr|Learning rate|0.001|
|train.l2_reg_lambda|regularization parameter of loss function|0|
|log.output_dir|Output folder to save dictionaries and model checkpoints|Model/runs|
|log.print_every|Output the training loss and accuracy after this number of batches|50|
|log.eval_every|Evaluate the model on the validation set after this number of batches|500|
|log.checkpoint_every|Checkpoint the model after this number of batches. Only save the model checkpoint if the validation loss is improved.|500|

Test parameters include:

|**Parameter**|**Description**|**Default**|
|-----------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------|
|data.max_len_words|The maximum number of words in a URL. The URL is either truncated or padded with a `<PADDING>` token to reach this length.|200|
|data.max_len_chars|The maximum number of characters in a URL. The URL is either truncted or padded with a `<PADDING>` character to reach this length.|200|
|data.max_len_subwords|The maximum number of characters in each word in a URL. Each word is either truncated or padded with a `<PADDING>` character to reach this length.|20|
|data.max_tokens|The maximum number of tokens in the vocabulary|100000|
|data.data_dir|Directory of the test dataset|IntegratedData/test.csv|
|data.word_dict_dir|Directory of the word dictionary file. Dictionary file is in pickle extension `.pickle`|Model/runs/emb1_32dim_minwf1_1conv3456_5ep/subwords_dict.pickle|
|data.char_dict_dir|Directory of the character dictionary file. Dictionary file is in pickle extension `.pickle`|Model/runs/emb1_32dim_minwf1_1conv3456_5ep/chars_dict.pickle|
|data.subword_dict_dir|Directory of the character-in-word dictionary file. Dictionary file is in pickle extension `.pickle`|Model/runs/emb1_32dim_minwf1_1conv3456_5ep/ngrams_dict.pickle|
|model.emb_dim|Dimension size of word and character embedding.|32|
|model.emb_mode|1: only character-based CNN, 2: only word-based CNN, 3: character and word CNN, 4: character-level word CNN, 5: character and character-level word CNN|1|
|test.batch_size|Number of URLs in each test batch|64|
|log.checkpoint_dir|Directory of the model checkpoints|Model/runs/emb1_32dim_minwf1_1conv3456_5ep/checkpoints/|
|log.output_dir|Directory of the test results|Model/runs/emb1_32dim_minwf1_1conv3456_5ep/train_test.txt|

The test parameters such as `model.emb_mode` and `data.max_tokens` have to be consistent with the trained model to get accurate test results.
