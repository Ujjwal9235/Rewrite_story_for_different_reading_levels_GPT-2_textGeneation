import numpy as np
import pandas as pd
import glob
import re

text_lists = glob.glob('./new/*.txt', recursive=True)
text_info = {"text_name": [], "num_of_sentence": [],
             "num_word_mean": [], "num_word_std": [],
             "unique_word_mean": [], "unique_word_std": [],
             "repeat_word_mean": [], "repeat_word_std": [],
             "repeat_word_idx_std": [], "perplexity": []}
for text_name in text_lists:
  with open(text_name, 'r') as f:
    text = f.read().strip().replace("?", ".").replace("!", ".").replace('"', '')
  # create a list of sentences in the text
  sentences = re.split(r'\.+', text.replace('\n', '').lower())
  sentences = [sentence for sentence in sentences if len(sentence) > 1]
  sentences = [sentence.strip().replace(',', ' ') for sentence in sentences]
  num_of_sentence = len(sentences)
  # create a word list for each sentence
  words = [sentence.strip().split() for sentence in sentences]
  # number of words per sentence
  num_word_mean = np.mean(np.array([len(word) for word in words]))
  num_word_std = np.std(np.array([len(word) for word in words]), ddof=1)
  # print("{:f} {:f} {:f}".format(num_of_sentence, num_word_mean, num_word_std))

  # we can do word count by sentences (each sentence counts at most 1 occurrence)
  # or we do not limit the number of occurrences in each sentence
  word_dict = {}
  repeat_word_idx = {}
  for sen_words in words:
    for i in range(len(sen_words)):
      word = sen_words[i]
      if word not in word_dict.keys():
        word_dict[word] = 1
        repeat_word_idx[word] = [i]
      else:
        word_dict[word] += 1
        repeat_word_idx[word].append(i)
  word_count = np.array(list(word_dict.values()))
  # print(word_count)
  # compute how the repeated word dispersed in the sentences by indices.
  unique_word_indices = list(repeat_word_idx.values())
  repeat_word_indices = [np.array(list1) for list1 in unique_word_indices if len(list1) >= 2]
  repeat_word_indices_std = np.mean([np.std(list1) for list1 in repeat_word_indices])
  # print(repeat_word_idx)
  # print(repeat_word_indices_std)
  
  # mean of the number of repeated words normalized by the number of sentences
  # (when the number of sentences grows, the chance to get repeated
  # words grows, so we do a normalization.)
  repeat_word_mean = np.mean(word_count) / num_of_sentence
  repeat_word_std = np.std(word_count, ddof=1) / num_of_sentence
  word_count_ex1 = np.array([count for count in word_count if count != 1])
  repeat_word_ex1_mean = np.mean(word_count_ex1) / num_of_sentence
  repeat_word_ex1_std = np.std(word_count_ex1, ddof=1) / num_of_sentence
  # print("{:f} {:f}".format(repeat_word_mean, repeat_word_std))
  # print("{:f} {:f}".format(repeat_word_ex1_mean, repeat_word_ex1_std))

  # perplexity for a unigram model with independent sentences
  perplexity_sum = np.log(word_count / np.sum(word_count)) @ word_count
  perplexity = np.exp(- perplexity_sum / np.sum(word_count))

  text_info["text_name"].append(text_name.rstrip('.txt').lstrip('./new/'))
  text_info["num_of_sentence"].append(num_of_sentence)
  text_info["num_word_mean"].append(num_word_mean)
  text_info["num_word_std"].append(num_word_std)
  text_info["unique_word_mean"].append(repeat_word_mean)
  text_info["unique_word_std"].append(repeat_word_std)
  text_info["repeat_word_mean"].append(repeat_word_ex1_mean)
  text_info["repeat_word_std"].append(repeat_word_ex1_std)
  text_info["repeat_word_idx_std"].append(repeat_word_indices_std)
  text_info["perplexity"].append(perplexity)

text_info_df = pd.DataFrame(text_info)
text_info_df.to_csv('text_info.csv', index=False)
# print(text_info_df.head(10))


