import math
import re
import operator
import pandas as pd
import analyze_reddit
from collections import Counter

BUNDLE_REDDITS  = ['Justrolledintotheshop', 'LifeProTips', 'todayilearned',
 'AskReddit', 'aww', 'gaming', 'doctorwho', 'anime', 'worldnews', 'programming']


def create_dict(subreddits):
	'''
	Get dictionary mapping subreddits with lists of strings from title and 
	text of subreddits
	Inputs:
		subreddits(list): subreddit objects

	Return:
		dictionary matching each subreddit name with its string

	'''
	redd_dict = {}
	for sub in subreddits:
		title_lst = sub.posts['title'].to_list()
		text_lst = sub.posts['text'].to_list()
		redd_lst = title_lst + text_lst
		redd_str = ' '.join(redd_lst)
		redd_dict[sub.name] = redd_str

	return redd_dict


def clean_words(subreddits):
	'''
	Extract unique words from the string from each subreddit

	Inputs:
		subreddits(list): subreddit objects
		
	Returns:
		dictionary mapping each subreddit name to a list of its unique words
	'''

	redd_dict = create_dict(subreddits)
	word_dict = {}

	for subreddit, string in redd_dict.items():
		words_list = string.split(" ")
		unique_words = []
		for word in words_list:
			word = word.lower()
			word = re.search("[a-z0-9\-'’]+", word)
			#removed is a common and useless word in the content of a subreddits
			if word != None and word != 'removed':
				word = word.group(0)
				if word not in unique_words:
					unique_words.append(word)
		
		word_dict[subreddit] = unique_words

	return  word_dict


def find_bundle():
	'''
	Find unique words in random bundle of subreddits to train tf_idf

	Inputs:
		bundle_reddits(list): universal variable of a bundle of random subreddits

	Returns:
		list of lists of unique words from bundle subreddits
	'''
	start_date = [2019, 5, 1]
	end_date = [2019, 5, 20]
	n = 1000
	l_sub = []
	for name in BUNDLE_REDDITS:
		l_sub.append(analyze_reddit.Subreddit(name, start_date, end_date, n))
	d = clean_words(l_sub)

	return [d[k] for k in d]


def find_tf(word, lst):
	'''
	Find the tf value for one word in list of words

	Inputs:
		word: a specified word
		lst: a list of words

	Returns: tf value of this word in list
	'''
	freq = 0
	if len(lst) > 0:
		max_freq = Counter(lst).most_common(1)[0][1] 
	for element in lst:
		if word == element:
			freq += 1
	tf = 0.5 + 0.5 * freq / max_freq
	
	return tf


def find_idf(word, bundle):
	'''
	Find the idf value for one word in bundle

	Inputs:
		word: a specified word
		bundle: a list of lists of words

	Returns: idf value of this word in bundle
	'''
	D = 0
	for sub in bundle:
		if word in sub:
			D += 1
	if D == 0:
		D = 1
	
	idf = math.log(len(bundle) / D)
	
	return idf


def find_hot_words(subreddits, k):
	'''
	Find hot words in each subreddit

	Inputs:
		subreddits(list): subreddit objects
		bundle_reddits: universal variable of a bundle of random subreddits
		k: number of hottest words

	Return:
		dictionary mapping each subreddit with its list of k hottest words
	'''

	word_dict = clean_words(subreddits)
	bundle = find_bundle()
	hotwords = {}
	for subreddit, lst in word_dict.items():
		d = {}
		for word in lst:
			one_pair = {word: find_tf(word, lst) * find_idf(word, bundle)}
			d.update(one_pair)
		listofTuples = sorted(d.items() , reverse=True, key=lambda x: x[1])
		top_k = [i for i in listofTuples[: k+1]]
		hotwords[subreddit] = top_k
		
	return hotwords