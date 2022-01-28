"""
CS 121: Analyzing Election Tweets

VAL ALVERN CUECO LIGO

Analyze module

Functions to analyze tweets.
"""

import unicodedata
import sys

from basic_algorithms import find_top_k, find_min_count, find_salient

##################### DO NOT MODIFY THIS CODE #####################

def keep_chr(ch):
    '''
    Find all characters that are classifed as punctuation in Unicode
    (except #, @, &) and combine them into a single string.
    '''
    return unicodedata.category(ch).startswith('P') and \
        (ch not in ("#", "@", "&"))

PUNCTUATION = " ".join([chr(i) for i in range(sys.maxunicode)
                        if keep_chr(chr(i))])

# When processing tweets, ignore these words
STOP_WORDS = ["a", "an", "the", "this", "that", "of", "for", "or",
              "and", "on", "to", "be", "if", "we", "you", "in", "is",
              "at", "it", "rt", "mt", "with"]

# When processing tweets, words w/ a prefix that appears in this list
# should be ignored.
STOP_PREFIXES = ("@", "#", "http", "&amp")


#####################  MODIFY THIS CODE #####################


############## Part 2 ##############

def generate_list(tweets, entity_desc):
    '''
    Generates list of sub_values

    Inputs:
        tweets: a list of tweets
        entity_desc: a triple such as ("hashtags", "text", True),
          ("user_mentions", "screen_name", False), etc.

    Returns: a list of sub_values
    '''

    ent, sub_ent, case = entity_desc
    sub_values = []
    for ind, _ in enumerate(tweets):
        tweet = tweets[ind]['entities']
        for sub in tweet[ent]:
            sub_values.append(sub[sub_ent])

    for ind, string in enumerate(sub_values):
        if not case:
            sub_values[ind] = string.lower()

    return sub_values

# Task 2.1
def find_top_k_entities(tweets, entity_desc, k):
    '''
    Find the k most frequently occuring entitites.

    Inputs:
        tweets: a list of tweets
        entity_desc: a triple such as ("hashtags", "text", True),
          ("user_mentions", "screen_name", False), etc.
        k: integer

    Returns: list of entities
    '''
    tokens = generate_list(tweets, entity_desc)
    top_k = find_top_k(tokens, k)
    return top_k

# Task 2.2
def find_min_count_entities(tweets, entity_desc, min_count):
    '''
    Find the entitites that occur at least min_count times.

    Inputs:
        tweets: a list of tweets
        entity_desc: a triple such as ("hashtags", "text", True),
          ("user_mentions", "screen_name", False), etc.
        min_count: integer

    Returns: set of entities
    '''

    tokens = generate_list(tweets, entity_desc)
    occurred_min = find_min_count(tokens, min_count)

    return occurred_min

############## Part 3 ##############

# Pre-processing step and representing n-grams

def word_stop(text_list, stop_w):
    '''
    Filters list of stop words

    Inputs:
        text_list: list of strings
        stop_w: list of stop words

    Returns: List of strings filtered of stop words
    '''

    stop_filt = [x for x in text_list if x not in stop_w]

    return stop_filt

def prefix_stop(text_list, prefix):
    '''
    Filters list of words with prefixes

    Inputs:
        text_list: list of strings
        prefix: list of prefixes

    Returns: List of strings filtered of prefixed words
    '''
    pre_filt = [word for word in text_list if not word.startswith(prefix, 0)]

    return pre_filt

def convert(tweets, case_sensitive, stop_w, prefix):
    '''
    Preprocesses tweets

    Inputs:
        tweets: a list of tweets
        case_sensitive: boolean
        stop_w: list of stop words
        prefix: list of prefixes

    Returns: Preprocessed tweets
    '''
    tweets = tweets['abridged_text']
    split_text = tweets.split()
    final_lst = []
    for word in split_text:
        word = word.strip(PUNCTUATION)
        if word == "":
            continue
        if not case_sensitive:
            word = word.lower()
        if word != "":
            final_lst.append(word)

    if stop_w is not None:
        final_lst = word_stop(final_lst, stop_w)

    final_lst = prefix_stop(final_lst, prefix)

    return final_lst

def create_corpus(tweets, n, case_sensitive, stop_w):
    '''
    Creates list corpus of all tweets

    Inputs:
        tweets: a list of tweets
        n: integer for length of individual tuples
        case_sensitive: boolean
        stop_w: corpus for stop_words

    Returns: list of filtered tweets
    '''

    corpus = []
    for ind, _ in enumerate(tweets):
        tweet = tweets[ind]
        converted = convert(tweet, case_sensitive, stop_w, STOP_PREFIXES)
        for word in converted:
            if len(converted) >= n:
                corpus.append(word)

    return corpus

def create_n_grams(tweet, n):
    '''
    Create list of n_grams

    Inputs:
        tweets: a list of tweets
        n: integer for length of individual tuples

    Returns: list of n-grams
    '''
    n_grams = []
    for ind, _ in enumerate(tweet):
        n_grams.append(tuple(tweet[ind:ind + n]))
        if ind + n == len(tweet):
            break

    return n_grams

# Task 3.1
def find_top_k_ngrams(tweets, n, case_sensitive, k):
    '''
    Find k most frequently occurring n-grams.

    Inputs:
        tweets: a list of tweets
        n: integer
        case_sensitive: boolean
        k: integer

    Returns: list of n-grams
    '''

    corpus = create_corpus(tweets, n, case_sensitive, STOP_WORDS)
    n_grams = create_n_grams(corpus, n)
    top_tweets = find_top_k(n_grams, k)

    return top_tweets

# Task 3.2
def find_min_count_ngrams(tweets, n, case_sensitive, min_count):
    '''
    Find n-grams that occur at least min_count times.

    Inputs:
        tweets: a list of tweets
        n: integer
        case_sensitive: boolean
        min_count: integer

    Returns: set of n-grams
    '''

    corpus = create_corpus(tweets, n, case_sensitive, STOP_WORDS)
    n_grams = create_n_grams(corpus, n,)
    occurred_min = find_min_count(n_grams, min_count)

    return occurred_min

# Task 3.3
def find_salient_ngrams(tweets, n, case_sensitive, threshold):
    '''
    Find the salient n-grams for each tweet.

    Inputs:
        tweets: a list of tweets
        n: integer
        case_sensitive: boolean
        threshold: float

    Returns: list of sets of strings
    '''

    n_gram_ls = []
    for ind, _ in enumerate(tweets):
        tweet = tweets[ind]
        converted = convert(tweet, case_sensitive, None, STOP_PREFIXES)
        n_grams = create_n_grams(converted, n)
        n_gram_ls.append(n_grams)

    salient = find_salient(n_gram_ls, threshold)

    return salient
