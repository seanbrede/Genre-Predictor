from collections import defaultdict
import gzip
import string

def LoadData(path):
    raw_data = [eval(line) for line in gzip.open(path, 'r+')]
    return raw_data

def CalcAcc(ps, ys):
    right, wrong = 0, 0
    for p, y in zip(ps, ys):
        if p == y: right += 1
        else:      wrong += 1
    return right / (right + wrong)

def FeaturizeTest(review, word_indices, batch, batch_size):
    lower = (batch+1) * batch_size  # lower threshold of word popularity to draw from
    upper = lower + batch_size      # upper threshold of word popularity to draw from
    features = [0] * (2 * batch_size)
    for word in review['text'].split():
        word = ''.join([c for c in word.lower() if not c in set(string.punctuation)])
        # if that word exists in the current corpus, keep a tally of it
        if 0 <= word_indices[word] < batch_size:
            features[word_indices[word]] += 1
        elif lower <= word_indices[word] < upper:
            features[word_indices[word] - (batch * batch_size)] += 1
    return features

def FeaturizeTrain(review, feature_dict, add_words):
    features = [0] * (500+add_words)
    for word in review['text'].split():
        word = ''.join([c for c in word.lower() if not c in set(string.punctuation)])
        # if that word exists in the current corpus, keep a tally of it
        if feature_dict[word] != -1:
            features[feature_dict[word]] += 1
    features.append(review['hours'])
    features.append(len(review['text']))
    return features

def CenterData(matrix):
    pass

def CountWords(reviews, tv_split):
    wordCounts = defaultdict(int)
    for review in reviews[:tv_split]:
        for word in review['text'].split():
            word = ''.join([c for c in word.lower() if not c in set(string.punctuation)])
            wordCounts[word] += 1
    wordCounts = [(k, v) for k, v in wordCounts.items()]
    wordCounts.sort(key=lambda kvp: kvp[1])
    wordCounts.reverse()
    return wordCounts

def BuildWordIndexDict(word_counts, top_words):
    word_indices = defaultdict(lambda: -1)
    for i in range(top_words):
        k, _ = word_counts[i]
        word_indices[k] = i
    return word_indices
