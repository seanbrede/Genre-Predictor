import collections as col
import gzip
import string


def countWords(reviews_raw, train_split):
    punctuation = set(string.punctuation)
    word_counts = col.defaultdict(int)
    for review in reviews_raw[:train_split]:  # only in the TRAINING set
        for word in review["text"].split():
            word = "".join([c for c in word.lower() if c not in punctuation])
            word_counts[word] += 1
    word_counts = [(k, v) for k, v in word_counts.items()]
    word_counts.sort(key=lambda kvp: kvp[1])
    word_counts.reverse()  # now we have the sorted list
    return word_counts


def featurize(review, word_to_index, dict_size):
    features = [0] * dict_size
    for word in review["text"].split():
        word = ''.join([c for c in word.lower() if c not in set(string.punctuation)])
        if word_to_index[word] != -1:
            features[word_to_index[word]] += 1
    # features.append(review['hours'])
    # features.append(len(review['text']))
    return features


def calcAcc(ps, ys):
    right, wrong = 0, 0
    for p, y in zip(ps, ys):
        if p == y: right += 1
        else:      wrong += 1
    return right / (right + wrong)
