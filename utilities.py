import collections as coll
import string      as stri


def countWords(reviews_raw, train_split):
    punctuation = set(stri.punctuation)
    word_counts = coll.defaultdict(int)

    for review in reviews_raw[:train_split]:  # only in the TRAINING set
        for word in review["text"].split():
            word = "".join([c for c in word.lower() if c not in punctuation])
            word_counts[word] += 1

    word_counts = [(word, count) for word, count in word_counts.items()]
    word_counts.sort(key=lambda x: x[1], reverse=True)
    # we now have a list of words sorted by count
    return word_counts


# TODO
def countTotalGrams(reviews_raw, train_split, grams):
    punctuation = set(stri.punctuation)
    gram_counts = [coll.defaultdict(int) for _ in range(grams)]

    for review in reviews_raw[:train_split]:
        # lowercase, remove punctuation, and split into tokens
        review = review["text"].lower()
        review = "".join([c for c in review if c not in punctuation])
        review = review.split()

        # count all grams in the review
        for i in range(len(review)):
            for g in range(grams):
                if i + g < len(review):
                    entry = "".join([review[j] for j in range(i, i + g + 1)])
                    gram_counts[g][entry] += 1

    # put the grams into a list and sort by count
    gram_counts_list = []
    for i in range(grams):
        gram_counts_entry = [(gram, count) for gram, count in gram_counts[i]]
        gram_counts_entry.sort(key=lambda x: x[1], reverse=True)

    # we now have a list of lists of n-grams, sorted by count
    return gram_counts_list


def featurize(review, word_to_index, dict_size):
    features = [0] * dict_size
    for word in review["text"].split():
        word = ''.join([c for c in word.lower() if c not in set(stri.punctuation)])
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
