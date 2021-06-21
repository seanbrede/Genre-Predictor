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
def countGrams(reviews_raw, train_split, grams):
    punctuation = set(stri.punctuation)
    gram_counts = [coll.defaultdict(int) for _ in range(grams)]

    for review in reviews_raw[:train_split]:
        review = review["text"].lower()
        review = "".join([c for c in review if c not in punctuation])
        review = review.split()

        for i in range(len(review)):
            for gram in range(grams):
                if i + gram < len(review):
                    entry = "".join([review[j] for j in range(i, i + gram + 1)])
                    gram_counts[gram][entry] += 1

    # derp
    word_counts = [(word, count) for word, count in gramlist for gramlist in gram_counts]
    word_counts.sort(key=lambda x: x[1], reverse=True)

    # we now have a list of words sorted by count
    return word_counts


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
