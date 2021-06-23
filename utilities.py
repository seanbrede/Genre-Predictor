import collections  as coll
import scipy.sparse as spar
import string       as stri


punctuation = set(stri.punctuation)


def processReview(raw_review):
    review_text = raw_review["text"].lower()
    review_text = "".join([c for c in review_text if c not in punctuation])
    review_text = review_text.split()
    return review_text


def countTotalGrams(raw_reviews, grams):
    gram_counts = [coll.defaultdict(int) for _ in range(grams)]

    for raw_review in raw_reviews:
        # lowercase, remove punctuation, split
        review = processReview(raw_review)

        # count grams in the review and add them to the total
        for i in range(len(review)):
            for g in range(grams):
                if i + g < len(review):
                    entry = "-".join([review[j] for j in range(i, i + g + 1)])
                    gram_counts[g][entry] += 1

    # put the grams into a list of lists sorted by count
    gram_counts_list = []
    for i in range(grams):
        gram_counts_entry = [(gram, count) for gram, count in gram_counts[i].items()]
        gram_counts_entry.sort(key=lambda x: x[1], reverse=True)
        gram_counts_list.append(gram_counts_entry)

    # we now have a list of lists of n-grams, sorted by count
    return gram_counts_list


def buildIndices(gram_counts_list, grams):
    gram_indices = coll.defaultdict(lambda: -1)
    next_index   = 0

    for g in range(len(grams)):
        for i in range(grams[g]):  # TODO fix possible bug if num of grams is larger than length
            gram, _            = gram_counts_list[g][i]
            gram_indices[gram] = next_index
            next_index += 1

    return gram_indices


def extractFeatures(raw_review, grams_indices, nums_grams):
    features = [0] * sum(nums_grams)

    # lowercase, remove punctuation, split
    review = processReview(raw_review)

    for i in range(len(review)):
        for g in range(len(nums_grams)):
            if i + g < len(review):
                entry = "".join([review[j] for j in range(i, i + g + 1)])
                index = grams_indices[g][entry]
                features[index] += 1

    return features


def buildSets(reviews_raw, nums_grams, grams_indices):
    X = spar.lil_matrix((len(reviews_raw), nums_grams))  # TODO fix this, use sum(nums_grams)?
    Y = [r['genreID'] for r in reviews_raw]

    for i in range(len(reviews_raw)):
        features = extractFeatures(reviews_raw[i], grams_indices, nums_grams)
        for j in range(nums_grams):
            if features[j] != 0:
                X[i, j] = features[j]

    return X, Y


def calcAccuracy(preds, ys):
    right, wrong = 0, 0
    for pred, y in zip(preds, ys):
        if pred == y: right += 1
        else:         wrong += 1
    return right / (right + wrong)
