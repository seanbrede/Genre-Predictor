from collections      import defaultdict
from sklearn          import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn          import preprocessing
from scipy.sparse     import lil_matrix
import gzip
import string
import warnings


def load_data(path):
    raw_data = [eval(line) for line in gzip.open(path, 'r+')]
    return raw_data

def calc_acc(ps, ys):
    right, wrong = 0, 0
    for p, y in zip(ps, ys):
        if p == y: right += 1
        else:      wrong += 1
    return right / (right + wrong)

def featurize(review, indexFromWord, dictSize):
    features = [0] * dictSize
    for word in review['text'].split():
        word = ''.join([c for c in word.lower() if not c in set(string.punctuation)])
        if indexFromWord[word] != -1:
            features[indexFromWord[word]] += 1
    #features.append(review['hours'])
    #features.append(len(review['text']))
    return features


# settings
trainSplit = 165000
dictSize   = 35000
warnings.filterwarnings('ignore')

# load the data
reviews = load_data('train_Category.json.gz')

# build a list of all words in the training set and their counts, sorted by order
punctuation = set(string.punctuation)
wordCount = defaultdict(int)
for review in reviews[:trainSplit]:  # only in the TRAINING set
    for word in review['text'].split():
        word = ''.join([c for c in word.lower() if not c in punctuation])
        wordCount[word] += 1
wordCount = [(k, v) for k, v in wordCount.items()]
wordCount.sort(key=lambda kvp: kvp[1])
wordCount.reverse()  # now we have the sorted list

# set up the dictionary of {word: feature_index}
wordIndexDict = defaultdict(lambda: -1)
for i in range(dictSize):
    k, _ = wordCount[i]
    wordIndexDict[k] = i

# build the training and validation sets
numFeatures = dictSize
y = [r['genreID'] for r in reviews]
X = lil_matrix((len(reviews), numFeatures))
for i in range(len(reviews)):
    features = featurize(reviews[i], wordIndexDict, dictSize)
    for j in range(numFeatures):
        if features[j] != 0:
            X[i,j] = features[j]
#X = preprocessing.scale(X, with_mean=False)
X_train = X[:trainSplit]
y_train = y[:trainSplit]
X_valid = X[trainSplit:]
y_valid = y[trainSplit:]

# train the model
mod = linear_model.LogisticRegression(max_iter=15000, n_jobs=-1, C=2.0)
mod.fit(X_train, y_train)

# get predictions and find the accuracy
preds = mod.predict(X_valid)
valid_acc = calc_acc(preds, y_valid)
print("Validation accuracy =", valid_acc)

# load the test set and build a table of {reviewid: review}
reviews = load_data('test_Category.json.gz')
idToReview = {}
for review in reviews:
    idToReview[review['reviewID']] = review

# generate and write predictions for the test set
predictions = open("predictions_Category.csv", 'w')
for l in open("pairs_Category.txt"):
    # header
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u, r = l.strip().split('-')
    review = idToReview[r]
    features = featurize(review, wordIndexDict, dictSize)
    p = mod.predict([features])[0]
    predictions.write(u + '-' + r + "," + str(p) + "\n")
predictions.close()

print("predictions successfully generated in predictions_Category.csv")
