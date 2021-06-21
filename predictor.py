import collections           as coll
import gzip                  as gzip
import sklearn.linear_model  as line
import scipy.sparse          as spar
import utilities             as util


# parameters
num_unigrams = 27000
reg_constant = 0.2
iterations   = 400
train_split  = 165000
algorithm    = "newton-cg"  # liblinear, lbfgs, newton-cg, sag, saga
train_file   = "train.json.gz"
test_file    = "test.json.gz"


# 1. load the training data
print("1. Loading data")
reviews_raw = [eval(review) for review in gzip.open(train_file, "r+")]
print("   Data loaded")

# 2. build a list of all words in the training set and their counts, sorted by order
print("2. Counting words")
word_counts = util.countWords(reviews_raw, train_split)
print("   Words counted")

# 3. set up the dictionary of {word: feature_index}
print("3. Establishing dictionary")
word_indices = coll.defaultdict(lambda: -1)
for i in range(num_unigrams):
    word, _ = word_counts[i]
    word_indices[word] = i
print("   Dictionary established")

# 4. build the training and validation sets
print("4. Building sets")
y = [r['genreID'] for r in reviews_raw]
X = spar.lil_matrix((len(reviews_raw), num_unigrams))
for i in range(len(reviews_raw)):
    features = util.featurize(reviews_raw[i], word_indices, num_unigrams)
    for j in range(num_unigrams):
        if features[j] != 0:
            X[i, j] = features[j]
X_train = X[:train_split]
y_train = y[:train_split]
X_valid = X[train_split:]
y_valid = y[train_split:]
print("   Sets built")

# 5. train the model
print("5. Training model")
model = line.LogisticRegression(max_iter=iterations, n_jobs=-1, C=reg_constant, solver=algorithm)
model.fit(X_train, y_train)
print("   Model trained")

# 6. get predictions and find the accuracy
print("6. Computing accuracy")
preds = model.predict(X_valid)
valid_acc = util.calcAcc(preds, y_valid)
print("   Accuracy computed ->", valid_acc)
