import warnings              as warn
import collections           as coll
import gzip                  as gzip
import sklearn.linear_model  as line
import scipy.sparse          as spar
import utilities             as util



# parameters
train_split  = 165000
num_unigrams = 27000
iterations   = 500
reg_constant = 2.0
train_file   = "train.json.gz"
test_file    = "test.json.gz"


# 1. load the training data
print("1. Loading data")
reviews_raw = [eval(line) for line in gzip.open(train_file, "r+")]
print("   Data loaded")

# 2. build a list of all words in the training set and their counts, sorted by order
print("2. Counting words")
word_to_count = util.countWords(reviews_raw, train_split)
print("   Words counted")

# 3. set up the dictionary of {word: feature_index}
print("3. Establishing dictionary")
word_to_index = coll.defaultdict(lambda: -1)
for i in range(num_unigrams):
    k, _ = word_to_count[i]
    word_to_index[k] = i
print("   Dictionary established")

# 4. build the training and validation sets
print("4. Building sets")
y = [r['genreID'] for r in reviews_raw]
X = spar.lil_matrix((len(reviews_raw), num_unigrams))
for i in range(len(reviews_raw)):
    features = util.featurize(reviews_raw[i], word_to_index, num_unigrams)
    for j in range(num_unigrams):
        if features[j] != 0:
            X[i, j] = features[j]
# X = prep.minmax_scale(X, feature_range=(0, 1))
# X = prep.scale(X, with_mean=False)
# min_max_scaler = skle.preprocessing.MinMaxScaler()
X_train = X[:train_split]
y_train = y[:train_split]
X_valid = X[train_split:]
y_valid = y[train_split:]
print("   Sets built")

# 5. train the model
print("5. Training model")
model = line.LogisticRegression(max_iter=iterations, n_jobs=-1, C=reg_constant)
# warn.filterwarnings("ignore")  # otherwise fit() will throw ConvergenceWarning
with warn.catch_warnings():  # otherwise fit() will throw ConvergenceWarning
    warn.simplefilter("ignore")
    model.fit(X_train, y_train)
print("   Model trained")

# 6. get predictions and find the accuracy
print("6. Computing accuracy")
preds = model.predict(X_valid)
valid_acc = util.calcAcc(preds, y_valid)
print("   Accuracy computed ->", valid_acc)
