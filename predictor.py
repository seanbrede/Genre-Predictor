import gzip                 as gzip
import sklearn.linear_model as line
import utilities            as util


# parameters
grams        = 20000, 2000, 200
reg_constant = 0.5
iterations   = 500
train_split  = 165000
algorithm    = "newton-cg"  # liblinear, lbfgs, newton-cg, sag, saga
train_file   = "train.json.gz"
test_file    = "test.json.gz"


# 1. load the training data
print("1. Loading data")
raw_reviews = [eval(review) for review in gzip.open(train_file, "r+")]
print("   Data loaded")

# 2. build a list of all words in the training set and their counts, sorted by order
print("2. Counting grams")
gram_counts_list = util.countTotalGrams(raw_reviews[:train_split], len(grams))
print("   Grams counted")

# 3. set up the list of dictionaries of {gram: feature_index}
print("3. Building dictionary")
gram_indices = util.buildIndices(gram_counts_list, grams)
print("   Dictionary built")

# 4. build the training and validation sets
print("4. Building sets")
X, y = util.buildSets(raw_reviews, grams, gram_indices)
X_train = X[:train_split]
y_train = y[:train_split]
X_valid = X[train_split:]
y_valid = y[train_split:]
print("   Sets built")

# 5. train the model
print("5. Training model")
model = line.LogisticRegression(max_iter=iterations, C=reg_constant, solver=algorithm, n_jobs=-1)
model.fit(X_train, y_train)
print("   Model trained")

# 6. get predictions and find the accuracy
print("6. Computing accuracy")
preds = model.predict(X_valid)
valid_acc = util.calcAccuracy(preds, y_valid)
print("   Accuracy computed ->", valid_acc)
