import pandas as pd
import numpy as np
import os
from sentence2vec import Sentence2Vec
from sklearn.linear_model import LogisticRegression

DATA_HOME = 'Fake_News_Detection/liar_dataset'
TEST_FILENAME = os.path.join(DATA_HOME, 'test.tsv')
TRAIN_FILENAME = os.path.join(DATA_HOME, 'train.tsv')
VALID_FILENAME = os.path.join(DATA_HOME, 'valid.tsv')
MODEL_FILENAME = './job_titles.model'

BINARY = '2-Way'
HEXARY = '6-Way'
REGRESS = 'Regress'
MODES = [BINARY, HEXARY, REGRESS]
MODE = MODES[0]

LABEL_TO_INT = {
	'pants-fire': 0,
	'false': 1,
	'barely-true': 2,
	'half-true': 3,
	'mostly-true': 4,
	'true': 5
}

# Loads (X, y) from filename
def load_raw_data(file_name=TEST_FILENAME):
	data = pd.read_csv(file_name, sep='\t', header=None).to_numpy()
	X = data[:,2:]
	y = data[:,1].reshape(-1)
	return X, y

# Convert a single row of data into a vector
def raw_to_numeric(x, model):
	assert len(x) == 12
	sentence_vec = model.get_vector(x[0])
	statement_counts = x[6:11]
	false_counts = np.array([x[6] + x[7] + x[10]])
	true_counts = np.array([x[8] + x[9]])
	z = np.concatenate((sentence_vec, 
		statement_counts, 
		false_counts, 
		true_counts))
	return z

# Convert entire raw dataset into numeric one
def raw_to_numeric_features(raw_X, model):
	assert raw_X.shape[1] == 12
	X = None
	for i in range(raw_X.shape[0]):
		if i % 100 == 0:
			print("On Data Point {} of {}".format(i, raw_X.shape[0]))
		if X is None:
			X = raw_to_numeric(raw_X[i], model)
		else:
			X = np.vstack((X, raw_to_numeric(raw_X[i], model)))
	return X

# Convert labels to numeric.
# BINARY: 0 for false, 1 for true (3 each)
# HEXARY: 0 for pants-fire, ..., 5 for true
# REGRESS: 0.0 for pants-fire, ..., 1.0 for true
def convert_labels(y, mode=MODE):
	assert mode in MODES
	if mode == BINARY:
		return np.array([LABEL_TO_INT[z] // 3 for z in y])
	elif mode == HEXARY:
		return np.array([LABEL_TO_INT[z] for z in y])
	elif mode == REGRESS:
		return np.array([LABEL_TO_INT[z] / 5.0 for z in y])

# Return a random batch.
def get_batch(X, y, batch_size=1, replace=True):
	assert X.shape[0] == len(y)
	n = len(y)
	indices = np.random.choice(n, batch_size, replace=replace)
	X_batch = X[indices]
	y_batch = y[indices]
	return X_batch, y_batch

# Load data
def load_data(model, file_name=TEST_FILENAME):
	print("Loading Raw Data...")
	raw_X, y = load_raw_data(file_name)
	print("Converting Sentences To Vectors...")
	X = raw_to_numeric_features(raw_X, model)
	print("Converting Labels...")
	y = convert_labels(y, MODE)
	return X, y

# TODO: what metric?
# Return MSE for regression, accuracy for classification
def evaluate(y, yhat, mode=MODE):
	assert mode in MODES
	assert len(y) == len(yhat)
	if mode == BINARY or mode == HEXARY:
		return np.mean([int(y[i] == yhat[i]) for i in range(len(y))])
	elif mode == REGRESS:
		# Return MSE
		return np.mean(np.square(y - yhat))

if __name__ == '__main__':
	model = Sentence2Vec(MODEL_FILENAME)
	train_X, train_y = load_data(model, file_name=VALID_FILENAME)
	print(train_X.shape)
	print(train_y.shape)
	print("Fitting Model...")
	log_reg = LogisticRegression().fit(train_X, train_y)
	val_X, val_y = load_data(model, file_name=TEST_FILENAME)
	val_yhat = log_reg.predict(val_X)
	print(evaluate(val_y, val_yhat))


