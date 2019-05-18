import pandas as pd
import numpy as np
import os

DATA_HOME = 'Fake_News_Detection/liar_dataset'
TEST_FILENAME = os.path.join(DATA_HOME, 'test.tsv')
TRAIN_FILENAME = os.path.join(DATA_HOME, 'train.tsv')
VALID_FILENAME = os.path.join(DATA_HOME, 'valid.tsv')

BINARY = '2-Way'
HEXARY = '6-Way'
REGRESS = 'Regress'
MODES = [BINARY, HEXARY, REGRESS]
MODE = MODES[2]

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
def raw_to_numeric(x):
	assert len(x) == 12
	# TODO: do something useful
	return np.array([1,2,3])

# Convert entire raw dataset into numeric one
def raw_to_numeric_features(raw_X):
	assert raw_X.shape[1] == 12
	X = None
	for i in range(raw_X.shape[0]):
		if X is None:
			X = raw_to_numeric(raw_X[i])
		else:
			X = np.vstack((X, raw_to_numeric(raw_X[i])))
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
def load_data(file_name=TEST_FILENAME):
	raw_X, y = load_raw_data(file_name)
	X = raw_to_numeric_features(raw_X)
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
	load_data()