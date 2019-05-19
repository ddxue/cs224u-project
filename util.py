import pandas as pd
import numpy as np
import os
from sentence2vec import Sentence2Vec
from sklearn.linear_model import LogisticRegression

DATA_HOME = 'Fake_News_Detection/liar_dataset'
MODEL_FILENAME = './job_titles.model'
TEST_FILENAME = os.path.join(DATA_HOME, 'test.tsv')
TRAIN_FILENAME = os.path.join(DATA_HOME, 'train.tsv')
VALID_FILENAME = os.path.join(DATA_HOME, 'valid.tsv')

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

def replace_nans(x, replacement=0):
	nans = np.isnan(x)
	x[nans] = replacement

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


def get_np_filenames(dataset):
	np_X_file_name = os.path.join(DATA_HOME, dataset + '_X.npy')
	np_y_file_name = os.path.join(DATA_HOME, dataset + '_y.npy')
	return np_X_file_name, np_y_file_name

def get_np_data(dataset):
	np_X_file_name, np_y_file_name = get_np_filenames(dataset)
	X, y = None, None
	if os.path.isfile(np_X_file_name):
		X = np.load(np_X_file_name)
	if os.path.isfile(np_y_file_name):
		y = np.load(np_y_file_name)
	return X, y

# Load data
def load_data_from_file(file_name, model):
	print("Loading Raw Data...")
	raw_X, y = load_raw_data(file_name)
	print("Converting Sentences To Vectors...")
	X = raw_to_numeric_features(raw_X, model)
	print("Converting Labels...")
	y = convert_labels(y, MODE)
	print("Done loading data...")
	return X, y

def load_all_data(model, cache=True, normalize=True):
	train_X, train_y = get_np_data('train')
	mean_X, _ = get_np_data('mean')
	std_X, _ = get_np_data('std')
	if train_X is None or train_y is None or mean_X is None or std_X is None:
		train_X, train_y = load_data_from_file(TRAIN_FILENAME, model)
		train_X = np.array(train_X, dtype=np.float32)
		replace_nans(train_X)
		mean_X = np.mean(train_X, axis=0)
		std_X = np.std(train_X, axis=0)
		if cache:
			np_train_X_file_name, np_train_y_file_name = get_np_filenames('train')
			np_mean_X_file_name, _ = get_np_filenames('mean')
			np_std_X_file_name, _ = get_np_filenames('std')
			np.save(np_train_X_file_name, train_X)
			np.save(np_train_y_file_name, train_y)
			np.save(np_mean_X_file_name, mean_X)
			np.save(np_std_X_file_name, std_X)
	if normalize:
		train_X = (train_X - mean_X) / std_X
	val_X, val_y = get_np_data('val')
	if val_X is None or val_y is None:
		val_X, val_y = load_data_from_file(VALID_FILENAME, model)
		val_X = np.array(val_X, dtype=np.float32)
		replace_nans(val_X)
		if cache:
			np_val_X_file_name, np_val_y_file_name = get_np_filenames('val')
			np.save(np_val_X_file_name, val_X)
			np.save(np_val_y_file_name, val_y)
	if normalize:
		val_X = (val_X - mean_X) / std_X
	test_X, test_y = get_np_data('test')
	if test_X is None or test_y is None:
		test_X, test_y = load_data_from_file(TEST_FILENAME, model)
		test_X = np.array(test_X, dtype=np.float32)
		replace_nans(test_X)
		if cache:
			np_test_X_file_name, np_test_y_file_name = get_np_filenames('test')
			np.save(np_test_X_file_name, test_X)
			np.save(np_test_y_file_name, test_y)
	if normalize:
		test_X = (test_X - mean_X) / std_X
	return train_X, train_y, val_X, val_y, test_X, test_y

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
	train_X, train_y, val_X, val_y, test_X, test_y = load_all_data(model)
	print(train_X.shape)
	print(train_y.shape)
	print("Fitting Model...")

	log_reg = LogisticRegression().fit(train_X, train_y)
	val_yhat = log_reg.predict(val_X)
	print("Evaluating..")
	print(evaluate(val_y, val_yhat))
