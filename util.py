import pandas as pd
import numpy as np
import os
from sentence2vec import Sentence2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from six import string_types

###################
# DATA PATHS
###################

DATA_HOME = 'Fake_News_Detection/liar_dataset'
MODEL_FILENAME = './job_titles.model'
TEST_FILENAME = os.path.join(DATA_HOME, 'test.tsv')
TRAIN_FILENAME = os.path.join(DATA_HOME, 'train.tsv')
VALID_FILENAME = os.path.join(DATA_HOME, 'valid.tsv')

###################
# TASKS
###################

BINARY = '2-Way'
HEXARY = '6-Way'
REGRESS = 'Regress'
MODES = [BINARY, HEXARY, REGRESS]
MODE = MODES[1]

LABEL_TO_INT = {
    'pants-fire': 0,
    'false': 1,
    'barely-true': 2,
    'half-true': 3,
    'mostly-true': 4,
    'true': 5
}

COLUMNS_NAMES = ["id", 
                "label", "statement", "subject", "speaker", "job", "state", "party",  # columns 1-8
                "barely-true-counts", "false-counts", "half-true-counts", "mostly-true-counts", "pants-fire-counts",  # columns 9-13 (counts)
                "venue"]  # names of the columns in the tsv files

###############################
# CATEGORICAL FEATURE CONSTANTS
###############################

SPEAKERS_LIST = ["barack-obama", "donald-trump", "hillary-clinton", "mitt-romney", 
            "scott-walker", "john-mccain", "rick-perry", "chain-email", 
            "marco-rubio", "rick-scott", "ted-cruz", "bernie-s", "chris-christie", 
            "facebook-posts", "charlie-crist", "newt-gingrich", "jeb-bush", 
            "joe-biden", "blog-posting","paul-ryan"]
JOB_LIST = ["president", "u.s. senator", "governor", "president-elect", "presidential candidate", 
            "u.s. representative", "state senator", "attorney", "state representative", "congress"]
JOB_DICT = {"president":0, "u.s. senator":1, "governor":2, "president-elect":3, "presidential candidate":4, 
            "u.s. representative":5, "state senator":6, "attorney":7, "state representative":8, "congress":9}
PARTY_DICT = {"republican":0,"democrat":1,"none":2,"organization":3,"newsmaker":4}    

# Possible groupings (50 groups + 1 for rest).
STATES_DICT = {"wyoming": 48, "colorado": 5, "washington": 45, "hawaii": 10, "tennessee": 40, 
               "wisconsin": 47, "nevada": 26, "north dakota": 32, "mississippi": 22, "south dakota": 39, 
               "new jersey": 28, "oklahoma": 34, "delaware": 7, "minnesota": 21, "north carolina": 31, 
               "illinois": 12, "new york": 30, "arkansas": 3, "west virginia": 46, "indiana": 13, 
               "louisiana": 17, "idaho": 11, "south  carolina": 38, "arizona": 2, "iowa": 14, "maine":49, "maryland": 18,
               "michigan": 20, "kansas": 15, "utah": 42, "virginia": 44, "oregon": 35, "connecticut": 6, 
               "montana": 24, "california": 4, "massachusetts": 19, "rhode island": 37, "vermont": 43, 
               "georgia": 9, "pennsylvania": 36, "florida": 8, "alaska": 1, "kentucky": 16, "nebraska": 25, 
               "new hampshire": 27, "texas": 41, "missouri": 23, "ohio": 33, "alabama": 0, "new mexico": 29}

# Possible groups (14).
SUBJECT_LIST = ["health","tax","immigration","election","education",
                "candidates-biography","economy","gun","jobs","federal-budget","energy","abortion","foreign-policy"]
SUBJECT_DICT = {"health":0,"tax":1,"immigration":2,"election":3,"education":4,
                "candidates-biography":5,"economy":6,"gun":7,"jobs":8,"federal-budget":9,"energy":10,"abortion":11,"foreign-policy":12}

VENUE_LIST = ["news release","interview","tv","radio",
              "campaign","news conference","press conference","press release",
              "tweet","facebook","email"]
VENUE_DICT = {"news release":0,"interview":1,"tv":2,"radio":3,
              "campaign":4,"news conference":5,"press conference":6,"press release":7,
              "tweet":8,"facebook":9,"email":10}

###########################
# MAP FEATURES, BINARIZING
###########################

def map_speaker(speaker):
    speaker_dict = {}
    for cnt, speaker in enumerate(SPEAKERS_LIST):
        speaker_dict[speaker] = cnt

    if isinstance(speaker, string_types):
        speaker = speaker.lower()
        matches = [s for s in SPEAKERS_LIST if s in speaker]
        if len(matches) > 0:
            return speaker_dict[matches[0]] #Return index of first match
        else:
            return len(SPEAKERS_LIST)
    else:
        return len(SPEAKERS_LIST) #Nans or un-string data goes here.
 
def map_job(job):
    """ Possible groupings could be (11 groups):
    # president, us senator, governor(contains governor), president-elect, presidential candidate, us representative,
    # state senator, attorney, state representative, congress (contains congressman or congresswoman), rest
    """
    if isinstance(job, string_types):
        job = job.lower()
        matches = [s for s in JOB_LIST if s in job]
        if len(matches) > 0:
            return JOB_DICT[matches[0]] #Return index of first match
        else:
            return 10 #This maps any other job to index 10
    else:
        return 10 #Nans or un-string data goes here.

def map_party(party):
    if party in PARTY_DICT:
        return PARTY_DICT[party]
    else:
        return 5 # default index for rest party is 5

def map_state(state):
    if isinstance(state, string_types):
        state = state.lower()  # convert to lowercase
        if state in STATES_DICT:
            return STATES_DICT[state]
        else:
            return 50 # This maps any other location to index 50
    else:
        return 50 # Nans or un-string data goes here.

#possibe groups (12)
#news release, interview, tv (television), radio, campaign, news conference, press conference, press release,
#tweet, facebook, email, rest
def map_venue(venue):
    if isinstance(venue, string_types):
        venue = venue.lower()
        matches = [s for s in VENUE_LIST if s in venue]
        if len(matches) > 0:
            return VENUE_DICT[matches[0]] #Return index of first match
        else:
            return 11 #This maps any other venue to index 11
    else:
        return 11 # Nans or un-string data goes here.

#health-care,taxes,immigration,elections,education,candidates-biography,guns,
#economy&jobs ,federal-budget,energy,abortion,foreign-policy,state-budget, rest
#Economy & Jobs is bundled together, because it occurs together
def map_subject(subject):
    if isinstance(subject, string_types):
        subject = subject.lower()
        matches = [s for s in SUBJECT_LIST if s in subject]
        if len(matches) > 0:
            return SUBJECT_DICT[matches[0]] #Return index of first match
        else:
            return 13 # This maps any other subject to index 13
    else:
        return 13 # Nans or un-string data goes here.

def make_one_hot(index, maximum):
    one_hot = np.zeros(maximum)
    one_hot[index] = 1
    return one_hot

############################
# Reputation Score(s)
# TODO: play with this!
############################
def reputation_vec(statement_counts):
    assert len(statement_counts) == 5
    x = np.array(statement_counts)
    false_counts = np.array([x[0] + x[1] + x[4]])
    true_counts = np.array([x[2] + x[3]])

    if np.sum(x) == 0:
        return np.ones(5) / 5
    return x / np.sum(x)

##################################
# FOR STANDARDIZING ACROSS MODELS,
# USE THIS FUNCTION!!
##################################

# Take a single data point and return
# [original statement, venue, np.array(other contextual features)]
def standardized_features(x):
    assert len(x) == 12
    features = []
    # Add original statement
    features.append(x[0])

    # Add venue
    features.append(x[-1])

    subject = make_one_hot(map_subject(x[1]), 14)
    speaker = make_one_hot(map_speaker(x[2]), 20)
    job = make_one_hot(map_job(x[3]), 11)
    state = make_one_hot(map_state(x[4]), 51)
    party = make_one_hot(map_party(x[5]), 6)
    contextual_features = np.concatenate((
        subject, speaker, job, state, party))
    assert np.sum(contextual_features) == 5

    # Counts of the statements
    statement_counts = x[6:11]
    contextual_features = np.concatenate((contextual_features, 
        reputation_vec(statement_counts)))
    features.append(contextual_features)

    return features

####################################

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
def raw_to_numeric_std(x, model):
    features = standardized_features(x)
    statement_vec = model.get_vector(features[0])
    venue_vec = model.get_vector(features[1])
    return np.concatenate((statement_vec, venue_vec, features[2]))

# Convert entire raw dataset into numeric one
def raw_to_numeric_features(raw_X, model):
    assert raw_X.shape[1] == 12
    X = None
    n = raw_X.shape[0]
    for i in range(n):
        if i % 100 == 0:
            print("On Data Point {} of {}".format(i, raw_X.shape[0]))
        if X is None:
            X = raw_to_numeric_std(raw_X[i], model)
        else:
            if i == 1:
                temp = np.zeros((n, X.shape[0]))
                temp[0] = X[0]
                X = temp
            else:
                X[i] = raw_to_numeric_std(raw_X[i], model)
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

def load_all_data(model, load_from_cache=False, cache=True, normalize=True):
    train_X, train_y = get_np_data('train')
    mean_X, _ = get_np_data('mean')
    std_X, _ = get_np_data('std')
    if not load_from_cache or train_X is None or train_y is None or mean_X is None or std_X is None:
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
    if not load_from_cache or val_X is None or val_y is None:
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
    if not load_from_cache or test_X is None or test_y is None:
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
        print(confusion_matrix(y, yhat))
        print("F1 Score: {}".format(f1_score(y, yhat, average='macro')))
        print("Accuracy: {}".format(accuracy_score(y, yhat)))
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
    train_yhat = log_reg.predict(train_X)
    print("Evaluating..")

    print("Train Results")
    evaluate(train_y, train_yhat)
    print("Val Results")
    evaluate(val_y, val_yhat)
