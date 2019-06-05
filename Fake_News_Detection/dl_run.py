import time
import sys
import os.path
import csv

import pickle

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

import keras.utils
from keras.utils.vis_utils import plot_model
from keras.callbacks import TensorBoard, CSVLogger
from keras import backend as K
from keras.models import Sequential
from keras.models import load_model
from keras.models import Model
from keras.layers import Dense,Flatten,LSTM,Conv1D,GlobalMaxPool1D,Dropout
from keras.layers import Input
from keras.layers.embeddings import Embedding
from keras import optimizers

#To access particular word_index. Just load these.
#To read a word in a sentence use keras tokenizer again, coz easy
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

from metadata_features import *

###########################
#  Constants            
###########################

# Glove file locations.
GLOVE_PATH = "glove.6B/glove.6B.100d.txt"  # where the glove embeddings live
EMBEDDING_DIM = 100

NUM_CLASSES = 2

# Metadata related values.
num_party = 6
num_state = 51
num_venue = 12
num_job = 11
num_sub = 14
num_speaker = 21

###########################
# Hyperparameters
###########################

lstm_size = 100
num_steps = 25
num_epochs = 30
batch_size = 40

###########################
# Metrics             
###########################

def f1(y_true, y_pred):
    
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

###########################
# Read Data             
###########################

def read_data(train_file, valid_file, test_file):
    """ Read in the training, validation, and test data from .tsv files.

    Parameters
    ----------
    train_file: string
        the path to the training file
    valid_file: string
            the path to the validation file
    test_file: string
            the path to the testing file
    """
    data_set = pd.read_table(train_file, delimiter="\t", quoting=csv.QUOTE_NONE, names=COLUMNS_NAMES)
    val_set = pd.read_table(valid_file, delimiter="\t", quoting=csv.QUOTE_NONE, names=COLUMNS_NAMES)
    test_set = pd.read_table(test_file, delimiter="\t", quoting=csv.QUOTE_NONE, names=COLUMNS_NAMES)

    return data_set, val_set, test_set

def read_pretrained_embeddings():
    # Use pretrained word embeddings (experiment).
    # Read GloVe vectors and get unique words in an array.
    embeddings_index = {}
    with open(GLOVE_PATH) as fp:
        for line in fp:
            values = line.split()
            vectors = np.asarray(values[1:], dtype="float32")
            embeddings_index[values[0].lower()] = vectors
    
    print("File reading is done.")
    print("Found %s word vectors." % len(embeddings_index))

    return embeddings_index

###########################
# Create Vocab / Embedding      
###########################

def get_vocab_dict(data_set, embeddings_index):
    # Tokenize statement and form/load vocabulary.
    vocab_dict = {}
    if not os.path.exists("vocab.p"):
        t = Tokenizer()
        t.fit_on_texts(data_set["statement"])
        vocab_dict = t.word_index
        pickle.dump( t.word_index, open( "vocab.p", "wb" ))
        print("Vocab dict is created")
        print("Saved vocab dict to pickle file")
    else:
        print("Loading vocab dict from pickle file")
        vocab_dict = pickle.load(open("vocab.p", "rb" ))

    # Create embedding matrix to feed in embeddings directly.
    num_words = len(vocab_dict) + 1
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word, i in vocab_dict.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    return vocab_dict, embedding_matrix

###########################
# Map Features (Text to Integers)
###########################

def map_statement_features(data_set, val_set, test_set, vocab_dict):
    def preprocess_statement(statement):
        text = text_to_word_sequence(statement) # built-in keras function!
        val = [vocab_dict[t] for t in text if t in vocab_dict] # Replace UNK words with 0 index
        return val

    # Load data and pad sequences to prepare training, validation and test data
    data_set["word_ids"] = data_set["statement"].apply(preprocess_statement)
    val_set["word_ids"] = val_set["statement"].apply(preprocess_statement)
    test_set["word_ids"] = test_set["statement"].apply(preprocess_statement)

    return data_set, val_set, test_set

def map_category_features(data_set, val_set, test_set):
    # Map speakers to speak_ids.
    data_set["speaker_id"] = data_set["speaker"].apply(map_speaker)
    val_set["speaker_id"] = val_set["speaker"].apply(map_speaker)
    test_set["speaker_id"] = test_set["speaker"].apply(map_speaker) #Speaker

    # Map jobs to job_ids.
    data_set["job_id"] = data_set["job"].apply(map_job)
    val_set["job_id"] = val_set["job"].apply(map_job)
    test_set["job_id"] = test_set["job"].apply(map_job) #Job

    # Map parties (hyperparameter -> num_party).
    data_set["party_id"] = data_set["party"].apply(map_party)
    val_set["party_id"] = val_set["party"].apply(map_party)
    test_set["party_id"] = test_set["party"].apply(map_party) #Party

    # Map states to state_ids.
    data_set["state_id"] = data_set["state"].apply(map_state)
    val_set["state_id"] = val_set["state"].apply(map_state)
    test_set["state_id"] = test_set["state"].apply(map_state) #State

    # Map subjects to subject_ids.
    data_set["subject_id"] = data_set["subject"].apply(map_subject)
    val_set["subject_id"] = val_set["subject"].apply(map_subject)
    test_set["subject_id"] = test_set["subject"].apply(map_subject) #Subject

    # Map venues to venue_ids.
    data_set["venue_id"] = data_set["venue"].apply(map_venue)
    val_set["venue_id"] = val_set["venue"].apply(map_venue)
    test_set["venue_id"] = test_set["venue"].apply(map_venue) #Venue

    # Create labels to label_ids.
    data_set["label_id"] = data_set["label"].apply(map_label)
    val_set["label_id"] = val_set["label"].apply(map_label)
    test_set["label_id"] = test_set["label"].apply(map_label) #Label

    return data_set, val_set, test_set

def create_numeric_features(data_set, val_set, test_set, vocab_dict):
    data_set, val_set, test_set = map_statement_features(data_set, val_set, test_set, vocab_dict)
    data_set, val_set, test_set = map_category_features(data_set, val_set, test_set)

    return data_set, val_set, test_set

###########################
# Prepare Features (Create 1-Hot Vectors)
###########################

def prepare_statement_feature(data_set, val_set, test_set):
    # Convert word_ids to X feature data.
    X_train = data_set["word_ids"]
    X_val = val_set["word_ids"]
    X_test = test_set["word_ids"]

    # Pad the statement sequence data.
    X_train = sequence.pad_sequences(X_train, maxlen=num_steps, padding="post",truncating="post")
    X_val = sequence.pad_sequences(X_val, maxlen=num_steps, padding="post",truncating="post")
    X_test = sequence.pad_sequences(X_test, maxlen=num_steps, padding="post",truncating="post")

    return X_train, X_val, X_test

def prepare_categorical_feature(data_set, val_set, test_set):
    a = keras.utils.to_categorical(data_set["party_id"], num_classes=num_party)
    b = keras.utils.to_categorical(data_set["state_id"], num_classes=num_state)
    c = keras.utils.to_categorical(data_set["venue_id"], num_classes=num_venue)
    d = keras.utils.to_categorical(data_set["job_id"], num_classes=num_job)
    e = keras.utils.to_categorical(data_set["subject_id"], num_classes=num_sub)
    f = keras.utils.to_categorical(data_set["speaker_id"], num_classes=num_speaker)

    a_val = keras.utils.to_categorical(val_set["party_id"], num_classes=num_party)
    b_val = keras.utils.to_categorical(val_set["state_id"], num_classes=num_state)
    c_val = keras.utils.to_categorical(val_set["venue_id"], num_classes=num_venue)
    d_val = keras.utils.to_categorical(val_set["job_id"], num_classes=num_job)
    e_val = keras.utils.to_categorical(val_set["subject_id"], num_classes=num_sub)
    f_val = keras.utils.to_categorical(val_set["speaker_id"], num_classes=num_speaker)

    a_test = keras.utils.to_categorical(test_set["party_id"], num_classes=num_party)
    b_test = keras.utils.to_categorical(test_set["state_id"], num_classes=num_state)
    c_test = keras.utils.to_categorical(test_set["venue_id"], num_classes=num_venue)
    d_test = keras.utils.to_categorical(test_set["job_id"], num_classes=num_job)
    e_test = keras.utils.to_categorical(test_set["subject_id"], num_classes=num_sub)
    f_test = keras.utils.to_categorical(test_set["speaker_id"], num_classes=num_speaker)

    # Concatenate all the X metadata feature data.
    X_train_meta = np.hstack((a,b,c,d,e,f))
    X_val_meta = np.hstack((a_val,b_val,c_val,d_val,e_val,f_val))
    X_test_meta = np.hstack((a_test,b_test,c_test,d_test,e_test,f_test))

    return X_train_meta, X_val_meta, X_test_meta

def prepare_count_feature(data_set, val_set, test_set):
    a = data_set["pants-fire-counts"]
    b = data_set["false-counts"]
    c = data_set["barely-true-counts"]
    d = data_set["half-true-counts"]
    e = data_set["mostly-true-counts"]

    a_val = val_set["pants-fire-counts"]
    b_val = val_set["false-counts"]
    c_val = val_set["barely-true-counts"]
    d_val = val_set["half-true-counts"]
    e_val = val_set["mostly-true-counts"]

    a_test = test_set["pants-fire-counts"]
    b_test = test_set["false-counts"]
    c_test = test_set["barely-true-counts"]
    d_test = test_set["half-true-counts"]
    e_test = test_set["mostly-true-counts"]

    X_train_counts = np.stack((a,b,c,d,e), axis=-1)
    X_val_counts = np.stack((a_val,b_val,c_val,d_val,e_val), axis=-1)
    X_test_counts = np.stack((a_test,b_test,c_test,d_test,e_test), axis=-1)

    return X_train_counts, X_val_counts, X_test_counts

def prepare_labels(data_set, val_set, test_set):
    # Convert label_id to Y labels.
    Y_train = data_set["label_id"]
    Y_val = val_set["label_id"]
    Y_test = test_set["label_id"]

    # Convert to categorical variables.
    Y_train = keras.utils.to_categorical(Y_train, num_classes=NUM_CLASSES)
    Y_val = keras.utils.to_categorical(Y_val, num_classes=NUM_CLASSES)
    Y_test = keras.utils.to_categorical(Y_test, num_classes=NUM_CLASSES)

    return Y_train, Y_val, Y_test

###########################
# MAIN
###########################

def run(train_file, valid_file, test_file, output_file):
    """ The function to run your ML algorithm on given datasets, 
    generate the output and save them into the provided file path.
    
    Parameters
    ----------
    train_file: string
        the path to the training file
    valid_file: string
            the path to the validation file
    test_file: string
            the path to the testing file
    output_file: string
        the path to the output predictions to be saved
    """

    print("="*20 + " Data Loading " + "="*20)
    print()

    # Read data and word embeddings.
    data_set, val_set, test_set = read_data(train_file, valid_file, test_file)

    # Tokenize statement and form/load vocabulary.
    embeddings_index = read_pretrained_embeddings()
    vocab_dict, embedding_matrix = get_vocab_dict(data_set, embeddings_index)

    # Create numeric features from the data.
    data_set, val_set, test_set = create_numeric_features(data_set, val_set, test_set, vocab_dict)

    print("Data loaded.")

    ###########################
    # Create Dataset & Labels
    ###########################

    print("="*20 + " Data Pre-Processing " + "="*20)
    print()

    # Get the features.
    X_train, X_val, X_test = prepare_statement_feature(data_set, val_set, test_set)
    X_train_meta, X_val_meta, X_test_meta = prepare_categorical_feature(data_set, val_set, test_set)
    X_train_counts, X_val_counts, X_test_counts = prepare_count_feature(data_set, val_set, test_set)

    print("Training Data Shape:", X_train.shape)
    print("Training Metadata Shape:", X_train_meta.shape)
    print("Training Counts Shape:", X_train_counts.shape)

    # Get the labels.
    Y_train, Y_val, Y_test = prepare_labels(data_set, val_set, test_set)   

    print("Data processed.")

    ###########################
    # Model Definitions
    ###########################
    
    vocab_length = len(vocab_dict.keys())
    hidden_size = EMBEDDING_DIM # has to be same as EMBEDDING_DIM

    ###########################
    # LSTM Model            
    ###########################

    statement_input = Input(shape=(num_steps,), dtype="int32", name="statement_input")

    x = Embedding(vocab_length+1,EMBEDDING_DIM,weights=[embedding_matrix],input_length=num_steps,trainable=False)(statement_input) #Preloaded glove embeddings
    # x = Embedding(output_dim=hidden_size, input_dim=vocab_length+1, input_length=num_steps)(statement_input) #Train embeddings from scratch
    lstm_in = LSTM(lstm_size,dropout=0.25)(x)

    metadata_input = Input(shape=(X_train_meta.shape[1],), name="metadata_input")
    counts_input = Input(shape=(X_train_counts.shape[1],), name="counts_input")

    x_meta = Dense(64, activation="relu")(metadata_input)
    x_counts = Dense(64, activation="relu")(counts_input)

    x_fc = keras.layers.concatenate([lstm_in, metadata_input, counts_input])
    main_output = Dense(NUM_CLASSES, activation="softmax", name="main_output")(x_fc)
    
    model = Model(inputs=[statement_input, metadata_input, counts_input], outputs=[main_output])

    ###########################
    # CNN Model             
    ###########################

    # # Hyperparameters for CNN.
    # kernel_sizes = [2,5,8]
    # filter_size = 128

    # # Statement input.    
    # statement_input = Input(shape=(num_steps,), dtype="int32", name="statement_input")
    
    # # Metadata input.
    # #x = Embedding(vocab_length+1,EMBEDDING_DIM,weights=[embedding_matrix],input_length=num_steps,trainable=False)(statement_input) #Preloaded glove embeddings
    # x = Embedding(output_dim=hidden_size, input_dim=vocab_length+1, input_length=num_steps)(statement_input) #Train embeddings from scratch

    # kernel_arr = []
    # for kernel in kernel_sizes:
    #     x_1 = Conv1D(filters=filter_size,kernel_size=kernel)(x)
    #     x_1 = GlobalMaxPool1D()(x_1)
    #     kernel_arr.append(x_1)
    # conv_in = keras.layers.concatenate(kernel_arr)
    # conv_in = Dropout(0.6)(conv_in)
    # conv_in = Dense(128, activation="relu")(conv_in)
    
    # metadata_input = Input(shape=(X_train_meta.shape[1],), name="metadata_input")
    # counts_input = Input(shape=(X_train_counts.shape[1],), name="counts_input")

    # # Add in metadata and counts before fully-connected layer.
    # x_meta = Dense(64, activation="relu")(metadata_input)
    # x_counts = Dense(64, activation="relu")(counts_input)
    
    # # Output.
    # x_fc = keras.layers.concatenate([conv_in, x_meta, x_counts])
    # main_output = Dense(NUM_CLASSES, activation="softmax", name="main_output")(x_fc)
    
    # model = Model(inputs=[statement_input, metadata_input, counts_input], outputs=[main_output])

    ###########################
    # Visualize Models
    ###########################

    #Visualize model
    plot_model(model, to_file="model_lstm.svg", show_shapes=True, show_layer_names=True)
    #from IPython.display import SVG
    #from keras.utils.vis_utils import model_to_dot
    #SVG(model_to_dot(model,show_shapes=True).create(prog="dot", format="svg"))

    ###########################
    # Training
    ###########################

    print("="*20 + " Training Phase " + "="*20)
    print()

    learning_rate = 0.025
    clip_value = 0.3
    
    #Compile model and print summary
    #Define specific optimizer to counter over-fitting
    sgd = optimizers.SGD(lr=learning_rate, clipvalue=clip_value, nesterov=True)
    #adam = optimizers.Adam(lr=0.000075, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=sgd,loss="categorical_crossentropy",metrics=["categorical_accuracy", f1])
    print(model.summary())

    #Output stuff to tensorboard and write csv too. So I will define some callbacks
    tb = TensorBoard()
    csv_logger = keras.callbacks.CSVLogger("logs/training.log")
    filepath = "weights.best.hdf5"
    # Use val_categorical_accuracy. Or val_loss depending on whatever the heck you want.
    checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor="val_categorical_accuracy", verbose=1, save_best_only=True, mode="max")
    
    #Start training the functional model here
    history = model.fit({"statement_input": X_train, "metadata_input": X_train_meta, "counts_input": X_train_counts},
                        {"main_output": Y_train}, 
                          epochs=num_epochs, batch_size=batch_size,
                          validation_data=(
                            {"statement_input": X_val, "metadata_input": X_val_meta, "counts_input": X_val_counts},
                            {"main_output": Y_val}),
                          callbacks=[tb,csv_logger,checkpoint])

    # Plot training & validation accuracy values
    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['val_categorical_accuracy'])
    plt.title('LSTM Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig("lstm_accuracy_curves.png", format = 'png')
    plt.close()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('LSTM Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig("lstm_loss_curves.png", format = 'png')
    plt.close()

    print("Model trained.")

    ###########################
    # Evaluation
    ###########################

    print("="*20 + " Evaluation Phase " + "="*20)
    print()

    # Load pre-trained model (if exists).
    pretrained_model = load_model("weights.best.hdf5", custom_objects={'f1': f1})

    evaluate_results = pretrained_model.evaluate({"statement_input": X_test, "metadata_input": X_test_meta, "counts_input": X_test_counts},
                   {"main_output": Y_test},
                    batch_size=batch_size,
                   )
    test_loss, test_accuracy, test_f1 = evaluate_results[0], evaluate_results[1], evaluate_results[2] 
    
    print("Results:")
    print("test_loss                :", test_loss)
    print("test_categorical_accuracy:", test_accuracy)
    print("test_f1                  :", test_f1)

    ###########################
    # Predictions (Write to File) 
    ###########################

    print("Writing predictions... ")

    preds = pretrained_model.predict([X_test, X_test_meta, X_test_counts], batch_size=batch_size)

    vf = open(output_file, "w+")
    num_preds = 0
    for pred in preds:  
        line_string = LABEL_LIST_REVERSE[np.argmax(pred)]
        vf.write(line_string+"\n")
        num_preds += 1
    print("%s predictions written to %s." % (num_preds, output_file))
    vf.close()

if __name__ == "__main__":
    if len(sys.argv) >= 5:
        run(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        run("liar_dataset/train.tsv", "liar_dataset/valid.tsv", "liar_dataset/test.tsv", "predictions.txt")

