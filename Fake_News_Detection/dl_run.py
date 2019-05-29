import time
import sys
import os.path

import pickle

import pandas as pd
import numpy as np

import keras.utils
from keras.callbacks import TensorBoard, CSVLogger
from keras import backend as K

from metadata_features import *

###########################
#  Constants            
###########################

# Glove file locations.
GLOVE_PATH = "glove.6B/glove.6B.100d.txt"  # where the glove embeddings live
EMBEDDING_DIM = 100

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
    
    # Read data.
    data_set = pd.read_table(train_file, names=COLUMNS_NAMES)
    val_set = pd.read_table(valid_file, names=COLUMNS_NAMES)
    test_set = pd.read_table(test_file, names=COLUMNS_NAMES)

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

    return data_set, val_set, test_set, embeddings_index

def create_features(data_set, val_set, test_set):
    # Create Labels.
    data_set["label_id"] = data_set["label"].apply(lambda x: LABEL_DICT[x])
    val_set["label_id"] = val_set["label"].apply(lambda x: LABEL_DICT[x])
    test_set["label_id"] = test_set["label"].apply(lambda x: LABEL_DICT[x])

    # Map speakers.
    data_set["speaker_id"] = data_set["speaker"].apply(map_speaker)
    val_set["speaker_id"] = val_set["speaker"].apply(map_speaker)
    test_set["speaker_id"] = test_set["speaker"].apply(map_speaker) #Speaker

    # Map jobs.
    data_set["job_id"] = data_set["job"].apply(map_job)
    val_set["job_id"] = val_set["job"].apply(map_job)
    test_set["job_id"] = test_set["job"].apply(map_job) #Job

    # Map parties (hyperparameter -> num_party).
    data_set["party_id"] = data_set["party"].apply(map_party)
    val_set["party_id"] = val_set["party"].apply(map_party)
    test_set["party_id"] = test_set["party"].apply(map_party) #Party

    # Map states.
    data_set["state_id"] = data_set["state"].apply(map_state)
    val_set["state_id"] = val_set["state"].apply(map_state)
    test_set["state_id"] = test_set["state"].apply(map_state) #State

    # Map subject.
    data_set["subject_id"] = data_set["subject"].apply(map_subject)
    val_set["subject_id"] = val_set["subject"].apply(map_subject)
    test_set["subject_id"] = test_set["subject"].apply(map_subject) #Subject

    #Map venues.
    data_set["venue_id"] = data_set["venue"].apply(map_venue)
    val_set["venue_id"] = val_set["venue"].apply(map_venue)
    test_set["venue_id"] = test_set["venue"].apply(map_venue) #Venue

    return data_set, val_set, test_set

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

    # Read data and word embeddings.
    data_set, val_set, test_set, embeddings_index = read_data(train_file, valid_file, test_file)

    # Featurize the data.
    data_set, val_set, test_set = create_features(data_set, val_set, test_set)

    ###########################
    #   Text preprocessing    #
    ###########################

    from keras.preprocessing.text import Tokenizer

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

    #To access particular word_index. Just load these.
    #To read a word in a sentence use keras tokenizer again, coz easy
    from keras.preprocessing.text import text_to_word_sequence
    from keras.preprocessing import sequence
    #text = text_to_word_sequence(data_set["statement"][0])
    #print text
    #val = [vocab_dict[t] for t in text]
    #print val

    def pre_process_statement(statement):
        text = text_to_word_sequence(statement)
        val = [vocab_dict[t] for t in text if t in vocab_dict] #Replace unk words with 0 index
        return val

    # Creating embedding matrix to feed in embeddings directly.
    num_words = len(vocab_dict) + 1
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word, i in vocab_dict.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    ###########################
    #   Hyper parameters      #
    ###########################

    vocab_length = len(vocab_dict.keys())
    hidden_size = 100 #Has to be same as EMBEDDING_DIM
    lstm_size = 100
    num_steps = 25
    num_epochs = 30
    batch_size = 40

    # hyperparameters for CNN
    kernel_sizes = [2,5,8]
    filter_size = 128

    # Metadata related values.
    num_party = 6
    num_state = 51
    num_venue = 12
    num_job = 11
    num_sub = 14
    num_speaker = 21

    ###########################
    #   Training instances    #
    ###########################

    #Load data and pad sequences to prepare training, validation and test data
    data_set["word_ids"] = data_set["statement"].apply(pre_process_statement)
    val_set["word_ids"] = val_set["statement"].apply(pre_process_statement)
    test_set["word_ids"] = test_set["statement"].apply(pre_process_statement)
    X_train = data_set["word_ids"]
    Y_train = data_set["label_id"]
    X_val = val_set["word_ids"]
    Y_val = val_set["label_id"]
    X_test = test_set["word_ids"]

    X_train = sequence.pad_sequences(X_train, maxlen=num_steps, padding="post",truncating="post")
    Y_train = keras.utils.to_categorical(Y_train, num_classes=6)
    X_val = sequence.pad_sequences(X_val, maxlen=num_steps, padding="post",truncating="post")
    Y_val = keras.utils.to_categorical(Y_val, num_classes=6)
    X_test = sequence.pad_sequences(X_test, maxlen=num_steps, padding="post",truncating="post")

    #Meta data preparation
    a = keras.utils.to_categorical(data_set["party_id"], num_classes=num_party)
    b = keras.utils.to_categorical(data_set["state_id"], num_classes=num_state)
    c = keras.utils.to_categorical(data_set["venue_id"], num_classes=num_venue)
    d = keras.utils.to_categorical(data_set["job_id"], num_classes=num_job)
    e = keras.utils.to_categorical(data_set["subject_id"], num_classes=num_sub)
    f = keras.utils.to_categorical(data_set["speaker_id"], num_classes=num_speaker)

    X_train_meta = np.hstack((a,b,c,d,e,f))#concat a and b
    a_val = keras.utils.to_categorical(val_set["party_id"], num_classes=num_party)
    b_val = keras.utils.to_categorical(val_set["state_id"], num_classes=num_state)
    c_val = keras.utils.to_categorical(val_set["venue_id"], num_classes=num_venue)
    d_val = keras.utils.to_categorical(val_set["job_id"], num_classes=num_job)
    e_val = keras.utils.to_categorical(val_set["subject_id"], num_classes=num_sub)
    f_val = keras.utils.to_categorical(val_set["speaker_id"], num_classes=num_speaker)

    X_val_meta = np.hstack((a_val,b_val,c_val,d_val,e_val,f_val))#concat a_val and b_val
    a_test = keras.utils.to_categorical(test_set["party_id"], num_classes=num_party)
    b_test = keras.utils.to_categorical(test_set["state_id"], num_classes=num_state)
    c_test = keras.utils.to_categorical(test_set["venue_id"], num_classes=num_venue)
    d_test = keras.utils.to_categorical(test_set["job_id"], num_classes=num_job)
    e_test = keras.utils.to_categorical(test_set["subject_id"], num_classes=num_sub)
    f_test = keras.utils.to_categorical(test_set["speaker_id"], num_classes=num_speaker)

    X_test_meta = np.hstack((a_test,b_test,c_test,d_test,e_test,f_test))#concat all test data

    ###########################
    #   Model definitions     #
    ###########################
    from keras.models import Sequential
    from keras.layers import Dense,Flatten,LSTM,Conv1D,GlobalMaxPool1D,Dropout
    from keras.layers.embeddings import Embedding
    from keras import optimizers
    from keras.layers import Input
    from keras.models import Model


    ###########################
    # LSTM Model            
    ###########################

    #statement_input = Input(shape=(num_steps,), dtype="int32", name="main_input")
    #x = Embedding(vocab_length+1,EMBEDDING_DIM,weights=[embedding_matrix],input_length=num_steps,trainable=False)(statement_input) #Preloaded glove embeddings
    #x = Embedding(output_dim=hidden_size, input_dim=vocab_length+1, input_length=num_steps)(statement_input) #Train embeddings from scratch
    #lstm_in = LSTM(lstm_size,dropout=0.25)(x)
    #meta_input = Input(shape=(X_train_meta.shape[1],), name="aux_input")
    #x_meta = Dense(64, activation="relu")(meta_input)
    #x = keras.layers.concatenate([lstm_in, x_meta])
    #main_output = Dense(6, activation="softmax", name="main_output")(x)
    #model = Model(inputs=[statement_input, meta_input], outputs=[main_output])


    ###########################
    # CNN Model             
    ###########################
    kernel_arr = []
    statement_input = Input(shape=(num_steps,), dtype="int32", name="main_input")
    #x = Embedding(vocab_length+1,EMBEDDING_DIM,weights=[embedding_matrix],input_length=num_steps,trainable=False)(statement_input) #Preloaded glove embeddings
    x = Embedding(output_dim=hidden_size, input_dim=vocab_length+1, input_length=num_steps)(statement_input) #Train embeddings from scratch

    for kernel in kernel_sizes:
        x_1 = Conv1D(filters=filter_size,kernel_size=kernel)(x)
        x_1 = GlobalMaxPool1D()(x_1)
        kernel_arr.append(x_1)
    conv_in = keras.layers.concatenate(kernel_arr)
    conv_in = Dropout(0.6)(conv_in)
    conv_in = Dense(128, activation="relu")(conv_in)

    #Meta input
    meta_input = Input(shape=(X_train_meta.shape[1],), name="aux_input")
    x_meta = Dense(64, activation="relu")(meta_input)
    x = keras.layers.concatenate([conv_in, x_meta])
    main_output = Dense(6, activation="softmax", name="main_output")(x)
    model = Model(inputs=[statement_input, meta_input], outputs=[main_output])



    ###########################
    #   Visualize models      #
    ###########################

    #Visualize model
    #from keras.utils.vis_utils import plot_model
    #plot_model(model, to_file="model_lstm.png", show_shapes=True, show_layer_names=True)
    #from IPython.display import SVG
    #from keras.utils.vis_utils import model_to_dot
    #SVG(model_to_dot(model,show_shapes=True).create(prog="dot", format="svg"))



    ###########################
    #   Training part         # 
    ###########################
    
    #Compile model and print summary
    #Define specific optimizer to counter over-fitting
    sgd = optimizers.SGD(lr=0.025, clipvalue=0.3, nesterov=True)
    #adam = optimizers.Adam(lr=0.000075, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=sgd,loss="categorical_crossentropy",metrics=["categorical_accuracy", f1])
    print(model.summary())

    #Output stuff to tensorboard and write csv too. So I will define some callbacks
    tb = TensorBoard()
    csv_logger = keras.callbacks.CSVLogger("logs/training.log")
    filepath= "weights.best.hdf5"
    # Use val_categorical_accuracy. Or val_loss depending on whatever the heck you want.
    checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor="val_categorical_accuracy", verbose=1, save_best_only=True, mode="max")
    #Start training the functional model here

    model.fit({"main_input": X_train, "aux_input": X_train_meta},
              {"main_output": Y_train},epochs=num_epochs, batch_size=batch_size,
              validation_data=({"main_input": X_val, "aux_input": X_val_meta},{"main_output": Y_val}),
             callbacks=[tb,csv_logger,checkpoint])


    ###########################
    #   Make predictions      # 
    ###########################

    from keras.models import load_model
    #Load a pre-trained model if any
    pretrained_model = load_model("weights.best.hdf5", custom_objects={'f1': f1})
    preds = pretrained_model.predict([X_test,X_test_meta], batch_size=batch_size, verbose=1)

    ###########################
    #   Write predictions     # 
    ###########################
    vf = open(output_file, "w+")
    counter = 0
    for pred in preds:  
        line_string = LABEL_LIST_REVERSE[np.argmax(pred)]
        vf.write(line_string+"\n")
        counter += 1
    print(counter)
    print("Predictions written to file")
    vf.close()


if __name__ == "__main__":
    if len(sys.argv) >= 5:
        run(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        run("liar_dataset/train.tsv", "liar_dataset/valid.tsv", "liar_dataset/test.tsv", "predictions.txt")

