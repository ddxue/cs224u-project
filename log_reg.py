from util import *

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