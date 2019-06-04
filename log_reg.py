from util import *

class ValResult():
    def __init__(self, param_names, params, train_conf, train_f1, train_acc, val_conf, val_f1, val_acc):
        self.param_names = param_names
        self.params = params
        self.train_conf = train_conf
        self.train_f1 = train_f1
        self.train_acc = train_acc
        self.val_conf = val_conf
        self.val_f1 = val_f1
        self.val_acc = val_acc

    def __str__(self):
        res = ''
        for i in range(len(self.param_names)):
            res += self.param_names[i] + ': ' + str(self.params[i]) + ' '
        res += '|| train f1: ' + str(self.train_f1)
        res += ' train acc: ' + str(self.train_acc)
        res += ' val f1: ' + str(self.val_f1)
        res += ' val acc: ' + str(self.val_acc)
        return res

def hyperparam_search(mode=HEXARY):
    model = Sentence2Vec(MODEL_FILENAME)
    val_results = []
    for normalize in [True, False]:
        train_X, train_y, val_X, val_y, test_X, test_y \
            = load_all_data(model, normalize=normalize, mode=mode)
        for C in [0.1, 1, 10]:
            for reg in ['l1', 'l2']:
                val_results.append(get_results(train_X, train_y, val_X, val_y, 
                    normalize=normalize, C=C, reg=reg))

    print("Summary of results")
    best_result = val_results[0]
    for val_result in val_results:
        print(val_result)
        if val_result.val_f1 > best_result.val_f1:
            best_result = val_result

    print("Best result:")
    print(best_result)

def get_results(train_X, train_y, val_X, val_y, normalize=False, C=1, reg='l2'):
    print("Fitting Model...")
    log_reg = LogisticRegression(C=C, 
        penalty=reg,
        solver=('lbfgs' if reg == 'l2' else 'saga'), 
        multi_class='multinomial').fit(train_X, train_y)

    val_yhat = log_reg.predict(val_X)
    train_yhat = log_reg.predict(train_X)
    print("Evaluating..")
    print("Train Results")
    train_conf, train_f1, train_acc = evaluate(train_y, train_yhat)
    print("Val Results")
    val_conf, val_f1, val_acc = evaluate(val_y, val_yhat)
    return ValResult(['normalize', 'C', 'reg'], 
        [normalize, C, reg], 
        train_conf, train_f1, train_acc, 
        val_conf, val_f1, val_acc)

if __name__ == '__main__':
    hyperparam_search()