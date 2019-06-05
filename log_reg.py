from util import *
from sklearn.linear_model import LogisticRegression

class ValConf():
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
        #tex = str(self.train_f1) + ' & ' + str(self.train_acc) + ' & ' + str(self.val_f1) + ' & ' + str(self.val_acc)
        tex = str(self.val_f1) + ' & ' + str(self.val_acc) + '\\\\'
        
        return tex
        res = ''
        for i in range(len(self.param_names)):
            res += self.param_names[i] + ': ' + str(self.params[i]) + ' '
        res += '|| train f1: ' + str(self.train_f1)
        res += ' train acc: ' + str(self.train_acc)
        res += ' val f1: ' + str(self.val_f1)
        res += ' val acc: ' + str(self.val_acc)

        return res + '\n' + tex
        return res

def hyperparam_search(mode=HEXARY):
    model = Sentence2Vec(MODEL_FILENAME)
    val_results = []
    for normalize in [True, False]:
        train_X, train_y, val_X, val_y, test_X, test_y \
            = load_all_data(model, normalize=normalize, mode=mode)
        #train_X = train_X[:,200:-5]
        #val_X = val_X[:,200:-5]
        for reg in ['l1', 'l2']:
            for C in [10, 1, 0.1]:
                val_results.append(get_results(train_X, train_y, val_X, val_y, 
                    normalize=normalize, C=C, reg=reg, mode=mode))

    print("Summary of results")
    best_result = val_results[0]
    for val_result in val_results:
        print(val_result)
        if val_result.val_f1 > best_result.val_f1:
            best_result = val_result

    print("Best result:")
    print(best_result)

def ablation_study(mode=BINARY):
    model = Sentence2Vec(MODEL_FILENAME)
    val_results = []
    train_X, train_y, val_X, val_y, test_X, test_y \
        = load_all_data(model, normalize=False, mode=mode)
    # print(np.mean(val_y))
    # Hardcoded best settings from hyperparam search
    # Both had normalize=False and reg='l1'
    C = (10 if mode == BINARY else 0.1)
    
    for lang in [True, False]:
        for con in [True, False]:
            for rep in [True, False]:
                print("Lang: {}, Con: {}, Rep: {}".format(lang, con, rep))
                if lang or con or rep:
                    if lang and con and rep:
                        mod_train_X, mod_val_X = train_X, val_X
                    elif lang and con and not rep:
                        mod_train_X, mod_val_X = train_X[:,:-5], val_X[:,:-5]
                    elif lang and not con and rep:
                        #todo
                        mod_train_X, mod_val_X = np.hstack((train_X[:,:200], 
                            train_X[:,-5:])), np.hstack((val_X[:,:200], val_X[:,-5:]))
                    elif lang and not con and not rep:
                        mod_train_X, mod_val_X = train_X[:,:200], val_X[:,:200]
                    elif not lang and con and rep:
                        mod_train_X, mod_val_X = train_X[:,200:], val_X[:,200:]
                    elif not lang and con and not rep:
                        mod_train_X, mod_val_X = train_X[:,200:-5], val_X[:,200:-5]
                    elif not lang and not con and rep:
                        mod_train_X, mod_val_X = train_X[:,-5:], val_X[:,-5:]
                    val_results.append(get_results(mod_train_X, train_y, mod_val_X, val_y, 
                        normalize=False, C=C, reg='l1', mode=mode))

    print("Summary of results")
    best_result = val_results[0]
    for val_result in val_results:
        print(val_result)
        if val_result.val_f1 > best_result.val_f1:
            best_result = val_result

def get_results(train_X, train_y, val_X, val_y, normalize=False, C=1, reg='l2', mode=HEXARY):
    print("Fitting Model...")
    log_reg = LogisticRegression(C=C, 
        penalty=reg,
        solver=('lbfgs' if reg == 'l2' else 'saga'),
        max_iter=150, 
        multi_class='multinomial').fit(train_X, train_y)

    val_yhat = log_reg.predict(val_X)
    train_yhat = log_reg.predict(train_X)
    print("Evaluating..")
    print("Train Results")
    train_conf, train_f1, train_acc = evaluate(train_y, train_yhat, mode=mode)
    print("Val Results")
    val_conf, val_f1, val_acc = evaluate(val_y, val_yhat, mode=mode)
    return ValConf(['normalize', 'C', 'reg'], 
        [normalize, C, reg], 
        train_conf, train_f1, train_acc, 
        val_conf, val_f1, val_acc)

if __name__ == '__main__':
    #hyperparam_search(mode=HEXARY)
    ablation_study(mode=BINARY)