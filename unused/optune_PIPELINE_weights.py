import optuna
from train import train_and_eval

"""
DEFINE THE GOAL HERE:
    * We are trying to obtain paramaters/weights to give us high F1 (minority and majority) and accuracy
    * to seperate the wieghts and test each one, we need to find a way to handle feeidn the wieghts in
    * one weight at a time...
    """


def objective(trial):
    # function to be optimized
    x = trial.suggest_float('weights_str', 1,20, log=True)

    param_dict['weights_str'] = x
    f1, accuracy, val_loss = train_and_eval(use_glcm=True,
                       patch_size=int(param_dict.get('patch_size')),
                       stride=int(param_dict.get('stride')),
                       batch_size=int(param_dict.get('batch_size')),
                       epochs=int(param_dict.get('epochs')),
                       lr=float(param_dict.get('lr')),
                       root='../data',
                       verbose=False,
                       sample_size=int(param_dict.get('sample_size')),
                       levels=int(param_dict.get('levels')),
                       save_name='./___________________________',
                       weights_str=str(param_dict.get('weights_str')),
                       class0and1percent=10)
    trial.set_user_attr("f1", f1)
    return f1

# load initial data
param_dict = {'batch_size': 1, 'patch_size': 128, 'stride':64, 'epochs':10, 'lr':1e-6, 'sample_size':128, 'levels':32,
              'weights_str':'1,1,1,1,1'}
classes = class1, class2, class3, class4, class5 = param_dict['weights_str'].split(',')

# adjust lower classes first
for class_ in reversed(classes):
    study = optuna.create_study(directions= ['maximize', 'maximize', 'minimize'])
    study.optimize(objective, n_trials=10)
    best_params = study.best_params
    found_x = best_params['weights_str']
    print(f"Found 'weights_str': {found_x}")
    print(class_)

param_dict['weights_str'] = f'{class1}, {class2}, {class3}, {class4}, {class5}'







""""
study = optuna.create_study(directions= ['maximize', 'maximize', 'minimize'])
study.optimize(objective, n_trials=10)
best_params = study.best_params

found_x = best_params['weights_str']
print(f"Found 'weights_str': {found_x}")
"""

class1, class2, class3, class4, class5 = param_dict['weights_str'].split(',')


#study = optuna.create_study(directions= ['maximize', 'maximize', 'minimize'])
#study.optimize(objective, n_trials=10)
#best_params = study.best_params

#found_x = best_params['batch_size']
#print(f"Found 'batch_size': {found_x}")