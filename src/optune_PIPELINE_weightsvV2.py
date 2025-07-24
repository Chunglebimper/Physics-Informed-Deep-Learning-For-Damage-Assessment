import optuna
from train import train_and_eval
from log import Log
import os

def objective(trial, class_to_optimize):
    # Function to be optimized
    x = trial.suggest_float(class_to_optimize, 1,20, log=True)
    weights_dict[class_to_optimize] = x

    # Prep weights to feed model
    weights = []
    for key in weights_dict:
        weights.append(str(weights_dict[key]))
    to_model = ','.join(weights)

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
                       save_name='./____________optune2',
                       weights_str=str(to_model), # <----------------- what we care about
                       class0and1percent=10)
    trial.set_user_attr("f1", f1)
    trial.set_user_attr("accuracy", accuracy)
    trial.set_user_attr("val_loss", val_loss)
    return f1, accuracy, val_loss






log = Log("./pipe/pipelogv2.txt")
log.open()
os.makedirs(f'./pipe', exist_ok=True)

param_dict = {'batch_size': 4, 'patch_size': 128, 'stride':64, 'epochs':10, 'lr':1e-6, 'sample_size':128, 'levels':32,
              'weights_str':'1,1,1,1,1'}
weights_dict = {'class5': 1, 'class4': 1, 'class3': 1, 'class2': 1, 'class1': 1}

for class_key in weights_dict:
    study = optuna.create_study(directions=['maximize', 'maximize', 'minimize'])
    study.optimize(lambda trial: objective(trial, class_to_optimize=class_key), n_trials=50) # what?????
    print("--------------------------------------------")
    # My code:
    #best_params = study.best_params
    #found_x = best_params[class_key]

    # Correct code?
    pareto_trials = study.best_trials
    try:
        print(pareto_trials)
        log.append(f'pareto_trials {pareto_trials}')
    except Exception as e:
        print(e)
        log.append(f'{e}\n')
    best_trial = max(pareto_trials, key=lambda t: t.user_attrs["f1"]) # dont know what this line does but it solves an issue i had
    found_x = best_trial.params[class_key]

    msg = f"Found '{class_key}': {found_x}"
    print(msg)
    log.append(f'{msg}\n')

for key in weights_dict:
    print(weights_dict[key])

log.close()
