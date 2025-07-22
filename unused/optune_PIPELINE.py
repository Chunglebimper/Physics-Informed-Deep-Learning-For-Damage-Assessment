import optuna
from train import train_and_eval


def objective(trial):
    # function to be optimizd
    x = trial.suggest_int('batch_size', 1,100, log=True)

    param_dict['batch_size'] = x
    f1, _, _ = train_and_eval(use_glcm=True,
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

param_dict = {'batch_size': 1, 'patch_size': 128, 'stride':64, 'epochs':10, 'lr':1e-6, 'sample_size':128, 'levels':32, 'weights_str':'1,1,1,1,1'}

study = optuna.create_study()
study.optimize(objective, n_trials=10)
best_params = study.best_params

found_x = best_params['batch_size']
print(f"Found 'batch_size': {found_x}")