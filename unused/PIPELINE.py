import os
from os import chdir
from train import train_and_eval
from log import Log


# divide and conquer

def run(num, param):
    global param_dict
    param_dict[param] = num
    x = train_and_eval(use_glcm=True,
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
    log.append(f'{param_dict[param]} : {num}')
    log.append(f'\tF1: {x}\n')
    return x

# --------------------------------------------------------------------------------------------------------

def find_max(arr, low, high, best, param):
    """
    :param arr: Array of numbers to try
    :param low: Lower bound
    :param high: Upper bound
    :param best:
    :param param:
    :return: returns best param value based on F1 scoring from run()
    """
    global recursive_step

    if low == high:
        return arr[low]  # Base case: one element

    mid = (low + high) // 2

    left_max = find_max(arr, low, mid, best, param)
    right_max = find_max(arr, mid + 1, high, best, param)
    recursive_step += 1
    print(recursive_step)

    if run(left_max, param) > run(right_max, param):
        return left_max
    else:
        return right_max

# --------------------------------------------------------------------------------------------------------

"""
log = Log('pipelog.txt')
log.open()
arr = list(range(1000001))
____param_dict = {'batch_size':  2,
              'patch_size': 64,
              'stride':     32,
              'epochs':      0,
              'lr':          0,
              'sample_size' :0,
              'levels' :     0,
              'weights_str': [1,1,1,1,1]}
"""

param_dict = {}
PARAMS_TO_CHANGE = ['batch_size', 'patch_size', 'stride', 'epochs', 'lr', 'sample_size', 'levels', 'weights_str']

# ---------------- Load params ----------------
key_num = 0
with open('../src/pipe/test.txt', 'r') as src:
    for line in src:
        line = line.strip()
        param_dict[ PARAMS_TO_CHANGE[ key_num ] ] = line
        key_num += 1
# ---------------------------------------------

# Open log to write updates
log = Log("./pipe/pipelog.txt")
log.open()
os.makedirs(f'../src/pipe', exist_ok=True)

# Open another log for best new params
with open('../src/pipe/test.txt', 'w') as src:

    # Initiate param optimization
    for param in param_dict:
        recursive_step = 0
        if param == 'batch_size':
            src.write(f"{ find_max(range(1,100), 1, 101, 0, 'batch_size') }\n")
            #src.write(f"2\n")
            src.flush()

        elif param == 'patch_size':
            src.write('64\n')
            pass
        elif param == 'stride':
            src.write('32\n')
            pass
        elif param == 'epochs':
            src.write('15\n')
            pass
        elif param == 'lr':
            src.write('1e-6\n')
            pass
        elif param == 'sample_size':
            src.write('128\n')
            pass
        elif param == 'levels':
            src.write('32\n')
            pass
        elif param == 'weights_str':
            src.write('1,1,1,1,1')
            pass
        elif param == 'new_param_name_here':
            pass
        else:
            print(f"Parameter: '{param}' array to run not found")

    #print(f"{find_max(arr, 0, len(arr)-1, 0, 'batch_size')}")

log.close()








"""
Read baseline file (baseline.txt)
Run model with baseline params, save this param and F1
Check first param and adjust; save this param and F1
    If this param makes F1 lower by > 10% change param 

    

compare.txt:    Values to compare against
test.txt:       The best values so far
pipelog.txt:    Review all changes

Make an alogorithm to narrow it down
"""
"""
def run_param(param, number):
    # run 10 tests
    # return avg f1
    pass

listy = list(range(1,100 +1))
print(listy)

def recursive_higher_lower(listy, start, end, best, param, powerOf2=False):

    :param param: The number
    :param powerOf2: is it a number of power 2
    :return:
    
    # if len(listy) == 1:
        # break
    # elif F1 is greater than best:
        #return recursive(listy[start:end/2])                 # error with floats
    # else:
        # return (recursive_higher_lower(listy, start, end, best, param)

# elif

"""
"""
    if len(listy) == 1:
        return listy[0]
    elif run_param(param, listy[0]) > best:
        mid = (start + end) // 2
        
        return recursive_higher_lower(listy[start, mid])             # error with floats
    else:



    if train_and_eval(use_glcm=True,
                patch_size=int(patch_size),
                stride=int(stride),
                batch_size=int(batch_size),
                epochs=int(epochs),
                lr=float(lr),
                root='../data',
                verbose=False,
                sample_size=int(sample_size),
                levels=int(levels),
                save_name='./___________________________',
                weights_str=str(weights_str),
                class0and1percent=10) >

    output_of_F1
    return None

    # Read params
    with open('test.txt', 'r') as src:
        for line in src:
            x = line.strip().split(' ')
            batch_size, patch_size, stride, epochs, lr, sample_size, levels, weights_str = x
            print(x)
            log.append(x)


best_F1 = 0
params = [batch_size, patch_size, stride, epochs, lr, sample_size, levels, weights_str]
while best_F1 < 99:
    for param in params:
        # if param == 'patch_size' power of 2

        optimal_num = higher_lower(1, 100) # return number that has best F1
        log.append(optimal_number)
















BATCH_SIZE = [128, 64, 32, 16, 8, 4, 2, 1]
PATCH_SIZE = [256, 128, 64, 32, 16, 8, 4]
STRIDE = [128,64, 32, 16, 8, 4, 2]


log = Log(path='pipelog.txt')
log.open()
best_f1 = 0

# While F1 is not optimal
while best_f1 < 99:

    # Read params
    with open('test.txt', 'r') as src:
        for line in src:
            x = line.strip().split(' ')
            batch_size, patch_size, stride, epochs, lr, sample_size, levels, weights_str = x
            print(x)
            log.append(x)

    # Use read configs for 10 trials
    to_average = []
    for i in range(10):
        to_average.append(
            train_and_eval(
                use_glcm=True,
                patch_size=int(patch_size),
                stride=int(stride),
                batch_size=int(batch_size),
                epochs=int(epochs),
                lr=float(lr),
                root='../data',
                verbose=False,
                sample_size=int(sample_size),
                levels=int(levels),
                save_name='./___________________________',
                weights_str=str(weights_str),
                class0and1percent=10
            )
        )

    # Record average F1
    avg_f1 = sum(to_average) / len(to_average)

    # Save F1 and adjust params or loop back
    # If avgF1 greater than bestF1
    if avg_f1 > best_f1:
        best_f1 = avg_f1
        log.append()

    # If bestf1 is the run again
    elif avg_f1 == best_f1:
        pass

    else:
        with open('test.txt', 'w') as src:
            for line in src:
                src.write(f"{batch_size * 2}")
        log.append()

    log.append(f"{batch_size, patch_size, stride, epochs, lr, sample_size, levels, weights_str}")

log.close()
"""
