import os
from os import chdir
from train import train_and_eval
from log import Log


# divide and conquer

def run(num, param):
    return num

def find_max(arr, low, high, best, param):
    if low == high:
        return arr[low]  # Base case: one element

    mid = (low + high) // 2

    left_max = find_max(arr, low, mid, best, param)
    right_max = find_max(arr, mid + 1, high, best, param)

    if run(left_max, param) > run(right_max, param):
        return left_max
    else:
        return right_max



arr = list(range(1000001))
print(find_max(arr, 0, len(arr)-1, 0, 'null'))








"""
Read baseline file (baseline.txt)
Run model with baseline params, save this param and F1
Check first param and adjust; save this param and F1
    If this param makes F1 lower by > 10% change param 

    

compare.csv:    Values to compare against
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
