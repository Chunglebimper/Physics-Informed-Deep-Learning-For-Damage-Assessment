import os
from os import chdir
from train import train_and_eval
from log import Log

"""
Read baseline file (baseline.txt)
Run model with baseline params, save this param and F1
Check first param and adjust; save this param and F1
    If this param makes F1 lower by > 10% change param 
compare.csv:    Values to compare against
test.txt:       The best values so far
pipelog.txt:    Review all changes
"""
log = Log(path='pipelog.txt')
log.open()
best_f1 = 0

while best_f1 < 99:

    # Read params
    with open('test.txt', 'r') as src:
        for line in src:
            x = line.strip().split(' ')
            batch_size, patch_size, stride, epochs, lr, sample_size, levels, weights_str = x
            print(x)
            log.append(x)

    # Use configs for 10 trials
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
    if avg_f1 > best_f1:
        best_f1 = avg_f1
    elif avg_f1 == best_f1:
        continue
    else:
        continue
