def run(num, param):
    global param_dict
    x = num + 1
    return x

def find_max(arr, low_idx, high_idx, best, param):
    """
    :param arr: Array of numbers to try
    :param low_idx: Lower bound index
    :param high_idx: Upper bound index
    :param best:
    :param param:
    :return: returns best param value based on F1 scoring from run()
    """
    global recursive_step

    if low_idx == high_idx:
        return arr[low_idx]  # Base case: one element

    mid_idx = (low_idx + high_idx) // 2

    left_max = find_max(arr, low_idx, mid_idx, best, param)
    right_max = find_max(arr, mid_idx + 1, high_idx, best, param)
    recursive_step += 1
    #print(recursive_step)
    print(f'{recursive_step}: {arr[low_idx:mid_idx]}   {arr[mid_idx+1:high_idx]}')

    if run(left_max, param) > run(right_max, param):
        return left_max
    else:
        return right_max

recursive_step = 0
print(find_max(range(1,101), 0, 99, 'null', 'null'))
