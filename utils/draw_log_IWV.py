import re
import matplotlib.pyplot as plt
import numpy as np

# Read and extract data from the log file
log_filename = '/home/bill/Documents/Projects/missing_modal/logs/SoftLCKD/visual/draw_log/' \
               'combine2_20230518-114835_train_20230518_BraTS18_[80,160,160]_SGD_b2_lr-2_KDLossWt.1_val5_8w_randInit_Softmax_Adam.log'

data = []
with open(log_filename, 'r') as file:
    lines = file.readlines()


def moving_average(data, window_size):
    npa = np.asarray(data, dtype=np.float32)
    smoothed_data = []
    for i in range(len(npa)):
        start_idx = max(0, i - window_size + 1)
        end_idx = i + 1
        window = npa[start_idx:end_idx]
        smoothed_value = window.sum(axis=0) / len(window)
        smoothed_data.append(smoothed_value)
    return smoothed_data


iter_values = []
tensor_data = []
window_size = 1000

for idx in range(len(lines)):
    line = lines[idx]
    iter_match = re.match(r'iter = (\d+)', line)

    if iter_match:
        iter_value = int(iter_match.group(1))
        if iter_value < 0: continue
        iter_values.append(iter_value)
        print(iter_value)

        tensor_match = re.search(r'\[([-\d.e, ]+)\]', lines[idx-3])
        if tensor_match == None: tensor_match = re.search(r'\[([-\d.e, ]+)\]', lines[idx-2])
        if tensor_match == None: tensor_match = re.search(r'\[([-\d.e, ]+)\]', lines[idx-1])

        try:
            tensor_values = list(map(float, tensor_match.group(1).split(',')))
        except BaseException:
            print("Break!")
            break
        tensor_data.append(tensor_values)

iter_values = iter_values[:len(tensor_data)]
tensor_data = tensor_data[:len(iter_values)]

tensor_data = moving_average(tensor_data, window_size)

# Draw
tensor_data = list(zip(*tensor_data))
colors = ['b', 'g', 'r', 'c']
labels = ['Flair', 'T1', 'T1c', 'T2']

plt.figure(figsize=(10, 6))

for i, color in enumerate(colors):
    plt.plot(iter_values, tensor_data[i], color, label=labels[i])

fontsize = 20
plt.xlabel('Iteration', fontsize=fontsize)
plt.ylabel('IWV Value', fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
# plt.title('Tensor Values over Iterations')
plt.legend(fontsize=fontsize)
plt.grid(True)
plt.show()
