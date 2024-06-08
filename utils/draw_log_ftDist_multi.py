import re
import matplotlib.pyplot as plt
import numpy as np

# Read and extract data from the log file
log_filename = '/home/bill/Documents/Projects/missing_modal/logs/SoftLCKD/visual/draw_log/' \
               '2w_20240130-141423_train_20240130_BraTS18_[80,160,160]_SGD_b2_lr-2_KDLossWt.1_val5_8w_randInit_Softmax_Adam_printFtDist_11.5w.log'

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
L2_data = []
cos_data = []
window_size = 10

for idx in range(len(lines)):
    line = lines[idx]

    if re.match(r'validate ...', line):
        iter_match = re.search(r'iter = (\d+)', lines[idx - 1])
        iter_value = int(iter_match.group(1))
        if iter_value < 400: continue
        iter_values.append(iter_value)
        print(iter_value)

        tensor_match = re.search(r'\d+\.\d+', lines[idx+4])
        L2_match = re.search(r'\d+\.\d+', lines[idx+5])
        cos_match = re.search(r'\d+\.\d+', lines[idx+6])

        try:
            tensor_values = tensor_match[0]
            L2_values = L2_match[0]
            cos_values = cos_match[0]
        except BaseException:
            print("Break!")
            break
        tensor_data.append(tensor_values)
        L2_data.append(L2_values)
        cos_data.append(cos_values)

iter_values = iter_values[:len(tensor_data)]
tensor_data = tensor_data[:len(iter_values)]

tensor_data = moving_average(tensor_data, window_size)

# Draw
colors = ['b', 'g', 'r', 'c']
labels = ['L1', 'L2', 'Cosine']

plt.figure(figsize=(10, 6))

# plt.plot(iter_values, tensor_data, colors[0], linewidth=3.0, label=labels[0])
# plt.plot(iter_values, L2_data, colors[1], linewidth=3.0, label=labels[1])
plt.plot(iter_values, cos_data, colors[2], linewidth=3.0, label=labels[2])

plt.xlabel('Iteration', fontsize="20")
plt.ylabel('Feature distance', fontsize="20")
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
# plt.title('Tensor Values over Iterations')
plt.legend(fontsize="15")
plt.grid(True)
plt.show()
