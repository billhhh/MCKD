import re
import matplotlib.pyplot as plt
import numpy as np

# Read and extract data from the log file
log_filename = '/home/bill/Documents/Projects/missing_modal/logs/SoftLCKD/visual/draw_log/' \
               '20230808-221854_train_20230808_BraTS18_[80,160,160]_SGD_b2_lr-2_KDLossWt1_val5_8w_randInit_Softmax_Adam_L2.log'

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
iter_values_val = []
tensor_data_val = []
window_size = 100
start_from = 100

for idx in range(len(lines)):
    line = lines[idx]

    if ' completed, lr = ' in line:
        iter_match = re.search(r'iter = (\d+)', line)
        iter_value = int(iter_match.group(1))
        if iter_value < start_from: continue
        iter_values.append(iter_value)
        print('iter = ', iter_value)

        tensor_match = re.search(r'seg_loss = (\d+\.\d+)', line)

        try:
            tensor_values = float(tensor_match.group(1))
        except BaseException:
            print("Break!")
            break
        tensor_data.append(tensor_values)

    elif 'validate ...' in line:
        iter_match = re.search(r'iter = (\d+)', lines[idx - 1])
        iter_value = int(iter_match.group(1))
        if iter_value < start_from: continue
        iter_values_val.append(iter_value)
        print('iter = ', iter_value)

        if 'val_loss: ' in lines[idx + 3]:
            tensor_match = re.search(r'val_loss: (\d+\.\d+)', lines[idx + 3])
        elif 'val_loss: ' in lines[idx + 4]:
            tensor_match = re.search(r'val_loss: (\d+\.\d+)', lines[idx + 4])
        elif 'val_loss: ' in lines[idx + 5]:
            tensor_match = re.search(r'val_loss: (\d+\.\d+)', lines[idx + 5])

        try:
            tensor_values = float(tensor_match.group(1))
        except BaseException:
            print("Break!")
            break
        tensor_data_val.append(tensor_values)

iter_values = iter_values[:len(tensor_data)]
tensor_data = tensor_data[:len(iter_values)]
iter_values_val = iter_values_val[:len(tensor_data_val)]
tensor_data_val = tensor_data_val[:len(iter_values_val)]

tensor_data = moving_average(tensor_data, window_size)
tensor_data_val = moving_average(tensor_data_val, window_size)

# Draw
colors = ['b', 'g', 'r', 'c']
labels = ['Training', 'Validation']

plt.figure(figsize=(10, 6))

plt.plot(iter_values, tensor_data, colors[0], linewidth=3.0, label=labels[0])
plt.plot(iter_values_val, tensor_data_val, colors[2], linewidth=3.0, label=labels[1])

plt.xlabel('Iteration', fontsize="20")
plt.ylabel('Loss Value', fontsize="20")
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
# plt.title('Tensor Values over Iterations')
plt.legend(fontsize=17)
plt.grid(True)
plt.show()
