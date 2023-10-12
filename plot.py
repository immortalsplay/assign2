import matplotlib.pyplot as plt
import numpy as np
import GP
import math
# data_list1 = []
# with open('Random_long_lengths.txt', 'r') as file:
#     lines = file.readlines()
#     for line in lines:
#         data_list1.append(eval(line.strip()))

# # 读取文件并处理数据
# with open('HC_history.txt', 'r') as file:
#     lines = file.readlines()

# loss_values = []

# for line in lines:
#     if "Loss:" in line:
#         loss = float(line.split(',')[0].split(':')[1].strip())
#         loss_values.append(loss)


with open('GA_movie.txt', 'r') as file:
    lines = file.readlines()

# 获取最后一个数据
# last_line = lines[-1]
# gen_number = int(last_line.split(' ')[1])

# 复制最后一个数据4000次
# for _ in range(4000):
#     gen_number += 1
#     new_line = last_line.replace(f"Generation {lines[-1].split(' ')[1]}", f"Generation {gen_number}")
#     lines.append(new_line)

# # 保存修改后的数据回文件
# with open('GA.txt', 'w') as file:
#     file.writelines(lines)

# loss_values_ga = []

# for line in lines:
#     if "Loss:" in line:
#         loss = float(line.split('Loss:')[1].strip())
#         loss_values_ga.append(loss)

TERMINALS = list(range(1, 10)) + ['x']

def compute_tree(tree, x):
    if not tree:
        return 0  # 或返回其他默认值
    func = tree[0]
    if func in TERMINALS:
        return x if func == 'x' else float(func)
    elif func == '+':
        return compute_tree(tree[1:int(len(tree)/2)+1], x) + compute_tree(tree[int(len(tree)/2)+1:], x)
    elif func == '-':
        return compute_tree(tree[1:int(len(tree)/2)+1], x) - compute_tree(tree[int(len(tree)/2)+1:], x)
    elif func == '*':
        return compute_tree(tree[1:int(len(tree)/2)+1], x) * compute_tree(tree[int(len(tree)/2)+1:], x)
    elif func == '/':
        denominator = compute_tree(tree[int(len(tree)/2)+1:], x)
        if denominator != 0:
            return compute_tree(tree[1:int(len(tree)/2)+1], x) / denominator
        else:
            return 1
    elif func == 'sin':
        return math.sin(compute_tree(tree[1:], x))
    elif func == 'cos':
        return math.cos(compute_tree(tree[1:], x))
    else:
        print(f"Unrecognized function or terminal: {func}")
        print(tree)
        return 0  # Default value if unrecognized
    
generations = []
best_trees = []

# for line in lines:
#     parts = line.split(" Best Tree: ")
#     generations.append(int(parts[0].split(' ')[1]))
#     best_trees.append(parts[1].split(" Loss:")[0].strip())
# 绘制曲线
for line in lines:
    parts = line.strip().split('\n')
    generations = [int(line.split()[1]) for line in parts]
    best_trees = [line.split(":")[1].strip().split() for line in parts]


data = np.loadtxt('data\Bronze.txt', delimiter=',')
x_data = data[:, 0]
y_data = data[:, 1]


for i, best_tree in enumerate(best_trees):
    y_pred = [compute_tree(best_tree, x) for x in x_data]
    
    plt.figure(figsize=(10,6))
    plt.plot(x_data, y_data, label="True Data", color='black')
    plt.plot(x_data, y_pred, label=f"Gen {generations[i]}", linestyle="--")
    
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.title(f"True vs. Predicted Data for Generation {generations[i]}")
    plt.savefig(f"true_vs_predicted_gen_{generations[i]}.png")
    plt.show()



# plt.figure(1)

# # plt.plot(loss_values_ga, label='Loss', color='blue')

# # # 添加error bars每隔5000次迭代
# # for i in range(0, len(loss_values_ga), 100):
# #     plt.errorbar(i, loss_values_ga[i], yerr=1000/(i+1), color='red', fmt='o', capsize=5)

# # plt.title('Loss over Generations with Error Bars')
# # plt.xlabel('Generation')
# # plt.ylabel('Loss')
# # plt.legend()
# # plt.grid(True)

# # plt.scatter(range(len(loss_values_ga)), loss_values_ga, marker='o')
# # plt.title('Dot plot')
# # plt.grid(True)
# # plt.show()

# # plt.plot(loss_values_ga, marker='o')
# # plt.xlabel('Generation')
# # plt.ylabel('Loss')
# # plt.grid(True)

# # x2 = np.arange(0, len(loss_values)) * 50000/len(loss_values)  # 这里的间隔需要和你实际的间隔相匹配
# # plt.plot(x2, loss_values, label='Loss over HC')
# # for i in range(0, len(x2), 5):
# #     plt.errorbar(i, loss_values[i], yerr=5000/(i+1), color='red', fmt='o', capsize=5)
# # plt.xlabel('Iteration')
# # plt.ylabel('Loss')
# # plt.grid(True)

# # iterations = np.arange(0, 50000, 1)  # 这里的间隔需要和你实际的间隔相匹配
# # plt.plot(iterations, data_list1, label='Random Average Cost')
# # plt.legend()
# # plt.xlabel("iteration number")
# # plt.ylabel("fitness")
# # plt.title("Variation of Problem Length with the Number of Iterations")

# plt.show()
