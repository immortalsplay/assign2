import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import matplotlib.pyplot as plt
import random

data = np.loadtxt('data\Bronze.txt', delimiter=',')

# 提取x和y
x_data = data[:, 0]
y_data = data[:, 1]

def term_evaluation(coef, val, operation):
    if operation == 0:
        return coef * val
    elif operation == 1:
        return -coef * val
    elif operation == 2:
        return coef * val * val
    elif operation == 3:
        return 1 / (coef * val + 1e-6)
    elif operation == 4:
        return np.sin(coef * val)
    elif operation == 5:
        return np.cos(coef * val)

def custom_function(a, b, c, x, operations):
    term1 = term_evaluation(a, x**2, operations[0])
    term2 = term_evaluation(b, x, operations[1])
    term3 = term_evaluation(c, 1, operations[2])
    # term3 = term_evaluation(d, 1, operations[2])
    
    return term1 + term2 + term3

def loss_function(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred))

# def random_optimize(x_data, y_true, iteration_num=50000):
    operations_initial = [0, 0, 0]
    y_pred_initial = custom_function(a=1, b=1, c=1, x=x_data, operations=operations_initial)
    loss_initial = loss_function(y_data, y_pred_initial)

    # Hillclimber参数
    best_operations = operations_initial
    best_loss = loss_initial

    for i in range(iteration_num):
    # 对于每个操作，生成所有可能的邻居
        neighbors = []
        for idx in range(3):
            for op in range(6):
                if op != best_operations[idx]:
                    new_operations = best_operations.copy()
                    new_operations[idx] = op
                    neighbors.append(new_operations)

        # 计算所有邻居的损失并找到最小损失的邻居
        losses = []
        for neighbor in neighbors:
            y_pred_neighbor = custom_function(a=1, b=1, c=1, x=x_data, operations=neighbor)
            losses.append(loss_function(y_data, y_pred_neighbor))

        min_loss_idx = np.argmin(losses)
        
        if i % 500 == 0:
            print("Iteration: {}, Loss: {}".format(i, best_loss))

        # 如果找到的邻居的损失更小，则更新最佳解
        if losses[min_loss_idx] < best_loss:
            best_operations = neighbors[min_loss_idx]
            best_loss = losses[min_loss_idx]

    print("Best operations:", best_operations)
    print("Best loss:", best_loss)

    return best_operations, best_loss

def random_optimize(x_data, y_true, iteration_num=50000):
    operations_initial = [0, 0, 0]
    a, b, c = 1, 1, 1  # 初始化参数值
    y_pred_initial = custom_function(a, b, c, x_data, operations_initial)
    loss_initial = loss_function(y_data, y_pred_initial)

    best_operations = operations_initial
    best_loss = loss_initial
    best_parameters = [a, b, c]

    history = []
    history.append((best_loss, best_parameters, best_operations))

    for i in range(iteration_num):
        # 对于每个操作，生成所有可能的邻居
        neighbors = []
        for idx in range(3):
            for op in range(6):
                if op != best_operations[idx]:
                    new_operations = best_operations.copy()
                    new_operations[idx] = op
                    neighbors.append(new_operations)

        # 随机选择一个参数并为其分配一个新的随机值
        param_to_change = np.random.choice(['a', 'b', 'c'])
        new_value = float(np.random.choice(np.arange(-10, 10.1, 0.1)))
        if param_to_change == 'a':
            a = new_value
        elif param_to_change == 'b':
            b = new_value
        else:
            c = new_value

        # 计算所有邻居的损失
        losses = []
        for neighbor in neighbors:
            y_pred_neighbor = custom_function(a, b, c, x_data, operations=neighbor)
            losses.append(loss_function(y_data, y_pred_neighbor))

        min_loss_idx = np.argmin(losses)

        if i % 500 == 0:
            print("Iteration: {}, Loss: {}".format(i, best_loss))

        # 如果找到的邻居的损失更小，则更新最佳解
        if losses[min_loss_idx] < best_loss:
            best_operations = neighbors[min_loss_idx]
            best_loss = losses[min_loss_idx]
            best_parameters = [a, b, c]

            history.append((best_loss, best_parameters, best_operations))

    print("Best operations:", best_operations)
    print("Best parameters: a={}, b={}, c={}".format(best_parameters[0], best_parameters[1], best_parameters[2]))
    print("Best loss:", best_loss)

    return best_operations, best_loss, best_parameters, history

def simulated_annealing(initial_operations, initial_temp, cooling_rate, num_iterations):
    current_operations = initial_operations
    current_loss = loss_function(y_data, custom_function(a=1, b=1, c=1, x=x_data, operations=current_operations))
    best_operations = current_operations
    best_loss = current_loss
    temp = initial_temp

    for i in range(num_iterations):
        # 选择一个随机邻居
        idx_to_change = np.random.randint(0, 3)
        new_operations = current_operations.copy()
        new_operations[idx_to_change] = np.random.randint(0, 6)
        
        # 计算新损失
        new_loss = loss_function(y_data, custom_function(a=1, b=1, c=1, x=x_data, operations=new_operations))
        
        # 计算损失差值
        loss_diff = new_loss - current_loss

        # 如果新的损失更小或在温度的影响下随机选择新的解，则接受新的解
        if loss_diff < 0 or np.random.rand() < np.exp(-loss_diff / temp):
            current_operations = new_operations
            current_loss = new_loss

            if current_loss < best_loss:
                best_operations = current_operations
                best_loss = current_loss
        
        if i % 500 == 0:
            print("Iteration: {}, Loss: {}".format(i, best_loss))
        
        # 降低温度
        temp *= cooling_rate

    return best_operations, best_loss

initial_temp = 10
cooling_rate = 0.995
num_iterations = 50000
operations_initial = [0, 0, 0]
# best_operations, best_loss = simulated_annealing(operations_initial, initial_temp, cooling_rate, num_iterations)

# print("Best operations:", best_operations)
# print("Best loss:", best_loss)

def plot_results(x_data, y_data, y_pred):
    plt.plot(x_data, y_data, 'b.')
    plt.plot(x_data, y_pred, 'r-')
    plt.show()

best_operations, best_loss, best_parameters,history= random_optimize(x_data, y_data, iteration_num=5000)
y_pred = custom_function(a=best_parameters[0], b=best_parameters[1], c=best_parameters[2], x=x_data, operations=best_operations)
plot_results(x_data, y_data, y_pred)


# 保存history到txt文件
with open('HC_history.txt', 'w') as file:
    for record in history:
        loss, params, ops = record
        file.write(f"Loss: {loss}, Params: {params}, Operations: {ops}\n")

