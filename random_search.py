import numpy as np
import matplotlib.pyplot as plt
import random
# 读取数据
data = np.loadtxt('data\Bronze.txt', delimiter=',')

# 提取x和y
x_data = data[:, 0]
y_data = data[:, 1]



# 定义函数
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

# 定义损失函数
def loss_function(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred))

def random_optimize(x_data, y_true, iteration_num=50000):
    best_loss = float('inf')
    best_operations = None
    best_params = None
    loss_history= []
    a, b, c = np.random.uniform(-10, 10, 3)  

    for i in range(iteration_num):
        # a, b, c = np.random.uniform(-10, 10, 3)  # 每次迭代中随机选择a, b, c的值
        param_to_change = np.random.choice(['a', 'b', 'c'])
        new_value = float(np.random.choice(np.arange(-10, 10.1, 0.1)))
        if param_to_change == 'a':
            a = new_value
        elif param_to_change == 'b':
            b = new_value
        else:
            c = new_value
        operations = [random.randint(0, 5) for _ in range(3)]
        y_pred = custom_function(a, b, c, x_data, operations)
        current_loss = loss_function(y_true, y_pred)       

        if current_loss < best_loss:
            best_loss = current_loss
            best_operations = operations
            best_params = (a, b, c)
        loss_history.append(best_loss)

        if i % 500 == 0:
            print("Iteration: {}, Loss: {}".format(i, best_loss))

    print("Best operations:", best_operations)
    print("Best parameters: a={}, b={}, c={}".format(*best_params))
    print("Minimum loss:", best_loss)
    return best_operations, best_params,y_pred ,loss_history

best_operation,best_params,y_pred,loss_history = random_optimize(x_data, y_data)

# save file
str_avg_fitness_list = [str(f) + '\n' for f in loss_history]
with open('Random_long_lengths.txt', 'w') as f:
    f.writelines(str_avg_fitness_list)

plt.figure(figsize=(10, 6))
plt.plot(loss_history)
plt.title("Loss Over Iterations")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

plt.figure(1)
plt.plot(x_data, y_data, 'b.')
plt.plot(x_data, y_pred, 'r-')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.title('brone Data')
plt.grid(True)
plt.show()
# plt.figure(2)
# # plt.plot(x_data, y_data, 'b.')
# plt.plot(x_data, y_pred, 'r-')
# plt.xlabel('X values')
# plt.ylabel('Y values')
# plt.title('Data from txt file')
# plt.grid(True)
# plt.show()