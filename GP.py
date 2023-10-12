import numpy as np
import random
import math
import matplotlib.pyplot as plt

# 定义初始函数和终结符集合
FUNCTIONS = ['+', '-', '*', '/', 'sin', 'cos']
TERMINALS = list(range(1, 10)) + ['x']

# 随机生成函数树（数组表示）
def generate_tree(depth):
    if depth == 0:
        return [random.choice(TERMINALS)]
    else:
        func = random.choice(FUNCTIONS)
        left_child = generate_tree(depth-1)
        if func in ['sin', 'cos']:
            return [func] + left_child
        else:
            right_child = generate_tree(depth-1)
            return [func] + left_child + right_child

# 计算函数树的值
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

# 交叉操作
def crossover(parent1, parent2):
    crossover_point = random.randrange(1, min(len(parent1), len(parent2)))
    
    # 计算crossover_point之后的所有相关索引
    indices_to_swap = [crossover_point]
    i = 0
    while i < len(indices_to_swap):
        idx = indices_to_swap[i]
        if 2*idx < len(parent1):
            indices_to_swap.append(2*idx)
        if 2*idx + 1 < len(parent1):
            indices_to_swap.append(2*idx + 1)
        i += 1

    # 创建新的子代
    child1 = parent1.copy()
    child2 = parent2.copy()
    
    for idx in indices_to_swap:
        if idx < len(child1) and idx < len(child2):
            child1[idx], child2[idx] = child2[idx], child1[idx]

    return child1, child2

# 主GP流程
def tournament_selection(population, k=6):
    # 从种群中随机选择k个个体
    selected = random.sample(population, k)
    # 按照适应度评分对这k个个体进行排序
    selected.sort(key=lambda x: x[1])  # 假设低得分是更好的
    # 返回最佳的两个个体
    return selected[0][0], selected[1][0]

def mutate(tree, mutation_rate=0.05):
    # 遍历树的每个节点
    for i in range(len(tree)):
        if random.random() < mutation_rate:
            if tree[i] in FUNCTIONS:
                tree[i] = random.choice(FUNCTIONS)
            else:
                tree[i] = random.choice(TERMINALS)
    return tree

MAX_TREE_SIZE = 50  # 设定一个最大的树大小

def tree_size(tree):
    return len(tree)

def snip(tree, x_data, y_data):
    # 找出需要被替换的子树的索引
    index_to_snip = random.randrange(1, len(tree))
    
    # 获取该子树计算的输出
    sub_tree_output = [compute_tree(tree[index_to_snip:], x) for x in x_data]
    
    # 计算该子树的平均输出值
    average_output = sum(sub_tree_output) / len(sub_tree_output)
    
    # 替换子树为该平均值
    tree[index_to_snip:] = [str(average_output)]
    
    return tree

ELITE_SIZE = 10  # 例如，我们可以保存每一代的前10个精英个体

def genetic_programming(x_data, y_data, generations=3, population_size=1000, depth=3):
    population = [generate_tree(depth) for _ in range(population_size)]
    
    best_loss_per_generation = []  # 用于保存每一代的最佳loss
    best_tree_per_generation = []  # 用于保存每一代的最佳函数方程
    for generation in range(generations):
        scores = []
        for individual in population:
            y_pred = [compute_tree(individual, x) for x in x_data]
            score = loss_function(y_data, y_pred)
            scores.append((individual, score))
        scores.sort(key=lambda x: x[1])

        best_loss_per_generation.append(scores[0][1])
        best_tree_per_generation.append(scores[0][0])

        print(f"Generation {generation+1} Best Score: {scores[0][1]}")
        
        # 保存精英个体
        elites = [individual[0] for individual in scores[:ELITE_SIZE]]
        
        new_population = elites  # 将精英个体加入到新种群中
        while len(new_population) < population_size:
            parent1, parent2 = tournament_selection(scores)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1)
            child2 = mutate(child2)
            new_population.extend([child1, child2])
        
        population = new_population

    with open("GA.txt", "w") as file:
        for gen, (tree, loss) in enumerate(zip(best_tree_per_generation, best_loss_per_generation)):
            file.write(f"Generation {gen+1} Best Tree: {' '.join(map(str, tree))} Loss: {loss}\n")

    plt.plot(best_loss_per_generation)
    plt.xlabel("Generation")
    plt.ylabel("Best Loss")
    plt.title("Loss vs. Generation")
    plt.savefig("loss_vs_generation.png")
    plt.show()

    # 绘制最后的函数方程曲线
    best_tree = best_tree_per_generation[-1]
    y_pred = [compute_tree(best_tree, x) for x in x_data]
    plt.plot(x_data, y_data, label="True Data")
    plt.plot(x_data, y_pred, label="Predicted Data", linestyle="--")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.title("True vs. Predicted Data")
    plt.savefig("true_vs_predicted.png")
    plt.show()

    return scores[0][0]

# 加载数据和损失函数
# data = np.loadtxt('data\Bronze.txt', delimiter=',')
# x_data = data[:, 0]
# y_data = data[:, 1]
# def loss_function(y_true, y_pred):
#     return np.sum(np.abs(y_true - y_pred))

# # 运行GP
# best_tree = genetic_programming(x_data, y_data)

