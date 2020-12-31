import weightpredict as weight
import matplotlib.pyplot as plt
import numpy as np
import config as c


def main():
    y = []
    if c.err:
        weight.change_err()
    t_list, n_list, k_list = weight.run_file(iterations=c.iterations_b)
    for i in range(len(t_list)):
        y.append(i + 1)
    # plot the bad boys
    t_fig, t_axis = plt.subplots()
    t_axis.plot(y, t_list)
    t_axis.set_xlabel("Iteration number")
    t_axis.set_ylabel("Accuracy score")
    t_axis.set_title("Decision Tree")

    n_fig, n_axis = plt.subplots()
    n_axis.plot(y, n_list)
    n_axis.set_xlabel("Iteration number")
    n_axis.set_ylabel("Accuracy score")
    n_axis.set_title("Neural Network")

    k_fig, k_axis = plt.subplots()
    k_axis.plot(y, k_list)
    k_axis.set_xlabel("Iteration number")
    k_axis.set_ylabel("Accuracy score")
    k_axis.set_title("K Neighbor")

    # do some calculations
    t_mean, t_median, t_mode, t_range = list_analysis(t_list)
    n_mean, n_median, n_mode, n_range = list_analysis(n_list)
    k_mean, k_median, k_mode, k_range = list_analysis(k_list)

    # create "pretty" strings to output to a file
    t_pretty = pretty_data("Decision Tree", t_mean, t_median, t_mode, t_range)
    n_pretty = pretty_data("Neural Network", n_mean, n_median, n_mode, n_range)
    k_pretty = pretty_data("K Neighbor", k_mean, k_median, k_mode, k_range)

    if c.pic is False:
        plt.show()
    else:
        t_fig.savefig('out/Decision_Tree.png')
        n_fig.savefig('out/Neural_Network.png')
        k_fig.savefig('out/K_Neighbor.png')
    if c.stats:
        t_stats = open("out/Decision_Tree_Stats.txt", "w")
        print(t_pretty, file=t_stats)
        n_stats = open("out/Neural_Network_Stats.txt", "w")
        print(n_pretty, file=n_stats)
        k_stats = open("out/K_Neighbor_Stats.txt", "w")
        print(k_pretty, file=k_stats)
    else:
        print(t_pretty)
        print("\n")
        print(n_pretty)
        print("\n")
        print(k_pretty)


def pretty_data(name, mean, median, mode, ran):
    return f'''
Model : {name}
===============
mean = {mean}
median = {median}
mode = {mode}
Lower Bound = {ran[0]}
Upper Bound = {ran[1]}
Range = {ran[1] - ran[0]}
'''


def list_analysis(list):
    num_value = len(list)
    # mean calculation
    mean = sum(list) / num_value
    # median calculation
    list_sort = list.copy()
    list_sort.sort()
    if num_value % 2 == 0:
        median1 = list_sort[num_value // 2]
        median2 = list_sort[num_value // 2 - 1]
        median = (median1 + median2) / 2
    else:
        median = list_sort[num_value // 2]
    # mode calculation. sets the key to be based off the count in the list
    mode = max(list, key=list.count)
    # range calculation
    ran = [min(list), max(list)]
    return mean, median, mode, ran


if __name__ == '__main__':
    main()
