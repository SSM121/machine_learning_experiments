# the functions that predict the gender base off of weight
# the first function changes the stderror file. will look into
# better logging library (or make my own) in the future.
# The second function runs a basic small test on a small data set.
# more of a proof of concept.
# the third function is the big boy that runs the test on the big csv data. returns the lists of prediction
# gaussian test commented out in the file function due to really slow performance (on the order of 10 mins
# on my computer for a single iteration).it worked but I did not feel that I wanted to run it over and over again.

from sklearn import tree, neural_network, gaussian_process
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import accuracy_score
from sklearn import neighbors
from sklearn.model_selection import train_test_split
import sys
import config as c


# simple function ro change the error file
def change_err():
    # redirect stderr to a file
    sys.stderr = open(c.error, "w")


def run_test():
    print("Test Values:", file=sys.stderr)
    # classifier constructors
    t = tree.DecisionTreeClassifier()
    n = neural_network.MLPClassifier()
    g = gaussian_process.GaussianProcessClassifier(1.0 * RBF(1.0))
    k = neighbors.KNeighborsClassifier(3)
    # [height, weight, shoesize]
    x = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
         [190, 90, 47], [175, 64, 39],
         [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]
    y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
         'female', 'male', 'male']

    t = t.fit(x, y)
    n = n.fit(x, y)
    g = g.fit(x, y)
    k = k.fit(x, y)

    # should be male according to t
    t_prediction = t.predict([[190, 70, 43]])
    n_prediction = n.predict([[190, 70, 43]])
    g_prediction = g.predict([[190, 70, 43]])
    k_prediction = k.predict([[190, 70, 43]])

    n_accurate = accuracy_score(t_prediction, n_prediction)
    g_accurate = accuracy_score(t_prediction, g_prediction)
    k_accurate = accuracy_score(t_prediction, k_prediction)

    print(f"The control test is a tree. its value is {t_prediction}", file=sys.stderr)
    print(f"The neural network predicted: {n_prediction} which has an accuracy of: {n_accurate}", file=sys.stderr)
    print(f"The gaussian process predicted: {g_prediction} which has an accuracy of: {g_accurate}", file=sys.stderr)
    print(f"The k neighbors predicted: {k_prediction} which has an accuracy of: {k_accurate}", file=sys.stderr)


def run_file(iterations=c.iterations_a):
    # used over the 10000 values in weight-height.csv This csv was taken from
    # https://github.com/omairaasim/machine_learning/tree/master/project_9_predict_weight_sex
    print("Values from CSV:", file=sys.stderr)
    # classifier constructors
    t = tree.DecisionTreeClassifier()
    n = neural_network.MLPClassifier()
    g = gaussian_process.GaussianProcessClassifier(1.0 * RBF(1.0), max_iter_predict=1, n_jobs=-1)
    k = neighbors.KNeighborsClassifier(3)
    # no shoesize here because this test csv file did not contain it
    # [height, weight]
    x = []
    y = []

    # read in the data (I know there is a csv library, didn't feel like using it)
    f = open(c.fileName)
    # skip the first line and read in the csv file stripping the punctuation
    f.readline()
    for line in f:
        i = line.split(',')
        y.append(i[0].strip('"'))
        x.append([float(i[1]), float(i[2])])

    # lists of accuracies
    t_list = []
    n_list = []
    g_list = []
    k_list = []
    for j in range(iterations):
        print(f"running iteration: {j +  1}", file=sys.stderr)
        # split the data into test and train data. 20 percent. this line came from the guy mentioned above for the csv
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=c.tested, random_state=0)
        print("fitting t", file=sys.stderr)
        t = t.fit(x_train, y_train)
        print("fitting n", file=sys.stderr)
        n = n.fit(x_train, y_train)
        '''print("fitting g", file=sys.stderr)
         g = g.fit(x_train, y_train)'''
        print("fitting k", file=sys.stderr)
        k = k.fit(x_train, y_train)

        # calculate prediction
        print("Calc t prediction", file=sys.stderr)
        t_prediction = t.predict(x_test)
        print("Calc n prediction", file=sys.stderr)
        n_prediction = n.predict(x_test)
        '''print("Calc g prediction", file=sys.stderr)
         gPrediction = g.predict(x_test)'''
        print("Calc k prediction", file=sys.stderr)
        k_prediction = k.predict(x_test)

        # find accuracy score
        t_accurate = accuracy_score(y_test, t_prediction)
        n_accurate = accuracy_score(y_test, n_prediction)
        # gAccurate = accuracy_score(y_test, gPrediction)
        k_accurate = accuracy_score(y_test, k_prediction)

        # append accuracy scores to list
        t_list.append(t_accurate)
        n_list.append(n_accurate)
        # g_list.append(gAccurate)
        k_list.append(k_accurate)

        # print out the accuracy
        print(f"The decision tree predicted with an of: {t_accurate}", file=sys.stderr)
        print(f"The neural network predicted with an of: {n_accurate}", file=sys.stderr)
        # print(f"The gaussian process predicted with an of: {gAccurate}")
        print(f"The k neighbor predicted with an of: {k_accurate}", file=sys.stderr)
    # only have one of the 2 return statements uncommented. determined by your choice on using gaussian
    return t_list, n_list, k_list
    # return t_list, n_list, g_list, k_list


if __name__ == '__main__':
    if c.err:
        change_err()
    run_test()
    run_file()
