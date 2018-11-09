from PIL import Image
import numpy
import math
import matplotlib
import matplotlib.pyplot
import sys

def PCA(a):
    a = numpy.array(a)
    eigenvectors = numpy.linalg.svd(a - numpy.mean(a,axis=0))[2]
    coefficients = numpy.matmul(a - numpy.mean(a,axis=0) , eigenvectors.T) 
    arr1 = numpy.array(coefficients[:,:32])
    b = numpy.full((len(arr1),1), 1)
    arr1 = numpy.hstack((arr1,b))
    return numpy.array(arr1) , eigenvectors[0:32,:]

def predict(W,X):
    pred_ys = numpy.full((1,X.shape[1]), 0)
    pred_ys = numpy.argmax(W.dot(X), axis=0)
    return pred_ys



def gradient(W, X, y):
    dim = X.shape[0]
    num_train = X.shape[1]
    grad = numpy.zeros((W.shape[0],W.shape[1]))
    scores_exp_normalized = numpy.exp(W.dot(X)-numpy.max(W.dot(X))) / numpy.sum(numpy.exp(W.dot(X)-numpy.max(W.dot(X))), axis=0)
    scores_exp_normalized[y, range(num_train)] -= 1
    return (scores_exp_normalized.dot(X.T))/num_train


def main():
    a = []
    x_file = open(sys.argv[1], "r")
    name_classes = []
    a_name = []
    for file in x_file:
        sp = file.split() 
        a.append(numpy.array(Image.open(sp[0]).convert('L').resize((32,32)), dtype=numpy.uint8).flatten())
        sp[1]=sp[1].rstrip()
        a_name.append(str(sp[1]))
        name_classes.append(str(sp[1]))

    
    

    new_img = PCA(a)[0]
    eigenvectors = PCA(a)[1]
    a_name = numpy.array(a_name)

    unique_classes = list(set(a_name))
    assign_int = {}
    assign_name = {}
    p = len(unique_classes)
    i = 0
    while i < p:
        assign_int[i] = unique_classes[i]
        assign_name[unique_classes[i]] = i
        i += 1
    p = len(a_name)
    i = 0
    while i < p:
        a_name[i] = assign_name[a_name[i]]
        i += 1
    i = 0
    a_name = a_name.astype(int)

    num_classes = len(set(a_name))
    X = numpy.array(new_img.T)
    W = numpy.random.randn(num_classes, len(X)) * 0.001
    lr = 1e-4
    num_iters=5000
    Y = numpy.array(a_name)
    i = 0
    while i < num_iters:
        grad =gradient(W, X, Y)
        W -= lr * grad
        i += 1

    x_files = open(sys.argv[2], "r")
    res = []
    file_name = []
    for file in x_files:
        file = file.split('\n')
        res.append(numpy.array(Image.open(str(file[0])).convert("L").resize((32,32)), dtype=numpy.uint8).flatten())
        file_name.append(str(file[0]))

    res -= numpy.mean(a,axis=0)
    coefficiets = numpy.matmul(res , eigenvectors.T)
    b = numpy.full((len(res),1), 1)
    res = numpy.hstack((coefficiets,b))
    pred = predict(W,numpy.array(res).T)
    acc = len(pred)
    i = 0
    while i < acc:
        print(assign_int[pred[i]])
        i+=1;
    i = 0

if __name__ == "__main__":
    main()