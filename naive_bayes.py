
import numpy
import os
import PIL
import sys
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import warnings
warnings.filterwarnings("ignore")

max_number_of_labels = 8
mean_features = numpy.zeros((32, max_number_of_labels))
stddev_features = numpy.zeros((32, max_number_of_labels))
imap = dict()

def splitline(line):
    imgName, label = line.split(' ',1)
    return imgName, label

def rgb2gray(rgb):
    return numpy.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def gethigh(values, vectors, index):
    values = values[index]
    vectors = vectors[:,index]
    imp_vectors = vectors[:,0:32]
    return numpy.transpose(imp_vectors)

def pca_test(trainfile, testfile):
    global imap
    count = 0
    train_labels = []
    test_labels = []
    X_train = []
    counter = 0
    map1=dict()
    X_test = []
    f = open(trainfile, 'r')
    g = open(testfile, 'r')
    for line in f:
        imgName, label = splitline(line)
        if label not in map1:
            map1[label]=counter
            imap[counter]=label.rstrip()
            counter+=1
        img = numpy.array(Image.open(imgName).convert('L').resize((32,32)), dtype=numpy.uint8)
        img = img.flatten()
        X_train.append(img)
        train_labels.append(map1[label])
        # train_labels.append(label.rstrip())
        count = count + 1
    for line in g:
        test_labels.append(line[12])
        linesp=line.rstrip()
        img = numpy.array(Image.open(linesp).convert('L').resize((32,32)), dtype=numpy.uint8)
        X_test.append(img.flatten())
        
    X_train = numpy.array(X_train)              
    X_test = numpy.array(X_test)                    
    meanX_train = numpy.mean(X_train, axis = 0)
    XT_train = numpy.transpose(X_train-meanX_train)        
    XT_test = numpy.transpose(X_test-meanX_train)
    values = numpy.linalg.eig(numpy.cov(XT_train))[0]
    vectors = numpy.linalg.eig(numpy.cov(XT_train))[1]
    idx = (-values).argsort()
    imp_vectors_T = gethigh(values, vectors, idx)
    Z_final_train = numpy.transpose(numpy.dot(imp_vectors_T, XT_train))
    Z_final_test = numpy.transpose(numpy.dot(imp_vectors_T, XT_test))
    return Z_final_train, train_labels, Z_final_test, test_labels


X_train, train_labels, X_test, test_labels = pca_test(sys.argv[1], sys.argv[2])


train_labels = [int(i) for i in train_labels]
test_labels = [int(i) for i in test_labels]
all_labels = []
sizelabels=numpy.shape(train_labels)[0]
for i in range(sizelabels):
    if train_labels[i] not in all_labels:
        all_labels.append(train_labels[i])
number_of_labels = len(all_labels)
# count_ofeach_label = [[0 for y in range(1)] for x in range(max_number_of_labels)]
count_ofeach_label = numpy.zeros((max_number_of_labels,1))
p=0
sizelabels=numpy.shape(train_labels)[0]
while p<sizelabels:
    count_ofeach_label[train_labels[p]]+=1
    p+=1
# print(count_ofeach_label)
prior_probabilities = numpy.zeros((max_number_of_labels,1))
p=0
while p<max_number_of_labels:
    prior_probabilities[p] = count_ofeach_label[p]/numpy.shape(train_labels)[0]
    p+=1
p=0
# print(prior_probabilities)



for label in all_labels:
    l=numpy.shape(train_labels)[0]
    i=0
    all_with_current_label = []
    while i<l:
        if(train_labels[i] == label):
            all_with_current_label.append(X_train[i])
        i+=1
    all_with_current_label = numpy.array(all_with_current_label)
    i=0
    while i<32:
        mean_features[i, label] = numpy.mean(all_with_current_label[:,i])
        stddev_features[i, label] = numpy.std(all_with_current_label[:,i])
        i+=1
    i=0


predictions = []
count = 0;

def gauss_value(count, x, meanval, stddev):
    sq = stddev*stddev
    degree=2*numpy.pi
    sqrt=(numpy.sqrt(degree*sq))
    softpowe=((x - meanval)/stddev)
    return numpy.log((1.0/sqrt)*numpy.exp(-1.0*(softpowe**2)))


def predict(feature_vector):
    prob_max = -100000000
    curr_prob = -10000000
    feature_prob = numpy.zeros((32, max_number_of_labels))
    for label in all_labels:
        i=0
        while i<32:
            feature_prob[i, label] = gauss_value(i, feature_vector[i], mean_features[i, label], stddev_features[i, label])
            i=i+1
        i=0
    likelihood = numpy.zeros((max_number_of_labels, 1))
    for label in all_labels:
        i=0
        while i<32:
            if(feature_prob[i,label] is 0):
                continue
            else:
                likelihood[label] += feature_prob[i, label]
            i=i+1
        i=0
    ret = -1
    for i in range(max_number_of_labels):
        prob = prior_probabilities[i]
        if prob<=0:
            continue
        else:
            if(prob + likelihood[i] >= prob_max):
                ret = i
                prob_max = prob + likelihood[i]
            curr_prob = prob + likelihood[i]
    return ret


for i in range(numpy.shape(X_test)[0]):
    prediction = predict(X_test[i, :])
    predictions.append(prediction)
    pred = predictions[i]
    lab = test_labels[i]
    if pred==lab:
        count += 1
    print(imap[predictions[i]])
# print(count/numpy.shape(X_test)[0])

    

