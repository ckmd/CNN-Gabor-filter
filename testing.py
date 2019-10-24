import NumPyCNN as numpycnn
import numpy as np
import pandas as pd
import time, pickle, function, read_data, cv2
import new_filters as nf

def tanh(x):
    return ( 2 / 1 + np.exp(-2*x)) - 1

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_der(x):
    return sigmoid(x)*(1-sigmoid(x))

# feature_set = np.array([[0,1,0],[0,0,1],[1,0,0],[1,1,0],[1,1,1]])
# labels = np.array([[1,0,0,1,1]])
# labels = labels.reshape(5,1)

start = time.time()

data = read_data.data
labels = read_data.label
feature_set = data

# print(feature_set.shape, labels.shape, feature_set[0])
# exit()

np.random.seed(42)

weights = pickle.load(open("syn0.pickle", "rb"))
weights2 = pickle.load(open("syn1.pickle", "rb"))
bias = pickle.load(open("bias.pickle", "rb"))
bias2 = pickle.load(open("bias2.pickle", "rb"))

epoch = 1 * len(feature_set)
# epoch = 100
cocok = 0
for j in range(epoch):
    ravel_input = []
    ri = np.random.randint(len(feature_set))
    inputs = feature_set[ri]
    inputs = ( (inputs - np.amin(inputs) ) * 1 ) / ( np.amax(inputs) - np.amin(inputs) )

    # print(feature_set.shape, np.array([feature_set[1]]).shape)
    
    # convolutional 1
    l1_feature_map = numpycnn.conv(inputs, nf.filter1)
    l1_feature_map_relu_pool = numpycnn.pooling(l1_feature_map, 2, 2)

    l1_feature_map_i = numpycnn.conv(inputs, nf.filter1_i)
    l1_feature_map_relu_pool_i = numpycnn.pooling(l1_feature_map_i, 2, 2)

    magnitude = np.sqrt((l1_feature_map_relu_pool.T ** 2) + (l1_feature_map_relu_pool_i.T ** 2))
    phase = np.arctan(l1_feature_map_relu_pool_i.T / l1_feature_map_relu_pool.T)

    # Normalize 0 to 1
    magnitude = ( (magnitude - np.amin(magnitude) ) * 1 ) / ( np.amax(magnitude) - np.amin(magnitude) )
    # magnitude -= 1
    phase = ( (phase - np.amin(phase) ) * 1 ) / ( np.amax(phase) - np.amin(phase) )
    # phase -= 1

    for in1, conv1 in enumerate(magnitude):
        # print(np.amax(conv1), np.amin(conv1))
        # cv2.imwrite('magnitude'+str(in1)+'.jpg', conv1 * 255)
        ravel_input.append(conv1)

    for in1, conv1 in enumerate(phase):
        # print(np.amax(conv1), np.amin(conv1))
        # cv2.imwrite('phase'+str(in1)+'.jpg', conv1 * 255)
        ravel_input.append(conv1)

    ravel_input = np.array([np.array(ravel_input).ravel()])
    # print(np.amin(ravel_input), np.amax(ravel_input), np.mean(ravel_input))

    # feedforward step1
    l1 = sigmoid(np.dot(ravel_input, weights) + bias)
    z = sigmoid(np.dot(l1, weights2) + bias2)
    print(ri, z, np.argmax(z), labels[ri])
    if(np.argmax(z) == np.argmax(labels[ri])):
        cocok += 1

    print("Accuraccy : ",cocok/(j+1)*100,'%')
end = time.time()
print("execution time ", end - start)