import NumPyCNN as numpycnn
import numpy as np
import pandas as pd
import time, pickle, function, read_data, cv2
import new_filters as nf

def tanh(x):
    return ( 2 / 1 + np.exp(-2*x)) - 1

def tanh_der(x):
    return 1 - tanh(x) * tanh(x)

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

np.random.seed(1)
weights = 2 * np.random.rand(20608,500) - 1
weights2 = 2 * np.random.rand(500,2) - 1

bias = 2 * np.random.rand(1,500) - 1
bias2 = 2 * np.random.rand(1,2) - 1
lr = 0.01

weights = pickle.load(open("syn0.pickle", "rb"))
weights2 = pickle.load(open("syn1.pickle", "rb"))
bias = pickle.load(open("bias.pickle", "rb"))
bias2 = pickle.load(open("bias2.pickle", "rb"))

epoch = 10 * len(feature_set)
# epoch = 100
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
        print(np.amax(conv1), np.amin(conv1), conv1.shape)
        # cv2.imwrite('magnitude'+str(in1)+'.jpg', conv1 * 255)
        ravel_input.append(conv1)

    for in1, conv1 in enumerate(phase):
        print(np.amax(conv1), np.amin(conv1), conv1.shape)
        # cv2.imwrite('phase'+str(in1)+'.jpg', conv1 * 255)
        ravel_input.append(conv1)

    ravel_input = np.array([np.array(ravel_input).ravel()])
    print(np.amin(ravel_input), np.amax(ravel_input), np.mean(ravel_input))

    # feedforward step1
    l1 = sigmoid(np.dot(ravel_input, weights) + bias)
    z = sigmoid(np.dot(l1, weights2) + bias2)
    
    # backpropagation step 1
    error = z - np.array([labels[ri]])
    print(ri, labels[ri], z)
    # backpropagation step 2
    dcost_dpred = error
    dpred_dz = sigmoid_der(z)
    z_delta = dcost_dpred * dpred_dz

    l1_error = z_delta.dot(weights2.T)
    dpred_dl1 = sigmoid_der(l1_error)
    l1_delta = l1_error * dpred_dl1

    l1 = l1.T
    weights2 -= lr * np.dot(l1,z_delta)
    ravel_input = ravel_input.T
    # print(z_delta.shape, inputs.shape)
    weights -= lr * np.dot(ravel_input, l1_delta)

    for num in z_delta:
        bias2 -= lr * num

    for num in l1_delta:
        bias -= lr * num

    if(j % 1 == 0):
        current = time.time()
        print(round((current - start),1),'s',round((j/epoch*100),2),'%', error.sum(), l1_error.sum())

# save final synapse into pickle
pickle_out = open("syn0.pickle", "wb")
pickle.dump(weights, pickle_out)

pickle_out = open("syn1.pickle", "wb")
pickle.dump(weights2, pickle_out)

pickle_out = open("bias.pickle", "wb")
pickle.dump(bias, pickle_out)

pickle_out = open("bias2.pickle", "wb")
pickle.dump(bias2, pickle_out)

pickle_out.close()

end = time.time()
print(end - start)
exit()
single_point = np.array([1,1,1])
result = sigmoid(np.dot(single_point, weights) + bias)
z = sigmoid(np.dot(result, weights2) + bias2)
print(z)