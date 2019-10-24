# import the necessary packages
from helpers import sliding_window, pyramid
import new_filters as nf
import NumPyCNN as numpycnn
import argparse, pickle, time, cv2
import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_der(x):
    return sigmoid(x)*(1-sigmoid(x))

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

weights = pickle.load(open("syn0.pickle", "rb"))
weights2 = pickle.load(open("syn1.pickle", "rb"))
bias = pickle.load(open("bias.pickle", "rb"))
bias2 = pickle.load(open("bias2.pickle", "rb"))
start = time.time()

# load the image and define the window width and height
image = cv2.imread(args["image"])
image = cv2.resize(image,(400,300))
(winW, winH) = (124, 144)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# calculate the magnitude and phase of original image
inputs = ( (image - np.amin(image) ) * 1 ) / ( np.amax(image) - np.amin(image) )
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
print(magnitude.shape, phase.shape)
exit()

# loop over the image pyramid
for resized in pyramid(image, scale=1.2):
	# loop over the sliding window for each layer of the pyramid
    for (x, y, window) in sliding_window(resized, stepSize=16, windowSize=(winW, winH)):
        # if the window does not meet our desired window size, ignore it
        if window.shape[0] != winH or window.shape[1] != winW:
            continue

        ravel_input = []
        for in1, conv1 in enumerate(magnitude):
            # print(np.amax(conv1), np.amin(conv1))
            # cv2.imwrite('magnitude'+str(in1)+'.jpg', conv1 * 255)
            ravel_input.append(conv1)

        for in1, conv1 in enumerate(phase):
            # print(np.amax(conv1), np.amin(conv1))
            # cv2.imwrite('phase'+str(in1)+'.jpg', conv1 * 255)
            ravel_input.append(conv1)

        ravel_input = np.array([np.array(ravel_input).ravel()])
        print(ravel_input.shape)
        # print(np.amin(ravel_input), np.amax(ravel_input), np.mean(ravel_input))

        # feedforward step1
        l1 = sigmoid(np.dot(ravel_input, weights) + bias)
        z = sigmoid(np.dot(l1, weights2) + bias2)
        print(z)


        # THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
        # MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
        # WINDOW
        if(np.argmax(z) == 0):
            clone = resized.copy()
            cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
        # since we do not have a classifier, we'll just draw the window
            cv2.imshow("Window", clone)
            cv2.waitKey(1)
	# time.sleep(0.0025)
end = time.time()
print("finish", end - start)