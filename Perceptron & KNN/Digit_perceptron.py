# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 19:55:23 2021

@author: Madhav
"""
import numpy as np
import random
def feature_extract(image, square_size = 7):   
    features = [1] #Bias
                        
    for i in range(0,len(image),square_size): 
        for t in range(0,len(image),square_size):
            val = 0
            for j in range(i, i + square_size):
                for k in range(t, t + square_size):
                    if(image[j][k] != " "):
                        #print(image[j][k])
                        val += 1
                        
            features.append(val)          
                
    return features

class Digit_Perceptron:
    
    feature_matrix = []
    labels = []
    weights = []
    
    def __init__(self, features, labels, num_of_features):
        
        #self.features = features
        self.feature_matrix = features
        self.labels = labels
        self.weights = []
        
        weights_0 = np.random.rand(num_of_features + 1)
        weights_1 = np.random.rand(num_of_features + 1)
        weights_2 = np.random.rand(num_of_features + 1)
        weights_3 = np.random.rand(num_of_features + 1)
        weights_4 = np.random.rand(num_of_features + 1)
        weights_5 = np.random.rand(num_of_features + 1)
        weights_6 = np.random.rand(num_of_features + 1)
        weights_7 = np.random.rand(num_of_features + 1)
        weights_8 = np.random.rand(num_of_features + 1)
        weights_9 = np.random.rand(num_of_features + 1)
    
        self.weights = np.array([weights_0,weights_1,weights_2,weights_3,weights_4,weights_5,weights_6,
                              weights_7,weights_8,weights_9])
        
    
    def train(self):
        while True:        
             num_of_mistakes = 0
             for k in range(len(self.feature_matrix)):
            
                label = int(self.labels[k])
                values = np.dot(self.weights,np.array(self.feature_matrix[k]).T)
                index_of_max = int(np.where(values == values.max())[0][0])
                
                
                if(index_of_max != label):
                                    
                    self.weights[index_of_max] = self.weights[index_of_max] -  .01 * np.array(self.feature_matrix[k])
                    
                    self.weights[label] = self.weights[label] + .01 *np.array(self.feature_matrix[k])
                    num_of_mistakes += 1  
                
                #print(set_of_weights[0])
             if(num_of_mistakes == 0): break
         
    def predict(self,image_features):
         values = np.dot(self.weights,image_features)
         index_of_max = int(np.where(values == values.max())[0][0])
         return index_of_max
     
        
     
        
if __name__ == "__main__":
    
    X = [] # samples
    Y = [] #Labels
    X_test = [] # test images
    num_of_samples = 5000
    size_of_square = 1
    image_length = 28 
    num_of_features = int((image_length/size_of_square) ** 2)
    training_images =  open("trainingimages.txt")
    training_labels = open("trainingimageslabels.txt")
    test_images = open("testimages.txt")
    test_labels = open("testlabels.txt")
    num_of_features = int((image_length/size_of_square) ** 2)
    Y = [line[:-1] if line[-1] == '\n' else line for line in training_labels][0:num_of_samples]
    Y_test = [line[:-1] if line[-1] == '\n' else line for line in test_labels]
    
    
    for i in range(0, num_of_samples * image_length, image_length):  
        array = []
        for j in range(i, i + image_length):
            line = training_images.readline()[0:image_length]
            array.append(list(line))
            #print(line)
        features = feature_extract(array, square_size = size_of_square)
        X.append(features)
        
    for i in range(0, 1000 * image_length, image_length):  
        array = []
        for j in range(i, i + image_length):
            line = test_images.readline()[0:image_length]
            array.append(list(line))
            #print(line)
        feature = feature_extract(array, square_size = size_of_square)
        X_test.append(feature)
        
    perceptron_1 = Digit_Perceptron(X,Y,num_of_features)
    
    perceptron_1.train()
    
    
    ### Testing perceptron 1
    num_right = 0
    for k in range(len(X_test)):
        
        prediction = perceptron_1.predict(np.array(X_test[k]).T)
        if(prediction == int(Y_test[k])): num_right += 1
    percentage_right = str((num_right/1000) * 100)
    print("The perceptron classified " + percentage_right + " percent of the test images correctly" )    
       
       