# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 15:52:07 2021

@author: Madhav
"""
import numpy as np
import random
def feature_extract(image, square_size = 7):   
    features = [1]
                        
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


def perceptron(features, weights):
    
    output = 0
    for f,w in zip(features, weights):
        output += f*w
        
    if output > 0:
        return True
    
    else:
        return False
    
    
if __name__ == "__main__":
    
    X = [] # samples
    Y = [] #Labels
    X_test = [] # test images
    size_of_square = 2
    training_images =  open("trainingimages.txt")
    training_labels = open("trainingimageslabels.txt")
    test_images = open("testimages.txt")
    test_labels = open("testlabels.txt")
    num_of_samples = 1500
    Y = [line[:-1] if line[-1] == '\n' else line for line in training_labels][0:num_of_samples]
    Y_test = [line[:-1] if line[-1] == '\n' else line for line in test_labels]
    image_length = 28 
    num_of_features = int((image_length/size_of_square) ** 2) 
    
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
    
    set_of_weights = np.array([weights_0,weights_1,weights_2,weights_3,weights_4,weights_5,weights_6,
                              weights_7,weights_8,weights_9])
    
    
   
    for i in range(0, num_of_samples * image_length, image_length):  
        array = []
        for j in range(i, i + image_length):
            line = training_images.readline()[0:image_length]
            array.append(list(line))
            #print(line)
        features = feature_extract(array, square_size = size_of_square)
        X.append(features)
        
    
    
    for j in range(500):
        
        num_of_mistakes = 0
        for k in range(len(X)):
            
            label = int(Y[k])
            values = np.dot(set_of_weights,np.array(X[k]).T)
            index_of_max = int(np.where(values == values.max())[0][0])
            
            
            if(index_of_max != label):
                                
                set_of_weights[index_of_max] = set_of_weights[index_of_max] -  np.array(X[k])
                
                set_of_weights[label] = set_of_weights[label] + np.array(X[k])
                num_of_mistakes += 1
                
                #print(set_of_weights[0])
        if(num_of_mistakes == 0): break
        #print("next iteration proceeds") 
        #print("\n")
        #print(num_of_mistakes)
            
   
            
   ### Now we will start predicting!
   
   
    for i in range(0, 1000 * image_length, image_length):  
        array = []
        for j in range(i, i + image_length):
            line = test_images.readline()[0:image_length]
            array.append(list(line))
            #print(line)
        feature = feature_extract(array, square_size = size_of_square)
        X_test.append(feature)
   
    
    num_right = 0
    for k in range(len(X_test)):
            
        values = np.dot(set_of_weights,np.array(X_test[k]).T)
        index_of_max = int(np.where(values == values.max())[0][0])
        #print(index_of_max)
        if(index_of_max == int(Y_test[k])): num_right += 1
        
    percentage_right = str((num_right/1000) * 100)
    
    print("The perceptron classified " + percentage_right + " percent of the test images correctly" )
        
        #print(index_of_max)
    #print(np.array(X[0]).T)
    #print(weights_0)
    #print(array[0])
    #print(set_of_weights)
    #X.append(feature_extract(array))
    
    
    
   
       
  
    
    
    