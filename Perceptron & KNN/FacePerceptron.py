# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 19:47:54 2021

@author: Madhav
"""
import numpy as np
import random
def feature_extract(image, square_size = 7):   
    features = [1] #Bias
                        
    for i in range(0,len(image),square_size): 
        for t in range(0,len(image[0]),square_size):
            val = 0
            for j in range(i, i + square_size):
                for k in range(t, t + square_size):
                    if(image[j][k] != " "):
                        #print(image[j][k])
                        val += 1
                        
            features.append(val)          
                
    return features

class Face_Perceptron:
    
    feature_matrix = []
    labels = []
    weights = []
    
    def __init__(self, features, labels, num_of_features):
        
        #self.features = features
        self.feature_matrix = features
        self.labels = labels
        self.weights = np.random.rand(num_of_features + 1)
        
        
       
    
        #self.weights = np.array([weights_not_face,weights_face])
        
    #def step(self,val):
        #if (val >= 0): return 1;
        #else: return 0;
        
    def train(self):
        while True:        
             num_of_mistakes = 0
             for k in range(len(self.feature_matrix)):
            
                label = int(self.labels[k])
                prediction = (np.dot(self.weights,np.array(self.feature_matrix[k]).T));
                if(prediction >= 0): prediction = 1;
                else: prediction = 0;
            
                if(prediction != label):
                    #print("I made an error!")
                    num_of_mistakes += 1  
                    if(label == 0):
                        self.weights = self.weights - np.array(self.feature_matrix[k])
                    else:
                        self.weights = self.weights + np.array(self.feature_matrix[k])
                        
                
                #print(set_of_weights[0])
             if(num_of_mistakes == 0): break
        
    def predict(self,image_features):
         value = np.dot(self.weights,image_features)
         
         if(value >= 0): return 1;
         else: return 0;
     
        
if __name__ == "__main__":
    
    X = [] # samples
    Y = [] #Labels
    X_test = [] # test images
    num_of_samples = 451 #451 is the total number of training images
    size_of_square = 2
    image_length = 70
    image_width = 60
    training_images =  open("facedatatrain.txt")
    training_labels = open("facedatatrainlabels.txt")
    test_images = open("facedatatest.txt")
    test_labels = open("facedatatestlabels.txt")
    num_of_features = int((image_length/size_of_square) * (image_width/size_of_square))
    Y = [line[:-1] if line[-1] == '\n' else line for line in training_labels][0:num_of_samples]
    Y_test = [line[:-1] if line[-1] == '\n' else line for line in test_labels]
    
    
    
    for i in range(0, num_of_samples * image_length, image_length):  
        array = []
        for j in range(i, i + image_length):
            line = training_images.readline()[0:image_width]
            array.append(list(line))
            #print(line)
        features = feature_extract(array, square_size = size_of_square)
        X.append(features)
        
    for i in range(0, 150 * image_length, image_length):  
        array = []
        for j in range(i, i + image_length):
            line = test_images.readline()[0:image_width]
            array.append(list(line))
            #print(line)
        feature = feature_extract(array, square_size = size_of_square)
        X_test.append(feature)
        
    perceptron_1 = Face_Perceptron(X,Y,num_of_features)
    
    perceptron_1.train()
    
    ### Testing perceptron 1
    num_right = 0
    for k in range(len(X_test)):
        
        prediction = perceptron_1.predict(np.array(X_test[k]).T)
        if(prediction == int(Y_test[k])): num_right += 1
    #print(num_right)
    percentage_right = str((num_right/150) * 100)
    print("The face_perceptron classified " + percentage_right + " percent of the test images correctly" )
        
    