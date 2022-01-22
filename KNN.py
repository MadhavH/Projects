# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 15:15:12 2021

@author: Madhav


"""
from scipy.spatial import distance
from collections import Counter
import numpy as np

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
class KNN:
    def fit(self,X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    def predict(self, X_test):
        predictions = []
        for row in X_test:
            label = self.closest(row)
            predictions.append(label)
        return predictions   
    def closest(self, row, k = 9):
        
        closest_list = []
        for i in range(len(self.X_train)):
            
            dist = distance.euclidean(row,self.X_train[i])
            closest_list.append((self.y_train[i],dist))
            
        closest_list.sort(key = lambda x: x[1])
        closest_list = closest_list[0:k]
        list_of_labels = [l[0] for l in closest_list ]
        labels_freq = list(Counter(list_of_labels).items())
        labels_freq.sort(key = lambda x: x[1], reverse = True)
        return labels_freq[0][0]
            

def test_num_images():

    X = [] # samples
    Y = [] #Labels
    X_test = [] # test images
    num_of_samples = 5000
    size_of_square = 2
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
    
     
    digit_knn = KNN()
    digit_knn.fit(X,Y)
    predictions = digit_knn.predict(X_test)
    
    where_correct = np.array(predictions) == np.array(Y_test)
    percent_correct = (where_correct[where_correct == True].size/ where_correct.size) * 100
    #print(predictions)
    print("This knn classifier classified " + str(percent_correct) + "% of the digit test images correctly")
    
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
        
    face_knn = KNN()
    face_knn.fit(X,Y)
    predictions = face_knn.predict(X_test)
    where_correct = np.array(predictions) == np.array(Y_test)
    percent_correct = (where_correct[where_correct == True].size/ where_correct.size) * 100
    #print(predictions)
    print("This knn classifier classified " + str(percent_correct) + "% of the face test images correctly")
    
    
