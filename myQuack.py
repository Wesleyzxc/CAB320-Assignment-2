
'''

Scaffolding code for the Machine Learning assignment. 

You should complete the provided functions and add more functions and classes as necessary.
 
You are strongly encourage to use functions of the numpy, sklearn and tensorflow libraries.

You are welcome to use the pandas library if you know it.


'''
#import tensorflow as tf
import keras
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

import numpy as np
import csv
from sklearn import tree, datasets, neighbors, svm
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def my_team():
    '''
    Return the list of the team members of this assignment submission as a list
    of triplet of the form (student_number, first_name, last_name)
    
    '''
    return [ (9930141, 'Daryl', 'Tan'), (9972676, 'Wesley', 'Kok') ]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def prepare_dataset(dataset_path):
    '''  
    Read a comma separated text file where 
	- the first field is a ID number 
	- the second field is a class label 'B' or 'M'
	- the remaining fields are real-valued

    Return two numpy arrays X and y where 
	- X is two dimensional. X[i,:] is the ith example
	- y is one dimensional. y[i] is the class label of X[i,:]
          y[i] should be set to 1 for 'M', and 0 for 'B'

    @param dataset_path: full path of the dataset text file

    @return
	X,y
    '''
    ##         "INSERT YOUR CODE HERE"
    
    with open(dataset_path) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        X = np.array([tuple(value)[2:] for value in csv_reader])
        csvfile.seek(0)
        y = np.array([1 if value[1] == 'M' else 0 for value in csv_reader])
        
        # Safety net for new versions (warning messages)
        X = X.astype(np.float64)
        y = y.astype(np.float64)
            
    return X,y

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -



def build_DecisionTree_classifier(X_training, y_training):
    '''  
    Build a Decision Tree classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    ##         "INSERT YOUR CODE HERE"    
    clf = tree.DecisionTreeClassifier()
    tree_depth = np.arange(1,100)
    gs = GridSearchCV(clf, param_grid={'max_depth':tree_depth}, iid=True, cv=3)
    
    gs.fit(X_training, y_training)
#    print(gs.score(X_training[250:500], y_training[250:500]))
    return gs
    
    
#x = build_DecisionTree_classifier(prepare_dataset("medical_records.data")[0], prepare_dataset("medical_records.data")[1])

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_NearrestNeighbours_classifier(X_training, y_training):
    '''  
    Build a Nearrest Neighbours classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    ##         "INSERT YOUR CODE HERE"    
    clf = neighbors.KNeighborsClassifier()
    num_neighbours = np.arange(5,100)
    gs = GridSearchCV(clf, param_grid={ 'n_neighbors':num_neighbours  }, iid=True, cv=3)
    
    gs.fit(X_training, y_training)
#    print(clf.score(X_training[250:500], y_training[250:500]))
    return gs
    
#x = build_NearrestNeighbours_classifier(prepare_dataset("medical_records.data")[0], prepare_dataset("medical_records.data")[1])

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_SupportVectorMachine_classifier(X_training, y_training):
    '''  
    Build a Support Vector Machine classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    ##         "INSERT YOUR CODE HERE"    
    clf = svm.SVC(gamma="scale")
    C = np.arange(1, 10, dtype='float')
    gs = GridSearchCV(clf, param_grid={ 'C':C }, cv=3)
    gs.fit(X_training, y_training)
    return gs
    
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_NeuralNetwork_classifier(X_training, y_training):
    '''  
    Build a Neural Network with two dense hidden layers classifier 
    based on the training set X_training, y_training.
    Use the Keras functions from the Tensorflow library

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    model = Sequential()
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    
    model.fit(X_training, y_training,
              batch_size=128) 
#    score = model.evaluate(X_training, y_training, batch_size=128)\
    return model

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

if __name__ == "__main__":
    pass
#    X, y = prepare_dataset("medical_records.data")
    X_training, X_testing, y_training, y_testing = train_test_split(X, y, test_size=0.2, shuffle=True)
    
#    clfDT = build_DecisionTree_classifier(X_training, y_training)
#    K_fold_prediction = cross_val_score(clfDT, X_testing, y_testing, cv=3)
#    print(K_fold_prediction)
#    predicted = clfDT.predict(X_testing)
#    print(predicted)
#    print(confusion_matrix(y_testing, predicted))
#    print(clfDT.score(X_testing, y_testing))
#    print(clfDT.best_params_)
#    
#    clfKNN = build_NearrestNeighbours_classifier(X_training, y_training)
#    print(clfKNN.score(X_testing, y_testing))
#    print(clfKNN.best_params_)
#    
#    clfSVM = build_SupportVectorMachine_classifier(X_training, y_training)
#    print(clfSVM.score(X_testing, y_testing))
#    print(clfSVM.best_params_)
    
    
    # Write a main part that calls the different 
    # functions to perform the required tasks and repeat your experiments.
    clf = build_NeuralNetwork_classifier(X_training, y_training)
    score = model.evaluate(X_testing, y_testing, batch_size=128)
    print(score)

    # call your functions here
    


