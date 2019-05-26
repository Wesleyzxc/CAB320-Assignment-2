'''
Scaffolding code for the Machine Learning assignment. 
You should complete the provided functions and add more functions and classes as necessary.
 
You are strongly encourage to use functions of the numpy, sklearn and tensorflow libraries.
You are welcome to use the pandas library if you know it.
'''
#import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import math
import matplotlib.pyplot as plt
import numpy as np
import csv
from sklearn import tree, neighbors, svm
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Keeping constant seed for graph
np.random.seed(2)

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
    print("~ Decision Tree ~") 
    clf = tree.DecisionTreeClassifier()
    tree_depth = np.arange(1,100)
    gs = GridSearchCV(clf, param_grid={'max_depth':tree_depth}, iid=True, cv=3)

    K_fold_prediction = cross_val_score(gs, X_training, y_training, cv=3)
    print("Accuracy: %0.2f (+/- %0.2f)" % (K_fold_prediction.mean(), K_fold_prediction.std() * 2))
    print("Cross validation scores are:", K_fold_prediction)
    
    gs.fit(X_training, y_training)
    
#    dictResults = gs.cv_results_
#    fig1, ax1 = plt.subplots()
#    ax1.plot(tree_depth, dictResults.get("mean_test_score", ""))
#    ax1.set_title("Hyperparameter testing for Decision Tree")
#    ax1.set_xlabel('Max Depth (n)')
#    ax1.set_ylabel('Accuracy')
    
    
    print("The training set gave a best score of " + str(gs.best_score_))
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
    print("~ Nearest Neighbour ~")
    clf = neighbors.KNeighborsClassifier()
    num_neighbours = np.arange(1,25)
    gs = GridSearchCV(clf, param_grid={ 'n_neighbors':num_neighbours  }, iid=True, cv=3)
       
    K_fold_prediction = cross_val_score(gs, X_training, y_training, cv=3)
    print("Accuracy: %0.2f (+/- %0.2f)" % (K_fold_prediction.mean(), K_fold_prediction.std() * 2))
    print("Cross validation scores are:", K_fold_prediction)
    
    gs.fit(X_training, y_training)
    
#    dictResults = gs.cv_results_
#    fig2, ax2 = plt.subplots()
#    ax2.plot(num_neighbours, dictResults.get("mean_test_score", ""))
#    ax2.set_title("Hyperparameter testing for Nearest Neighbour")
#    ax2.set_xlabel('Number of neighbours (n)')
#    ax2.set_ylabel('Accuracy')
    
    
    print("The training set gave a best score of " + str(gs.best_score_))
    return gs
    

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
    print("~ Support Vector Machine ~")
    clf = svm.SVC(gamma="scale")
    C = np.arange(0.1, 5, dtype='float')
    gs = GridSearchCV(clf, param_grid={ 'C':C }, iid=True, cv=3)
    
    
    K_fold_prediction = cross_val_score(gs, X_training, y_training, cv=3)
    print("Accuracy: %0.2f (+/- %0.2f)" % (K_fold_prediction.mean(), K_fold_prediction.std() * 2))
    print("Cross validation scores are:", K_fold_prediction)
    
    gs.fit(X_training, y_training)
    
#    dictResults = gs.cv_results_
#    fig3, ax3 = plt.subplots()
#    ax3.plot(C, dictResults.get("mean_test_score", ""))
#    ax3.set_title("Hyperparameter testing for SVM")
#    ax3.set_xlabel('C value')
#    ax3.set_ylabel('Accuracy')
    
    
    
    print("The training set gave a best score of " + str(gs.best_score_))
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
    
    # Normalising training set
    scaler = StandardScaler()
    scaler.fit(X_training)
    X_train_scaled = scaler.transform(X_training)

    model = Sequential()
    model.add(Dense(60, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(60, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    
    rms = RMSprop(lr=0.001)
    model.compile(loss='binary_crossentropy',
                  optimizer=rms,
                  metrics=['accuracy'])
    epochSize = 150
    r = model.fit(X_train_scaled, y_training,
              batch_size=30, epochs=epochSize, verbose=0)
    
    accuracy = (r.history['acc'])
    fig4, ax4 = plt.subplots()
    ax4.plot(np.arange(1,epochSize+1, dtype="int"), accuracy)
    ax4.set_title("Scoring after each run for Neural network")
    ax4.set_xlabel('Epoch number (n)')
    ax4.set_ylabel('Accuracy')
    print("Accuracy of: ", accuracy[-1])

    return model

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    


if __name__ == "__main__":
    # Write a main part that calls the different 
    # functions to perform the required tasks and repeat your experiments.
    
    def printReport(clf, X_test, y_test ):
        predicted = clf.predict(X_test)
        # binary classification to extract each data values
        tn, fp, fn, tp = confusion_matrix(y_test, predicted).ravel()
    #    K_fold_prediction = cross_val_score(clf, X_test, y_test, cv=3)
    #    print("Cross validation scores are: ", K_fold_prediction)
        print(tp, "/" , tp+fp , " predicted M correctly", fp, "/" , tp+fp, "predicted M wrongly")
        print(tn, "/" , tn+fn , " predicted B correctly", fn, "/" , tn+fn, "predicted B wrongly")
        print("The test set gave an accuracy of ", clf.score(X_test, y_test))
        print("With the best parameters: ", clf.best_params_ , "\n")
    
    
    X, y = prepare_dataset("medical_records.data")
#    dataSplit = float(input("Enter a test_size double: "))
    dataSplit = 0.2
    X_training, X_testing, y_training, y_testing = train_test_split(X, y, test_size=dataSplit, shuffle=True, random_state=2)
    print("Out of ", math.ceil(len(X)*dataSplit), "predictions: ")
    
    clfDT = build_DecisionTree_classifier(X_training, y_training)
    printReport(clfDT, X_testing, y_testing)
    
    clfKNN = build_NearrestNeighbours_classifier(X_training, y_training)
    printReport(clfKNN, X_testing, y_testing)
 
    clfSVM = build_SupportVectorMachine_classifier(X_training, y_training)
    printReport(clfSVM, X_testing, y_testing)    

    print("~ Neural Network ~")
    clf = build_NeuralNetwork_classifier(X_training, y_training)
    score = clf.evaluate(X_training, y_training, verbose=0)
    print("Evaluated score is: ", score)