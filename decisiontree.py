# decisiontree.py
"""Predict Parkinson's disease based on dysphonia measurements using a decision tree."""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pdb
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix


# ***MODIFY CODE HERE***
ROOT = 'data'  # Relative path name
THIS = os.path.dirname(os.path.realpath(__file__))  # the current directory of this file

parser = argparse.ArgumentParser(description="Use a Decision Tree model to predict Parkinson's disease.")
parser.add_argument('-xtrain', '--training_data',
                    help='path to training data file, defaults to ROOT/training_data.txt',
                    default=os.path.join(ROOT, 'training_data.txt'))
parser.add_argument('-ytrain', '--training_labels',
                    help='path to training labels file, defaults to ROOT/training_labels.txt',
                    default=os.path.join(ROOT, 'training_labels.txt'))
parser.add_argument('-xtest', '--testing_data',
                    help='path to testing data file, defaults to ROOT/testing_data.txt',
                    default=os.path.join(ROOT, 'testing_data.txt'))
parser.add_argument('-ytest', '--testing_labels',
                    help='path to testing labels file, defaults to ROOT/testing_labels.txt',
                    default=os.path.join(ROOT, 'testing_labels.txt'))
parser.add_argument('-a', '--attributes',
                    help='path to file containing attributes (features), defaults to ROOT/attributes.txt',
                    default=os.path.join(ROOT, 'attributes.txt'))
parser.add_argument('--debug', action='store_true', help='use pdb.set_trace() at end of program for debugging')
parser.add_argument('--save', action='store_true', help='save tree image to file')
parser.add_argument('--show', action='store_true', help='show tree image while running code')

def main(args):
    print("Training a Decision Tree to Predict Parkinson's Disease")
    print("=======================")
    print("PRE-PROCESSING")
    print("=======================")

    # Parse input arguments
    training_data_path = os.path.expanduser(args.training_data)
    training_labels_path = os.path.expanduser(args.training_labels)
    testing_data_path = os.path.expanduser(args.testing_data)
    testing_labels_path = os.path.expanduser(args.testing_labels)
    attributes_path = os.path.expanduser(args.attributes)

    # Load data from relevant files
    # ***MODIFY CODE HERE***
    print(f"Loading training data from: {os.path.basename(training_data_path)}")
    #Load the training data into a 2d numpy array
    xtrain = np.loadtxt(training_data_path, dtype=float, delimiter=",")

    print(f"Loading training labels from: {os.path.basename(training_labels_path)}")
    ytrain = np.loadtxt(training_labels_path, dtype=int)

    print(f"Loading testing data from: {os.path.basename(testing_data_path)}")
    #Load the testing data into a 2d numpy array
    xtest = np.loadtxt(testing_data_path, dtype=float, delimiter=",")

    print(f"Loading testing labels from: {os.path.basename(testing_labels_path)}")
    ytest = np.loadtxt(testing_labels_path, dtype=int)

    print(f"Loading attributes from: {os.path.basename(attributes_path)}")
    attributes = np.loadtxt(attributes_path, dtype=str)

    print("\n=======================")
    print("TRAINING")
    print("=======================")
    # Use a DecisionTreeClassifier to learn the full tree from training data
    print("Training the entire tree...")
    # ***MODIFY CODE HERE***
    #clf's random_state is 0 because that random coice results in higher testing accuracy
    clf = DecisionTreeClassifier(criterion="entropy", random_state= 0)  # or choose another criterion if you prefer
    clf.fit(xtrain, ytrain)


    # Visualize the tree using matplotlib and plot_tree
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 5), dpi=150)
    # ***MODIFY CODE HERE***
    plot_tree(clf, rounded= True, filled= True, class_names= ["Healthy", "Parkinson's"], feature_names= attributes)

    # Saves the plotted tree into a png
    if args.save:
        filename = os.path.expanduser(os.path.join(THIS, 'tree.png'))
        print(f"  Saving to file: {os.path.basename(filename)}")
        plt.savefig(filename, bbox_inches='tight')
    plt.show(block=args.show)
    plt.close(fig)

    # Validating the root node of the tree by computing information gain
    print("Computing the information gain for the root node...")
    # ***MODIFY CODE HERE***

    index = clf.tree_.feature[0]  # index of the attribute that was determined to be the root node
    thold = clf.tree_.threshold[0]  # threshold on that attribute
    gain = information_gain(xtrain, ytrain, index, thold)
    print(f"  Root: {attributes[index]}<={thold:0.3f}, Gain: {gain:0.3f}")

    # Test the decision tree
    print("\n=======================")
    print("TESTING")
    print("=======================")
    # ***MODIFY CODE HERE***
    print("Predicting labels for training data...")
    ptrain = clf.predict(xtrain)
    print("Predicting labels for testing data...")
    ptest = clf.predict(xtest)
    #pdb.set_trace()

    print("\n=======================")
    print("PERFORMANCE METRICS")
    print("=======================")

    # Compare training and test accuracy
    # ***MODIFY CODE HERE***

    # Values are decimal form, we multipy by 100 to get percentage form
    accuracy_train = clf.score(xtrain, ytrain) * 100
    accuracy_test = clf.score(xtest, ytest) * 100
    # Accuracy: # correct predictions / # total predictions
    print(f"Training Accuracy: {np.sum(ytrain == ptrain)}/{ytrain.size} ({accuracy_train}%)")
    print(f"Testing Accuracy: {np.sum(ytest == ptest)}/{ytest.size} ({accuracy_test}%)")

    # Show the confusion matrix for test data
    # ***MODIFY CODE HERE***
    # let sklearn make a confusion matrix of the testing data
    cm =confusion_matrix(ytest, ptest)
    # Display the confusion matrix in a grid
    print(f"Confusion matrix:\n {str(cm).replace('[', '').replace(']', '')}") 

    # Debug (if requested)
    if args.debug:
        pdb.set_trace()




#Name: informatioin_gain()
#Input: numpy array of training data, numpy array of training labels, 
        # index of the attribute that was determined to be the root node,
        #  threshold on that attribute
def information_gain(x, y, index, thold):
    """Compute the information gain on y for a continuous feature in x (using index) by applying a threshold (thold).

    NOTE: The threshold should be applied as 'less than or equal to' (<=)"""
    
    # Calculate H(Y) entropy
    pzeros = np.count_nonzero(y == 0) / y.size
    pones = np.count_nonzero(y == 1) / y.size

    ent = -pzeros * np.log2(pzeros) - pones * np.log2(pones)
    # End H(Y)

    # Calculate (H(Y|X=xi)) specific conditional entropy 
    
    pover = 0 #Over the thresh-hold total
    plower = 0 #Under the thresh-hold total

    tlowpoz = 0 #Under the threshold positives
    tlowneg = 0 #Under the threshold negatives
    tgrepoz = 0 #Over the threshold positives
    tgreneg = 0 #Over the threshold negatives
    #For each sample in y
    for i in range (0, y.size):
        # If the sample is over the threshold
        if(x[i, index] > thold):
            # Increment the counter for all over the threshhold
            pover += 1
            #If the sample is negative for parkinsons (healthy)
            if(y[i] == 0):
                tgreneg += 1
            #If the sample is negatiive for parkinsons
            else:
                tgrepoz += 1
        else:
            # Increment the counter for all under the threshhold
            plower += 1
            #If the sample is negative for parkinsons (healthy)
            if(y[i] == 0):
                tlowneg += 1 
            #If the sample is negatiive for parkinsons
            else:
                tlowpoz += 1

    # Calculate probabilities
    tgrepoz /= pover
    tgreneg /= pover

    tlowneg /= plower
    tlowpoz /= plower
    
    plower = plower / y.size
    pover = pover / y.size

    hlower = -tlowneg * np.log2(tlowneg) - tlowpoz * np.log2(tlowpoz)
    hover = -tgreneg * np.log2(tgreneg) - tgrepoz * np.log2(tgrepoz)

    # End (H(Y|X=xi))

    # Calculate H(Y|X) conditional entropy

    hgeneral= (plower * hlower) + (pover * hover)
    # End H(Y|X) 

    # Calculate G(Y|X)
    gain = ent - hgeneral
    # End G(Y|X)

    # Return G(Y|X)
    return gain

if __name__ == '__main__':
    main(parser.parse_args())
