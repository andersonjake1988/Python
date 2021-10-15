##############################################################################
# Package loading
##############################################################################

# Install Packages in anaconda if you don't have them:
# conda install numpy 
# conda install pandas
# conda install keras
# conda install tensorflow

# Import Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense
from keras.utils import to_categorical
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
import itertools


##############################################################################
# Data importing and Object Defining
##############################################################################
# 'PassengerId', 
# Setting up Training Data
train_raw = pd.read_csv('C:/Users/ander/Documents/Python/titanic/train.csv')
train_raw.info()
train_raw1 = train_raw[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']]
train_raw1["male"] = np.multiply(train_raw.Sex == 'male', 1)
train_raw1["age_was_missing"] = np.multiply(train_raw.Age.isna(), 1)
train_raw1["Age"] = train_raw["Age"].fillna(value=train_raw["Age"].mean())
train_raw1.info()
train = np.array(train_raw1)

# Assign Model Training Variables
target = to_categorical(train_raw.Survived)
predictors = train
n_cols = predictors.shape[1]

##############################################################################
# Model Building 
##############################################################################

# Model Data
model = Sequential()
model.add(Dense(100, activation='relu', input_shape = (n_cols,)))
model.add(Dense(100, activation='relu'))
#model.add(Dense(100, activation='relu'))
#model.add(Dense(100, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics = ['accuracy'])
model.fit(predictors, target, validation_split=0.3, epochs=100)

# Save/Load Model 
model.save('C:/Users/ander/Documents/Python/titanic/titanic_model.h5')
titanic_model = load_model('C:/Users/ander/Documents/Python/titanic/titanic_model.h5')

##############################################################################
# Model Testing
##############################################################################

# Setting up Test data
test_raw = pd.read_csv("C:/Users/ander/Documents/Python/titanic/test.csv")
# test_raw.info()
test_raw1 = test_raw[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']]
test_raw1["male"] = np.multiply(test_raw.Sex == 'male', 1)
test_raw1["age_was_missing"] = np.multiply(test_raw.Age.isna(), 1)
test_raw1["Age"] = test_raw["Age"].fillna(value=test_raw["Age"].mean())
test_raw1["Fare"] = test_raw["Fare"].fillna(value=test_raw["Fare"].mean())
test = np.array(test_raw1)

# Actual test values
test_true_raw = pd.read_csv("C:/Users/ander/Documents/Python/titanic/gender_submission.csv")
test_true = np.array(test_true_raw.Survived)

# Use Model to Predict
predictions = titanic_model.predict(test)
test_pred = np.argmax(predictions, axis=-1)

##############################################################################
# Visualize Results 
##############################################################################
# Make Confusion Matrix (CM)
cm = confusion_matrix(y_true = test_true, y_pred = test_pred)
cm
# Aesthetics For Plotting CM
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)#, rotation=45)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm=cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Normalized confusion matrix')
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i,j],
                 horizontalalignment="center",
                 color="white" if cm[i,j] > thresh else "black")
        plt.tight_layout()
        plt.ylabel('True Survival')
        plt.xlabel('Predicted Survival')


# CM Plot
cm_plot_labels = ['Didn\'t Survive', 'Survived']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title="Titanic NN Confusion Matrix") 

# CM Statistics
accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
precision = cm[1,1]/(cm[1,1] + cm[0,1])
true_positive_rate = cm[1,1]/(cm[1,1] + cm[1,0])
true_negative_rate = cm[0,0]/(cm[0,0] + cm[0,1])
f1_score = (2 * precision * true_positive_rate)/(precision + true_positive_rate)
print('\nAccuracy = %.2f \nPrecision = %.2f \nRecall(TPR) = %.2f \nSpecificity(TNR) = %.2f \nF1 Score = %.2f'%
      (accuracy, precision, true_positive_rate, true_negative_rate, f1_score))









