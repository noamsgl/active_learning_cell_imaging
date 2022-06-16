import numpy as np
from matplotlib import pyplot as plt
from modAL import ActiveLearner
from modAL.uncertainty import uncertainty_sampling
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, plot_confusion_matrix

if __name__ == '__main__':
    data = pd.read_csv('Zaretski_Image_All.csv')
    # clean irrelevant data
    data = data.drop(['image_id', 'cell_id'], axis=1)
    # filter platelets
    data = data[data['cell_type'] != 'Platelets']
    # filter selected features
    # filter_data = data.filter([''])

    target = 'cell_type'
    label_encoder = preprocessing.LabelEncoder()
    y = label_encoder.fit_transform(data[target])
    X = data.iloc[:, 3:100]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    dummy_learner = LogisticRegression()
    dummy_learner.fit(X_train, y_train)
    predictions = dummy_learner.predict(X_test)
    score = confusion_matrix(y_test, predictions)
    plot_confusion_matrix(dummy_learner, X_test, y_test, labels=label_encoder.classes_)
    plt.show()
    print("acc score {}".format(score))
    x = 5

    learner_1 = ActiveLearner(
        estimator=LogisticRegression(),
        query_strategy=uncertainty_sampling
    )
