import numpy as np
from matplotlib import pyplot as plt
from modAL import ActiveLearner
import pandas as pd
from modAL.uncertainty import uncertainty_sampling, entropy_sampling, margin_sampling
from sklearn import svm, preprocessing
from sklearn.metrics import balanced_accuracy_score, accuracy_score, average_precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
#
# def active_learning_experiment(train_set, test_set):
#
#     # for each sample size we train the model with a dataset with size of i
#     # for each strategy, the train set will contains different samples.
#
#     random_strategy_accuracy_score = []
#     our1_strategy_accuracy_score = []
#     our2_strategy_accuracy_score = []
#
#     for i in range(train_set):
#         # --------- random strategy --------- #
#         # we pick i samples randomly from the train set then train the model
#         learner = ActiveLearner(
#             estimator=svm(),
#             query_strategy=uncertainty_sampling,
#             X_training=train_set['X'],
#             y_training=train_set['y']
#         )
#
#         # --------- our strategy 1 --------- #
#         # --------- our strategy 2 --------- #
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    train_set, test_set, strategy = None, None, None
    # train/test set should be a dict where 'X' key contains the features and 'y' key contains the labels

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
    num_of_row_to_begin = 45
    X = data.iloc[:, 3:]

    X_train, X_test, y_train, y_test = train_test_split(X.to_numpy(), y, test_size=0.33, random_state=42)

    # --- dummy vs acctive --- #
    # dummy_learner = LogisticRegression()
    # learner_1 = ActiveLearner(
    #     estimator=LogisticRegression(),
    #     query_strategy=uncertainty_sampling
    # )
    # learner_2 = ActiveLearner(
    #     estimator=LogisticRegression(),
    #     query_strategy=entropy_sampling
    # )
    # learner_3 = ActiveLearner(
    #     estimator=LogisticRegression(),
    #     query_strategy=margin_sampling
    # )

    # --- model robustness --- #
    dummy_learner = svm.SVC(probability=True)
    learner_1 = ActiveLearner(
        estimator=LogisticRegression(),
        query_strategy=uncertainty_sampling
    )
    learner_2 = ActiveLearner(
        estimator=RandomForestClassifier(),
        query_strategy=entropy_sampling
    )
    learner_3 = ActiveLearner(
        estimator=svm.SVC(probability=True),
        query_strategy=margin_sampling
    )

    base_size = 5
    X_train_base_dummy = X_train[:base_size]
    X_train_base_st1 = X_train[:base_size]
    X_train_base_st2 = X_train[:base_size]
    X_train_base_st3 = X_train[:base_size]
    y_train_base_dummy = y_train[:base_size]
    y_train_base_st1 = y_train[:base_size]
    y_train_base_st2 = y_train[:base_size]
    y_train_base_st3 = y_train[:base_size]

    X_train_new_dummy = X_train[base_size:]
    X_train_new_st1 = X_train[base_size:]
    X_train_new_st2 = X_train[base_size:]
    X_train_new_st3 = X_train[base_size:]
    y_train_new_dummy = y_train[base_size:]
    y_train_new_st1 = y_train[base_size:]
    y_train_new_st2 = y_train[base_size:]
    y_train_new_st3 = y_train[base_size:]

    dummy_scores = []
    st1_scores = []
    st2_scores = []
    st3_scores = []
    # range_epoch = X_train.shape[0] - 2
    range_epoch = 300
    score_meth = 'average_precision_score'
    for i in range(range_epoch):
        print(i)
        # train again the model
        learner_1.fit(X_train_base_st1, y_train_base_st1)
        learner_2.fit(X_train_base_st2, y_train_base_st2)
        learner_3.fit(X_train_base_st3, y_train_base_st3)
        dummy_learner.fit(X_train_base_dummy, y_train_base_dummy)

        # evaluate the models
        dummy_pred = dummy_learner.predict(X_test)
        st1_pred = learner_1.predict(X_test)
        st2_pred = learner_2.predict(X_test)
        st3_pred = learner_3.predict(X_test)

        if score_meth == 'auc':
            dummy_scores.append(accuracy_score(dummy_pred, y_test))
            st1_scores.append(accuracy_score(st1_pred, y_test))
            st2_scores.append(accuracy_score(st2_pred, y_test))
            st3_scores.append(accuracy_score(st3_pred, y_test))

        elif score_meth == 'average_precision_score':
            dummy_scores.append(average_precision_score(dummy_pred, y_test))
            st1_scores.append(average_precision_score(st1_pred, y_test))
            st2_scores.append(average_precision_score(st2_pred, y_test))
            st3_scores.append(average_precision_score(st3_pred, y_test))


        else:
            dummy_scores.append(accuracy_score(dummy_pred[np.where(y_test == 1)], y_test[np.where(y_test == 1)]))
            st1_scores.append(accuracy_score(st1_pred[np.where(y_test == 1)], y_test[np.where(y_test == 1)]))
            st2_scores.append(accuracy_score(st2_pred[np.where(y_test == 1)], y_test[np.where(y_test == 1)]))
            st3_scores.append(accuracy_score(st3_pred[np.where(y_test == 1)], y_test[np.where(y_test == 1)]))

        # pick next sample
        query_idx1, query_sample1 = learner_1.query(X_train_new_st1)
        query_idx2, query_sample2 = learner_2.query(X_train_new_st2)
        query_idx3, query_sample3 = learner_3.query(X_train_new_st3)

        # add randomly to the dummy database
        X_train_base_dummy = np.append(X_train_base_dummy, [X_train_new_dummy[0, :]], axis=0)
        y_train_base_dummy = np.concatenate([y_train_base_dummy, np.array([y_train_new_dummy[0]])], axis=0)
        X_train_new_dummy = X_train_new_dummy[1:]
        y_train_new_dummy = y_train_new_dummy[1:]

        # add by index to the smart database
        X_train_base_st1 = np.append(X_train_base_st1, X_train_new_st1[query_idx1], axis=0)
        y_train_base_st1 = np.concatenate([y_train_base_st1, y_train_new_st1[query_idx1]], axis=0)
        X_train_new_st1 = np.concatenate([X_train_new_dummy[:query_idx1[0]], X_train_new_dummy[query_idx1[0] + 1:]],
                                         axis=0)
        y_train_new_st1 = np.concatenate([y_train_new_dummy[:query_idx1[0]], y_train_new_dummy[query_idx1[0] + 1:]],
                                         axis=0)

        # add by index to the smart database
        X_train_base_st2 = np.append(X_train_base_st2, X_train_new_st2[query_idx2], axis=0)
        y_train_base_st2 = np.concatenate([y_train_base_st2, y_train_new_st2[query_idx2]], axis=0)
        X_train_new_st2 = np.concatenate([X_train_new_dummy[:query_idx2[0]], X_train_new_dummy[query_idx2[0] + 1:]],
                                         axis=0)
        y_train_new_st2 = np.concatenate([y_train_new_dummy[:query_idx2[0]], y_train_new_dummy[query_idx2[0] + 1:]],
                                         axis=0)

        # add by index to the smart database
        X_train_base_st3 = np.append(X_train_base_st3, X_train_new_st3[query_idx3], axis=0)
        y_train_base_st3 = np.concatenate([y_train_base_st3, y_train_new_st3[query_idx3]], axis=0)
        X_train_new_st3 = np.concatenate([X_train_new_dummy[:query_idx3[0]], X_train_new_dummy[query_idx3[0] + 1:]],
                                         axis=0)
        y_train_new_st3 = np.concatenate([y_train_new_dummy[:query_idx3[0]], y_train_new_dummy[query_idx3[0] + 1:]],
                                         axis=0)

        if i == range_epoch - 2:
            break

    plt.plot(list(range(range_epoch - 1)), st1_scores, label='Logistic Regression')
    plt.plot(list(range(range_epoch - 1)), st2_scores, label='Random Forest')
    plt.plot(list(range(range_epoch - 1)), st3_scores, label='Support Vector')
    plt.plot(list(range(range_epoch - 1)), dummy_scores, label='dummy Random Forest')

    # plt.plot(list(range(range_epoch - 1)), st1_scores, label='uncertainty sampling')
    # plt.plot(list(range(range_epoch - 1)), st2_scores, label='entropy sampling')
    # plt.plot(list(range(range_epoch - 1)), st3_scores, label='margin sampling')

    plt.xlabel('number of added samples')
    plt.ylabel('average precision score')
    plt.legend()
    plt.savefig("models robustness.pdf", bbox_inches='tight')
    plt.show()

    print("y dummy base progression: {}".format(y_train_base_dummy))
    print("y st1 base progression: {}".format(y_train_base_st1))

    x = 5

# TODO - train a baseline accuracy with all the training data
# TODO - active learning by batch --> give more then one samples
# TODO - try few models too show robust
