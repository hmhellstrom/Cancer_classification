"""Module containing the construction of the models 
and the training routine as well as computing evaluation
metrics"""

from tensorflow import keras
from keras import layers
from evaluation import compute_threshold, compute_tnr, compute_tpr
from sklearn.metrics import accuracy_score, confusion_matrix, auc, roc_curve, log_loss
import dill as pickle
import os
import numpy as np


def classify_images(
    train_data: np.ndarray,
    train_labels: np.ndarray,
    test_data: np.ndarray,
    test_labels: np.ndarray,
    iterations: int,
    epochs: int,
    deep_model: bool = True,
) -> None:
    """
    A function which trains convolutional neural
    networks and collects the results into a pkl-
    file
    """
    img_dims = train_data.shape
    for _ in range(iterations):
        print(f"Run number {_+1} starting")
        single_run = {}
        if deep_model:
            # this model is referred to as the "deep model"
            model = keras.Sequential(
                [
                    layers.Conv2D(
                        16,
                        3,
                        activation="relu",
                        input_shape=(img_dims[1], img_dims[2], img_dims[-1]),
                    ),
                    layers.Conv2D(16, 3, activation="relu"),
                    layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
                    layers.Conv2D(32, 3, activation="relu"),
                    layers.Conv2D(32, 3, activation="relu"),
                    layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
                    layers.Conv2D(64, 3, activation="relu"),
                    layers.Conv2D(64, 3, activation="relu"),
                    layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
                    layers.Conv2D(128, 3, activation="relu"),
                    layers.Conv2D(128, 3, activation="relu"),
                    layers.MaxPooling2D(strides=(2, 2)),
                    layers.Flatten(),
                    layers.Dense(128, activation="relu"),
                    layers.Dense(64, activation="relu"),
                    layers.Dense(32, activation="relu"),
                    layers.Dense(1, activation="sigmoid"),
                ]
            )
        else:
            # this model is referred to as the "shallow model"
            model = keras.Sequential(
                [
                    layers.Conv2D(
                        32,
                        3,
                        activation="relu",
                        input_shape=(img_dims[1], img_dims[2], img_dims[-1]),
                    ),
                    layers.Conv2D(32, 3, activation="relu"),
                    layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
                    layers.Conv2D(32, 3, activation="relu"),
                    layers.Conv2D(32, 3, activation="relu"),
                    layers.MaxPooling2D(strides=(2, 2)),
                    layers.Flatten(),
                    layers.Dense(128, activation="relu"),
                    layers.Dense(64, activation="relu"),
                    layers.Dense(32, activation="relu"),
                    layers.Dense(1, activation="sigmoid"),
                ]
            )
        # compile models and use SGD optimizer and
        # binary cross-entropy as loss function
        model.compile(
            optimizer=keras.optimizers.SGD(1e-3),
            loss=keras.losses.BinaryCrossentropy(),
            metrics=[keras.metrics.BinaryAccuracy(), keras.metrics.AUC()],
        )
        # fit model using constant epochs and validation split of 30%
        history = model.fit(x=train_data, y=train_labels, epochs=epochs, validation_split=0.3)
        # make real-valued predictions on separate test data (regression)
        test_predictions = model.predict(test_data)
        # compute classification threshold using Youden's J-statistic
        threshold = compute_threshold(test_labels, test_predictions)
        # make binary classification using computed threshold
        predicted_labels = [1 if t >= threshold else 0 for t in test_predictions]
        # find ROC-curve to compute sensitivity, specificity and F-score
        fpr, tpr, thresholds = roc_curve(test_labels, test_predictions, drop_intermediate=False)
        # compute performance metrics and store them in a dictionary
        single_run["auc"] = auc(fpr, tpr)
        single_run["accuracy"] = accuracy_score(test_labels, predicted_labels)
        single_run["confusion matrix"] = confusion_matrix(test_labels, predicted_labels)
        single_run["sensitivity"] = compute_tpr(confusion_matrix(test_labels, predicted_labels))
        single_run["specificity"] = compute_tnr(confusion_matrix(test_labels, predicted_labels))
        single_run["loss"] = log_loss(test_labels, predicted_labels)
        single_run["threshold"] = threshold
        single_run["history"] = history.history
        single_run["test_predictions"] = test_predictions
        single_run["predicted_labels"] = predicted_labels
        single_run["test_labels"] = test_labels
        print(f"Run number {_+1} finished")
        # check if directory for results exists already
        if not os.path.isdir("results"):
            # create it if not
            os.mkdir("results")
        # pickle the result dictionary for later use and
        # save it to the results folder
        with open(f"results\\model_results_{_+1}.pkl", "wb") as file:
            pickle.dump(single_run, file)
