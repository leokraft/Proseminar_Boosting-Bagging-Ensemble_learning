from sklearn.metrics import accuracy_score

def getMetrics(learner, X_train, X_test, y_train, y_test):
    """
    Predict the training and test data using the given learner and calculate the resulting accuracy.
    Arguments:
        learner: an integer
        X_train: The training data.
        X_test: The test data.
        y_train: The training data labels.
        y_test: The test data labels.
    Returns:
        None.
    """
    learner.fit(X_train, y_train)

    train_y_pred = learner.predict(X_train)
    test_y_pred =  learner.predict(X_test)

    train_acc = accuracy_score(y_train, train_y_pred)
    test_acc = accuracy_score(y_test, test_y_pred)

    print(f"Training accuracy: {train_acc}")
    print(f"Test accuracy: {test_acc}")
    print()