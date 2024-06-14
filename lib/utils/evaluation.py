from sklearn.metrics import balanced_accuracy_score, confusion_matrix


def get_balance_accuracy_score(preds, labels):
    return balanced_accuracy_score(labels, preds)


def print_confusion_matrix(preds, labels):
    print("Confusion Matrix")
    print(confusion_matrix(labels, preds))


