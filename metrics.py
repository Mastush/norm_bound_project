from keras.metrics import categorical_accuracy, top_k_categorical_accuracy


def get_metrics():
    return [categorical_accuracy, top_k_categorical_accuracy]