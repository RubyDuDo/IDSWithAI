
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import train_test_split

def train_and_evaluate( classfier, X_train, y_train, X_test, y_test ):
    classfier.fit( X_train, y_train )

    # 预测测试集
    y_pred = classfier.predict(X_test)

    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    return accuracy

# train with validation dataset(10% val, 90% train)
def train_and_evaluate_with_val( classfiers, X_train, y_train, X_test, y_test ):
    X_train, X_Val, y_train, y_val = train_test_split( X_train, y_train, test_size = 0.1, random_state=42)

    acc = 0
    bestC = None
    for classfier in classfiers:
        classfier.fit( X_train, y_train )
        acctrain = classfier.score( X_train, y_train )
        accuracy = classfier.score( X_Val, y_val )
        print(f"Test (on val dateset ) accuracy: {acctrain * 100:.2f}% __ {accuracy * 100:.2f}%")
        if accuracy > acc :
            acc = accuracy
            bestC = classfier

    accuracy = bestC.score( X_test, y_test )
    print(f"\n\nTest accuracy: {accuracy * 100:.2f}%")
    return accuracy, bestC


