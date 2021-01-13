import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

profile_train = pd.read_csv("RBF/src/treinamento2.csv")
#rofile_validation = pd.read_csv("RBF/src/validacao.csv")

X = profile_train.iloc[:, 0:21]

Y = profile_train.iloc[:, 21:22]

X.head()
Y.head()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

mlp = MLPClassifier(hidden_layer_sizes=(100, 250, 150, 10),
                    max_iter=50000)
mlp.fit(X_train, Y_train.values.ravel())

predictions = mlp.predict(X_test)

print(predictions)

print(confusion_matrix(Y_test, predictions))
print(classification_report(Y_test, predictions))
