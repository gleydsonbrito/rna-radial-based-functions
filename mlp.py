import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

profile_train = pd.read_csv("RBF/src/treinamento2.csv")

X = profile_train[profile_train.columns[:-1]]

Y = profile_train[profile_train.columns[-1]]

X.head()
Y.head()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

mlp = MLPClassifier(hidden_layer_sizes=(40, 60, 25, 10), random_state=1, activation='identity',
                    solver='adam', max_iter=5000, learning_rate_init=0.0015, learning_rate='adaptive')
mlp.fit(X_train, Y_train.values.ravel())

predictions = mlp.predict(X_test)

print(predictions)

print(confusion_matrix(Y_test, predictions))
print(classification_report(Y_test, predictions))
