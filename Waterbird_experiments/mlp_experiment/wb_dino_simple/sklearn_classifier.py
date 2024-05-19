import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

x_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')

test_dict = np.load('x_test_group_dict.npy', allow_pickle=True).item()

clf = LogisticRegression(random_state=2)
#clf = MLPClassifier(hidden_layer_sizes=(32,))

clf.fit(x_train, y_train)

for key in test_dict.keys():
    data = test_dict[key]
    label = float(key[:1]) * np.ones(len(data))
    acc = np.round(accuracy_score(label, clf.predict(data)), 3)
    print(f'group: {key}, acc: {acc}')