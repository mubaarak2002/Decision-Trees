import DecisionTreeClassifier
import numpy as np

clean_dataset = np.loadtxt("WIFI_db/clean_dataset.txt")
noisy_dataset = np.loadtxt("WIFI_db/noisy_dataset.txt")

test = DecisionTreeClassifier.Tree(clean_dataset)
test.show()