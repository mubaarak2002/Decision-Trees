import DecisionTreeClassifier
import RandomClassifier
from Testbench import Testbench
import numpy as np

clean_dataset = np.loadtxt("WIFI_db/clean_dataset.txt")
noisy_dataset = np.loadtxt("WIFI_db/noisy_dataset.txt")

#test = DecisionTreeClassifier.Tree(clean_dataset)
#test.run_test()

benchmark = Testbench(noisy_dataset, 10, DecisionTreeClassifier.Tree, RandomClassifier.RandomClassifier, RandomClassifier.NNClassifier)
benchmark.precision()
benchmark.plotNorm()

#test = RandomClassifier.NNClassifier(clean_dataset)
#print((test.confusion_constructor()[2]))
