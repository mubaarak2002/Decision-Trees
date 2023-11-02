import Decision_Tree_Classifier
import Other_Classifiers
import Testbenches as TB
import numpy as np

clean_dataset = np.loadtxt("WIFI_db/clean_dataset.txt")
noisy_dataset = np.loadtxt("WIFI_db/noisy_dataset.txt")

test = Decision_Tree_Classifier.Tree(clean_dataset)
test.show()

#benchmark = TB.Model_Comparison_TB(clean_dataset, 10, Decision_Tree_Classifier.Tree, Other_Classifiers.RandomClassifier, Other_Classifiers.NNClassifier)
#benchmark.all_metrics()


#benchmark.precision()
#benchmark.plotNorm()

#test = RandomClassifier.NNClassifier(clean_dataset)
#print((test.confusion_constructor()[2]))
