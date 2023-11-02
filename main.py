import Decision_Tree_Classifier
import Other_Classifiers
import Testbenches as TB
import numpy as np

dataset_path = "WIFI_db/clean_dataset.txt"
dataset = np.loadtxt(dataset_path)

test = Decision_Tree_Classifier.Tree(dataset)
test.show()

#benchmark = TB.Model_Comparison_TB(dataset, 10, Decision_Tree_Classifier.Tree, Other_Classifiers.RandomClassifier, Other_Classifiers.NNClassifier)
#benchmark.all_metrics()


#benchmark.precision()
#benchmark.plotNorm()

#test = RandomClassifier.NNClassifier(dataset)
#print((test.confusion_constructor()[2]))
