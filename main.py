import Decision_Tree_Classifier
import Other_Classifiers
import Tunable_Classes
import Testbenches as TB
import numpy as np
import sys

if len(sys.argv) == 3:
  dataset_path = sys.argv[1]
  depth = sys.argv[2]
else:
  print("arguements taken: <dataset path> <depth>")
  quit()
print("Loading dataset:", dataset_path)
print("Tree depth:", depth)

dataset = np.loadtxt(dataset_path)

#test = Decision_Tree_Classifier.Tree(dataset, max_depth = depth)
#print(test.evaluate([[-68,	-58,	-65,	-65,	-76,	-87,	-82,	1]]))
#test.show()

#benchmark = TB.Model_Comparison_TB(dataset, 10, depth, Decision_Tree_Classifier.Tree, Other_Classifiers.RandomClassifier, Other_Classifiers.NNClassifier)
#benchmark.all_metrics()

#benchmark.precision()
#benchmark.plotNorm()

hyperParam = TB.Depth_Hyperparameter_Tuning(dataset, Decision_Tree_Classifier.Tree)
hyperParam.show()

#test = RandomClassifier.NNClassifier(dataset)
#print((test.confusion_constructor()[2]))
