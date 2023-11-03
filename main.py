import Decision_Tree_Classifier
import Other_Classifiers
import Tunable_Classes
import Testbenches as TB
import numpy as np
import sys

def main():
  if len(sys.argv) == 4:
    dataset_path = sys.argv[1]
    depth = sys.argv[2]
    mode = sys.argv[3]
  else:
    print("arguements taken: <dataset path> <depth> <operating_mode>")
    quit()

  print("Loading dataset:", dataset_path)
  dataset = np.loadtxt(dataset_path)
  print("Tree depth:", depth)

  if mode == "show_tree":
     print("Showing the Tree for the {} dataset".format(dataset_path))
     tree_model = Decision_Tree_Classifier.Tree(dataset, max_depth = depth)
     tree_model.show()

  if mode == "metrics":
     print("Updating Metrics in the figures folder")
     benchmark = TB.Model_Comparison_TB(dataset, 10, depth, Decision_Tree_Classifier.Tree, Other_Classifiers.RandomClassifier, Other_Classifiers.NNClassifier)
     benchmark.all_metrics()
     
  if mode == "depth_benchmark":
    hyperParam = TB.Depth_Hyperparameter_Tuning(dataset, Decision_Tree_Classifier.Tree)
    hyperParam.run()


  
  


  
  #benchmark.precision()
  #benchmark.plotNorm()
  
  #test = RandomClassifier.NNClassifier(dataset)
  #print((test.confusion_constructor()[2]))
  
  #hyperParam = TB.Depth_Hyperparameter_Tuning(dataset, Decision_Tree_Classifier.Tree)
  #hyperParam.run()

if __name__ == '__main__':
    main()

