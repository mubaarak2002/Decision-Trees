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
  

  if mode == "show_tree":
     print("Tree depth:", depth)
     print("Showing the Tree for the {} dataset".format(dataset_path))
     print("Click on a node to view its decision properties")
     tree_model = Decision_Tree_Classifier.Tree(dataset, max_depth = depth)
     tree_model.show()

  if mode == "metrics":
     print("Tree depth:", depth)
     print("Updating Metrics in the figures folder")
     benchmark = TB.Model_Comparison_TB(dataset, 10, depth, Decision_Tree_Classifier.Tree, Other_Classifiers.RandomClassifier, Other_Classifiers.NNClassifier)
     benchmark.all_metrics()
     
  if mode == "depth_benchmark":
    print("Tree depth ignored, running range [4,70]")
    if dataset_path == "WIFI_db/noisy_dataset.txt":
      hyperParam = TB.Depth_Hyperparameter_Tuning(dataset, Decision_Tree_Classifier.Tree, name = "noisy")
    else:
      hyperParam = TB.Depth_Hyperparameter_Tuning(dataset, Decision_Tree_Classifier.Tree, name = "clean")
      
    hyperParam.run()
    
  if mode == "norm":
    benchmark = TB.Model_Comparison_TB(dataset, 10, depth, Decision_Tree_Classifier.Tree, Other_Classifiers.RandomClassifier, Other_Classifiers.NNClassifier)
    benchmark.plotNorm()
  


if __name__ == '__main__':
    main()

