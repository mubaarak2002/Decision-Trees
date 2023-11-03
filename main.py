import Decision_Tree_Classifier
import Other_Classifiers
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

  if depth == "tune":
    if dataset_path == "WIFI_db/noisy_dataset.txt":
      hyperParam = TB.Depth_Hyperparameter_Tuning(dataset, Decision_Tree_Classifier.Tree, name = "noisy")
    else:
      hyperParam = TB.Depth_Hyperparameter_Tuning(dataset, Decision_Tree_Classifier.Tree, name = "clean")
      
    depth = hyperParam.run()
    
  if mode == "show_tree":
     print("Tree depth:", depth)
     print("Showing the Tree for the {} dataset".format(dataset_path))
     print("Click on a node to view its decision properties")
     tree_model = Decision_Tree_Classifier.Tree(dataset, max_depth = depth)
     tree_model.show()

  elif mode == "metrics":
     print("Tree depth:", depth)
     print("Updating Metrics in the figures folder")
     benchmark = TB.Model_Comparison_TB(dataset, 10, depth, Decision_Tree_Classifier.Tree, Other_Classifiers.RandomClassifier, Other_Classifiers.NNClassifier)
     benchmark.all_metrics()
     print("All Metric Figures updated in the ./figures directory")
    
  elif mode == "normal":
    print("Generating Normal Distribution ...")
    benchmark = TB.Model_Comparison_TB(dataset, 10, depth, Decision_Tree_Classifier.Tree, Other_Classifiers.RandomClassifier, Other_Classifiers.NNClassifier)
    benchmark.plotNorm()
  else:
    print("Invalid operating_mode")


if __name__ == '__main__':
    main()

