import random
import numpy as np


test_proportion = 0.1
split_resolution = 10


class RandomClassifier:
  def __init__(self, dataset):
    
    self.unique_labels = []
    train, test = split_train_test(dataset, test_proportion)
    self.train = train
    self.test = test
    self.fit(dataset)    

  def fit(self, dataset):
    x_values, categories = extract_categories(dataset)
    
    labels, ymap = np.unique(categories, return_inverse=True)
    
    #labels = [1, 2, 3, 4]

    self.unique_labels = list((labels))

  def predict(self, dataset):
    random_indices = np.random.choice(self.unique_labels, len(dataset))
    return random_indices 
  
  def evaluate_internal(self):
 
      out = []

      
      for index, result in enumerate(np.random.choice(self.unique_labels, len(self.test))):
        if self.test[index][-1] == result:
            out.append(1)
        else:
            out.append(0)
      
      return out
  
  def name(self): return "Random Classifier"
  
  
  
  
  
def extract_categories(dataset):
  height, width = np.shape(dataset)
  no_of_attrtibutes = width - 1
  categories = dataset[:height, -1:].astype(int)
  dataset = dataset[:height, :no_of_attrtibutes]
  return dataset, categories

def split_train_test(dataset, test_proportion):
  """Splits dataset into train and test sets, according to test_proportion"""
  train = []
  test = []
  for i in range(len(dataset)):
    choice_test = random.random() < test_proportion
    if choice_test:
      test.append(dataset[i])
    else:
      train.append(dataset[i])

  train = np.array(train)
  test = np.array(test)
  return (train, test)

if __name__ == "__main__":
  """
  Do nothing for now if this is run directly

  """
  pass