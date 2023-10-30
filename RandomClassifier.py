import random
import numpy as np


test_proportion = 0.1
split_resolution = 10


class RandomClassifier:
  def __init__(self, dataset, train_given=None, test_given=None):
    
    x_values, categories = extract_categories(dataset)
    self.labels = categories

    #allows you to give train and test data, or generate it automatically
    if test is None and train is None:
      train, test = split_train_test(dataset, test_proportion)
      self.train = train
      self.test = test
    else:
      self.train = train_given
      self.test = test_given


    self.fit(dataset)
    self.tree_name = "Random Classifier"

  def fit(self, dataset):
    x_values, categories = extract_categories(dataset)
    
    labels, ymap = np.unique(categories, return_inverse=True)
    
    #labels = [1, 2, 3, 4]

    self.unique_labels = list((labels))

  def predict(self, dataset):
    random_indices = np.random.choice(self.unique_labels, len(dataset))
    return random_indices 
  
  def evaluate_internal(self, mode=0):
      '''
      Mode of 0 returns an array of 1's and 0's corresponding to correct and incorrect guesses
      Mode of 1 returns the confusion matrix form, which is the guesses
      '''

      out = []
      for index, result in enumerate(np.random.choice(self.unique_labels, len(self.test))):

        if mode == 0:
          if self.test[index][-1] == result:
              out.append(1)
          else:
              out.append(0)
        elif mode == 1:
          out.append(result)
      
      return out
  
  def name(self): return self.tree_name

  def confusion_constructor(self): return (np.unique(self.labels), np.array(self.test)[:,-1], self.evaluate_internal(mode=1))

  
  

class NNClassifier:
  def __init__(self, dataset, train_given=None, test_given=None):
    x_values, categories = extract_categories(dataset)
    self.labels = categories

    #allows you to give train and test data, or generate it automatically
    if test is None and train is None:
      train, test = split_train_test(dataset, test_proportion)
      self.train = train
      self.test = test
    else:
      self.train = train_given
      self.test = test_given


    self.tree_name = "Nearest Neighbour Classifier"

  def evaluate_internal(self, mode=0):
    '''
    Mode of 0 returns an array of 1's and 0's corresponding to correct and incorrect guesses
    Mode of 1 returns the confusion matrix form, which is the guesses
    '''
    out = []

    for test_sample in self.test:
      solution = test_sample[-1]

      closest_index = None
      closest_dist = None

      for train_index, train_sample in enumerate(self.train):
        
        dist_sum = 0
        for feature_index in range( len(train_sample) - 1 ):

          #print("Comparing {a} and {b}".format(a=test_sample[feature_index], b=train_sample[feature_index]))
          dist_sum += ( test_sample[feature_index] - train_sample[feature_index] ) ** 2

        dist = ( dist_sum ) ** 0.5
        

        if closest_index is None:
          closest_index = train_index
          closest_dist = dist
        else:
          #print("Comparing current Distance {a} and new sample {b}".format(a=closest_dist, b=dist))
          if dist < closest_dist:
            closest_index = train_index
            closest_dist = dist



      #print("Found closest match to be {a} at index {b}".format(a=self.train[closest_index], b=closest_index))
      if mode == 0:
        if self.train[closest_index][-1] == solution:
          out.append(1)
        else:
          out.append(0)
      elif mode == 1:
        out.append(self.train[closest_index][-1])

    return out
  
  
  def name(self): return self.tree_name
  
  def confusion_constructor(self): return (np.unique(self.labels), np.array(self.test)[:,-1], self.evaluate_internal(mode=1))

    







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