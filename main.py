import numpy as np
import random

# tuning parameters
test_proportion = 0.1
split_resolution = 10

def main():

  clean_dataset = np.loadtxt("WIFI_db/clean_dataset.txt")
  noisy_dataset = np.loadtxt("WIFI_db/noisy_dataset.txt")

  dataset, categories = extract_categories(clean_dataset)
  x_train,x_test,y_train,y_test = split_dataset(dataset, categories, test_proportion)
  print("Train set:", len(x_train), "Test set:", len(x_test))

  find_split(clean_dataset)


def extract_categories(dataset):
  height, width = np.shape(dataset)
  no_of_attrtibutes = width - 1
  categories = dataset[:height, -1:].astype(int)
  dataset = dataset[:height, :no_of_attrtibutes]
  return dataset, categories

def split_dataset(x, y, test_proportion):
  x_train = []
  x_test = []
  y_train = []
  y_test = []
  for i in range(len(x)):
    choice_test = random.random() < test_proportion
    if choice_test:
      x_test.append(x[i])
      y_test.append(y[i])
    else:
      x_train.append(x[i])
      y_train.append(y[i])

  x_train = np.array(x_train)
  x_test = np.array(x_test)
  y_train = np.array(y_train)
  y_test = np.array(y_test)
  return(x_train,x_test,y_train,y_test)
  
def calc_entropy(occurences):
  total = np.sum(occurences)
  entropy = np.sum(-(occurences / total) * np.log2(occurences / total, where=(occurences!=0))) # does not take log of zero
  return entropy

def decision_tree_learning(training_dataset, depth):
  return

def find_split(dataset):
  height, width = np.shape(dataset)
  categories = dataset[:height, -1:].astype(int).flatten()
  all_splits = []
  for i in range(width - 1):
    column = dataset[:, i]
    all_splits.append(best_split_in_feature(column, categories, split_resolution))

  best_split = [1000]
  for i in range(len(all_splits)):
    # information gain is entropy_before - entropy_after
    # largest information gain is smallest entropy_after
    if all_splits[i][0] < best_split[0]:
      best_split = all_splits[i]
      best_split.append(i)

  split = best_split[1]
  feature = best_split[2]
  dataset_a = dataset[dataset[:, feature] < split, :]
  dataset_b = dataset[dataset[:, feature] > split, :]
  print(split,dataset_a, dataset_b)
  return dataset_a, dataset_b, split, feature


def best_split_in_feature(column, categories, resolution):
  height = len(column)
  data_range = np.amin(column) - np.amax(column)
  step_size = data_range / resolution
  split_point = np.amax(column)
  entropy_list = []
  split_point_list = []
  for j in range(resolution):
    # index 0 = room 1 occurences, index 1 = room 2 occurences etc.
    occurences_above = [0,0,0,0]
    occurences_below = [0,0,0,0]
    for k in range(height):
      if column[k] >= split_point:
        occurences_above[categories[k]-1] += 1
      else:
        occurences_below[categories[k]-1] += 1
    above = np.array(occurences_above)
    below = np.array(occurences_below)
    entropy = calc_entropy(above) + calc_entropy(below)
    entropy_list.append(entropy)
    split_point_list.append(split_point)
    split_point += step_size
  
  lowest = [1000,0]
  for i in range(len(entropy_list)):
    # information gain is entropy_before - entropy_after
    # largest information gain is smallest entropy_after
    if entropy_list[i] < lowest[0]:
      lowest[0] = entropy_list[i]
      lowest[1] = split_point_list[i]
  return lowest

main()