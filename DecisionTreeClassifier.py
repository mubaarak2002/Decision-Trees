import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.widgets import Cursor
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

# tuning parameters
test_proportion = 0.1
split_resolution = 10
depth = 5
mode = "build"



test_data = np.array([

[-68,	-57,	-61,	-65,	-71,	-85,	-85,	1],
[-63,	-60,  -60,	-67,	-76,	-85,	-84,	1],
[-61,	-60,	-68,	-62,	-77,	-90,	-80,	1],
[-63,	-65,	-60,	-63,	-77,	-81,	-87,	1],
[-64,	-55,	-63,	-66,	-76,	-88,	-83,	1],
[-65,	-61,	-65,	-67,	-69,	-87,	-84,	1],
[-61,	-63,	-58,	-66,	-74,	-87,	-82,	1],
[-65,	-60,	-59,  -63,	-76,	-86,	-82,	1],
[-62,	-60,  -66,	-68,	-80,	-86,	-91,	1],
[-67,	-61,	-62,	-67,	-77,	-83,	-91,	1],
[-65,	-59,	-61,	-67,	-72,	-86,	-81,	1],
[-63,	-57,	-61,	-65,	-73,	-84,	-84,	1]



])



class Tree:
  '''
  Construtor:
    input: takes in the entire data set of WIFI points
    Class Attributes:
    1) self.start_node (the start node for the tree)
    2) self.training_data (the training data and labels)
    3) self.test (the test data and labels)
    4) self.all_labels (all labels so indexing stays consistant)

    Class Methods:
    1) tree.evaluate(test_data): run the test data, and give a summary of the test in a table
    2) tree.show(): plot the tree and all the thresholds (might be dificult but can give a go)
    3) tree.confusion_matrix(): plot the confusion matrix of the tree
    add any other visualisation functions
  '''

  def __init__(self, dataset, max_depth=depth):
    x_values, categories = extract_categories(dataset)
    self.labels = categories
    x_train, x_test, y_train, y_test = split_dataset(dataset, categories, test_proportion)
    self.x_train = x_train
    self.x_test = x_test
    self.y_train = y_train
    self.y_test = y_test
    self.depth = max_depth

    self.head = Node(x_train, 0)

  def evaluate(self, test_data):
    outlist = []
    for sample in test_data:
      outlist.append(self.head.evaluate(sample))
    return outlist


  def run_test(self):
    results = self.evaluate(self.x_train)

    sum = 0
    for i in range(len(results)):
      if results[i] == self.y_test[i]:
        sum += 1

    if mode == "debug": print("The accuracy for this test was {}".format(sum / len(results)))

  def show(self):
    '''
    shows a tree using a matplotlib frame with hoverable elements to see what is currently being split on

    Credit to the online tutorial for aiding in this project: https://matplotlib.org/matplotblog/posts/mpl-for-making-diagrams/ and https://matplotlib.org/stable/gallery/text_labels_and_annotations/demo_annotation_box.html

    '''

    #frame attributes
    w = 16
    h = 9
    border_width = 4
    border_colour = "black"
    colour = "lightgray"

    #finding how deep and how wide the tree is:
    left_counter = 0
    right_counter = 0
    max_tree_depth = 0

    root = self.head
    while True:

      if root.left is not None:
        left_counter += 1

        if root.depth > max_tree_depth:
          max_tree_depth = root.depth

        root = root.left
      else:
        break

    root = self.head
    while True:
      if root.right is not None:
        right_counter += 1

        if root.depth > max_tree_depth:
          max_tree_depth = root.depth

        root = root.right
      else:
        break

    #need to start on 1, not 0, so add one
    max_tree_depth += 1
    print("Tree found with max depth {a}, spanning {b} left entries and {c} right entries".format(a=max_tree_depth, b=left_counter, c=right_counter))

    total_span = left_counter + right_counter

    horisontal_bins = total_span
    vertical_bins = max_tree_depth
    print("\nNeeds {a} vertical bins and {b} horisontal bins\n".format(a=vertical_bins, b=horisontal_bins))

    if(1): #used to fold this down for ease of coding
      #Frame initialisation
      fig = plt.figure(figsize= (w, h) )

      ax = fig.add_axes( (0, 0, 1, 1) )

      ax.set_xlim(0, w)
      ax.set_ylim(0, h)

      ax.tick_params(bottom=False, top=False, left=False, right=False)
      ax.tick_params(labelbottom=False, labeltop=False, labelleft=False, labelright=False)


      ax.spines["top"].set_color(border_colour)
      ax.spines["bottom"].set_color(border_colour)
      ax.spines["left"].set_color(border_colour)
      ax.spines["right"].set_color(border_colour)
      ax.spines["top"].set_linewidth(border_width)
      ax.spines["bottom"].set_linewidth(border_width)
      ax.spines["left"].set_linewidth(border_width)
      ax.spines["right"].set_linewidth(border_width)

      ax.set_facecolor(colour)

    #Tree Construction
    if horisontal_bins > vertical_bins:
      radii = w / horisontal_bins
    else:
      radii = h / vertical_bins

    #for debugging
    radii = 0.2

    centres = []
    texts = {}
    arrows = {}

    root = self.head
    midpoint = w/2
    height_index = 0



    #the root node
    root_centre = ( (midpoint) , ( h *  ( ( vertical_bins - height_index ) / vertical_bins ) - radii  ) )
    centres.append( root_centre )
    texts[root_centre] = root.print()


    def plotTree(tree, min, max, bin_depth):
      '''
      Takes the current remaining portion of the screen and splits it into two, then completes the rest of the tree in those halves of the screen
      '''
      dot_centre = ( (max + min) / 2 , ( h *  ( ( vertical_bins - bin_depth ) / vertical_bins ) + radii  ) )
      centres.append( dot_centre )
      texts[dot_centre] = tree.print()
      if tree.left is not None:
        L = plotTree(tree.left, min, (max + min) / 2, bin_depth + 1)

        R = plotTree(tree.right, (max + min) / 2, max, bin_depth + 1)
        arrows[dot_centre] = [L,R]


      return dot_centre




    L = plotTree(root.left, 0, midpoint, height_index + 1)
    R = plotTree(root.right, midpoint, w, height_index + 1)
    arrows[root_centre] = [L,R]
    print(arrows)

    print("\n")
    print("radius is: {}".format(radii))
    print("\n\n")
    print(texts)
    print("\n")

    #draw all the nodes
    for index, centre in enumerate(centres):
      x, y = centre
      angle = np.linspace(0, 2*np.pi, 100)
      ax.plot(

        x + radii * np.cos(angle),
        y + radii * np.sin(angle),
        color = "midnightblue"

      )

    #draw all the arrows
    for start, ends in arrows.items():
      for end in ends:
        ax.annotate( 
                    "",
                    (end[0], end[1] + radii),
                    (start[0] , start[1] - radii),
                    arrowprops=dict(arrowstyle = "-|>")
                    )

    if(0):      
      for point, text in texts.items():
        cursor = Cursor(plt.gca(), useblit=True, color='red', linewidth=1)
        annotation = AnnotationBbox(OffsetImage(text, zoom=2), point,
                              xybox=(-30, 30),
                              boxcoords="offset points",
                              arrowprops=dict(arrowstyle="->"))

        plt.gca().add_artist(annotation)    


    fig.savefig("current_tree.png", dpi=300)
    plt.show()

class Node:
  '''
  Constructror:
    input: the current data set that is still undetermined after all prior branches, atributes, and other stuff
  Class Attributes:
    1) self.value (= None if is a branch node and = label if end node)
    2) self.attribute (The attribute to test (number of index of globally available label list in master tree))
    3) self.threshold (The value determining if you should split or not)
    4) self.less_than (= the next Node if test attribute is less than value (= None if self.value != None)) 
    5) self.greater_eq (= the next node if test attribute is geq than value (= None if self.value != None))
    6) self.All_labels (all labels in a list so indexing stays consistant)


  Constructor Code:
  1) Take all available data and collect available labels
    a) if num labels == 1: create end node w/ less_than
    b) if depth == k: create end node with most dominant value
  2) Run find_split() and best_split_in_feature() to determine the split
  3) assign found values into each Class Attribute

  Class Method node.run_test():
    input: current Sample
    This is the method run when in runtime

    if leaf:
      return self.value
    else:
      run condition and then pass the sample onto the next node

  '''
  """Node constructor"""
  def __init__(self, dataset, TreeDepth = 0):

    self.depth = TreeDepth
    x_values, categories = extract_categories(dataset)
    [labels, counts] = np.unique(categories, return_counts=True)
    if(len(labels) == 1):
      self.room = labels[0]
      self.left = None
      self.right = None
      return
    elif(TreeDepth == depth):
      majority = labels[np.argmax(counts)]
      self.room = majority
      self.left = None
      self.right = None
      return
    else:

      self.room = None
      dataset_a, dataset_b, split, feature = find_split(dataset)
      self.left = Node(dataset_a, TreeDepth + 1)
      self.right = Node(dataset_b, TreeDepth + 1)

      #feature is the room number
      self.feature = feature
      #split is the value its split on
      self.split = split

  def print(self):
    if self.room is not None:
      return "We are in room {}".format(self.room)
    else:
      return "Split on Feature {a} less than {b}".format(a=self.feature, b=round(self.split, 2))

  def evaluate(self, data):
    '''takes a single data entry, and then evaluates it based on the node's current splitting criteria'''
    if self.room is not None:
      return self.room

    if data[self.feature] < self.split:
      return self.left.evaluate(data)
    else:
      return self.right.evaluate(data)

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

def calc_entropy(occurences):
  """Calculates entropy of numpy array elements"""
  total = np.sum(occurences)
  entropy = np.sum(
      -(occurences / total) *
      np.log2(occurences / total,
              where=(occurences != 0)))  # does not take log of zero
  return entropy



def decision_tree_learning(training_dataset, depth):

  dataset, categories = extract_categories(training_dataset)
  num_available_labels = np.unique(categories)

  if len(num_available_labels) == 1:
    '''
    TODO: If the number of labels avaiable in this node is all the same, 
    then the recursion must be ended,and this must turn into a leaf
    '''

    pass





  pass

def find_split(dataset):
  """
  Finds the best split for the dataset by finding the best split for
  each feature and choosing the one with the highest information gain,
  then selects highest information gain out of all the features. Returns
  2 datasets after the split and the details of the split

  """
  if mode == "debug": print(dataset)
  height, width = np.shape(dataset)
  categories = dataset[:height, -1:].astype(int).flatten()
  all_splits = []
  # finding best split for each feature
  for i in range(width - 1):
    column = dataset[:, i]
    all_splits.append(
        best_split_in_feature(column, categories, split_resolution))

  # finds feature with highest information gain
  best_split = [1000]
  for i in range(len(all_splits)):
    # information gain is entropy_before - entropy_after
    # largest information gain is smallest entropy_after
    if all_splits[i][0] < best_split[0]:
      best_split = all_splits[i]
      best_split.append(i)

  # split dataset according to best_split
  split = best_split[1]
  feature = best_split[2]
  dataset_a = dataset[dataset[:, feature] < split, :]
  dataset_b = dataset[dataset[:, feature] >= split, :]
  #if mode == "debug": print(split, dataset_a, dataset_b)
  #if mode == "debug": print("------")
  #if mode == "debug": print(feature, split)
  return dataset_a, dataset_b, split, feature


def best_split_in_feature(column, categories, resolution):
  """
  finds best split of a feature
  generates 'resolution' number of splits and returns the one with
  highest information gain

  """
  height = len(column)
  data_range = np.amin(column) - np.amax(column)
  step_size = data_range / resolution
  split_point = np.amax(column)
  entropy_list = []
  split_point_list = []
  for j in range(resolution):
    # index 0 = room 1 occurences, index 1 = room 2 occurences etc.
    occurences_above = [0, 0, 0, 0]
    occurences_below = [0, 0, 0, 0]

    # counts number of times each feature is above the split point
    # and is below the split point
    for k in range(height):
      if column[k] >= split_point:
        occurences_above[categories[k] - 1] += 1
      else:
        occurences_below[categories[k] - 1] += 1

    above = np.array(occurences_above)
    below = np.array(occurences_below)
    entropy = calc_entropy(above) + calc_entropy(below)
    entropy_list.append(entropy)
    split_point_list.append(split_point)
    split_point += step_size

  # finds split with highest information gain
  lowest = [1000, 0]
  for i in range(len(entropy_list)):
    # information gain is entropy_before - entropy_after
    # largest information gain is smallest entropy_after
    if entropy_list[i] < lowest[0]:
      lowest[0] = entropy_list[i]
      lowest[1] = split_point_list[i]
  return lowest


if __name__ == "__main__":
  """
  Do nothing for now if this is run directly
  
  """
  pass