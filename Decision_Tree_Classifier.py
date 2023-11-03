import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.widgets import Cursor
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.offsetbox import OffsetBox, AnnotationBbox, TextArea
from matplotlib.text import OffsetFrom

# tuning parameters

depth = 50
mode = "build"
figure_root = "./figures/"

#needs to be declared in global scope
annotations = []

# hide - "RuntimeWarning: invalid value encountered in divide"
np.seterr(divide='ignore', invalid='ignore')

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

  def __init__(self, dataset, max_depth, train_given=None, test_given=None, split_resolution=20, split_threshold=0):
    x_values, categories = extract_categories(dataset)
    self.labels = categories

    global depth
    depth = max_depth

    #allows you to give train and test data, or generate it automatically
    if test_given is None and train_given is None:
      train, test = split_train_test(dataset)
      self.train = train
      self.test = test
    else:
      self.train = train_given
      self.test = test_given

    self.split_resolution = split_resolution
    self.depth = max_depth
    self.tree_name = "Decision Tree Classifier"


    self.head = Node(self.train, 0, split_resolution=split_resolution, split_threshold=0) #unused as tree is not tunable
    self.head.clean()

  def evaluate(self, test_data):
    '''Used for when test data is being given from an outside source'''
    outlist = []
    for sample in test_data:
      outlist.append(self.head.evaluate(sample))
    #if mode == "debug": print(outlist)
    return outlist
      
  def evaluate_internal(self, mode=0):
    '''does the same as above, but uses the test data that is generated in the constructor'''
    
    results = self.evaluate(self.test)
    out = []
    for i in range(len(results)):


      if mode == 0:
        if results[i] == self.test[i][-1]:
          out.append(1)
        else:
          out.append(0)
      elif mode == 1:
        out.append(results[i])
    
    return out


  def confusion_constructor(self): return (np.unique(self.labels), np.array(self.test)[:,-1], self.evaluate_internal(mode=1))

  def name(self): return self.tree_name
  
  def run_test(self):
    results = self.evaluate(self.test)

    sum = 0
    for i in range(len(results)):
      if results[i] == self.test[i][-1]:
        sum += 1

    if mode == "debug": print("The accuracy for this test was {}".format(sum / len(results)))
    print("The accuracy for this test was {}".format(sum / len(results)))

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

    def find_max_depth(curr_node):
      
      if curr_node.room is not None:
        return curr_node.depth
      else:
        L_depth = find_max_depth(curr_node.left)
        R_depth = find_max_depth(curr_node.right)
        
        return L_depth if (L_depth > R_depth) else R_depth

    max_tree_depth = find_max_depth(self.head)
    
    
    root = self.head
    while True:

      if root.left is not None:
        left_counter += 1
        root = root.left
      else:
        break

    root = self.head
    while True:
      if root.right is not None:
        right_counter += 1
        root = root.right
      else:
        break

    #need to start on 1, not 0, so add one
    max_tree_depth += 1
    if mode == "debug": print("Tree found with max depth {a}, spanning {b} left entries and {c} right entries".format(a=max_tree_depth, b=left_counter, c=right_counter))

    total_span = left_counter + right_counter

    horisontal_bins = total_span
    vertical_bins = max_tree_depth
    if mode == "debug": print("\nNeeds {a} vertical bins and {b} horisontal bins\n".format(a=vertical_bins, b=horisontal_bins))

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
      radii = w / horisontal_bins - 0.2
    else:
      radii = h / vertical_bins - 0.2

    #for debugging
    radii = 0.06

    centres = []
    texts = {}
    arrows = {}
    
 

    root = self.head

    if left_counter == right_counter:
      midpoint = w/2
    else:
      midpoint = w * (  (left_counter) / (left_counter + right_counter)  )
    
    height_index = 0
    top_offset = 0.1


    #the root node
    root_centre = ( (midpoint) , ( h *  ( ( vertical_bins - height_index ) / vertical_bins ) - radii - top_offset * 2 ) )
    centres.append( root_centre )
    texts[root_centre] = root.print()


    def plotTree(tree, min, max, bin_depth):
      '''
      Takes the current remaining portion of the screen and splits it into two, then completes the rest of the tree in those halves of the screen
      '''
      
      dot_centre = ( (max + min) / 2 , ( h *  ( ( vertical_bins - bin_depth ) / vertical_bins ) + radii + top_offset  ) )
      centres.append( dot_centre )
      texts[dot_centre] = tree.print()
      
      #defaults: width_scalar = 4, tolerance = 1
      width_scalar = 14
      tolerance = 10
      
      new_midpoint = (max + min) / 2
      if tree.room is None:
        
        #There tree has enough space to grow
        if new_midpoint - min > radii * tolerance and max - new_midpoint > (radii * tolerance) ** 0.5:
          L = plotTree(tree.left, min, new_midpoint, bin_depth + 1)
          R = plotTree(tree.right, new_midpoint, max, bin_depth + 1)
          
        #The Tree does not have space to grow, so some ammendments need to be made
        else:
          #The new nodes will intersect with each other 
          if new_midpoint - width_scalar * radii * tolerance > 0:
            L = plotTree(tree.left, new_midpoint - width_scalar * radii, new_midpoint, bin_depth + 1)
            R = plotTree(tree.right, new_midpoint, new_midpoint + width_scalar * radii, bin_depth + 1)

          #The Tree is growing into the left hand side of the wall
          else:
            L = plotTree(tree.left, radii, new_midpoint + width_scalar * radii, bin_depth + 1)
            R = plotTree(tree.right, new_midpoint + width_scalar * radii, new_midpoint + 2 * width_scalar * radii, bin_depth + 1)




        arrows[dot_centre] = [L,R]
      
      
      return dot_centre
      

    
    L = plotTree(root.left, 0.2, midpoint, height_index + 1)
    R = plotTree(root.right, midpoint, w - 0.2, height_index + 1)
    arrows[root_centre] = [L,R]

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
      
      def display_text(event):
        
        
        global annotations
            
        for note in annotations:
          note.remove()
        annotations = []
          
        x, y = event.xdata, event.ydata
        text = "Click on a node to view the node"
        for (t_x, t_y), nodeText in texts.items():
          dist =  (  (t_x - x) ** 2 + (t_y - y) ** 2  ) ** 0.5
          if dist < radii:
            text = nodeText
            break
            
      
        offsetbox = TextArea(text)
 
        ab = AnnotationBbox(offsetbox, (x, y), xycoords='data', boxcoords="offset points",)
        ax.add_artist(ab)
        annotations.append(ab)
        plt.draw()
      
    if(1):      
      

            
      for (t_x, t_y), nodeText in texts.items():
        offsetbox = TextArea(nodeText)
        ab = AnnotationBbox(offsetbox, (t_x, t_y), xycoords='data', boxcoords="offset points",)
        ax.add_artist(ab)
        annotations.append(ab)
        plt.draw()



    fig.savefig(figure_root + "current_tree.png", dpi=300)
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
  def __init__(self, dataset, TreeDepth=0,  split_resolution=20, split_threshold=0):

    
    self.depth = TreeDepth
    x_values, categories = extract_categories(dataset)
    [labels, counts] = np.unique(categories, return_counts=True)
    
    if(len(labels) == 1):
      self.room = labels[0]
      self.left = None
      self.right = None
      return
    elif(TreeDepth == int(depth)):
      majority = labels[np.argmax(counts)] # fails here
      self.room = majority
      self.left = None
      self.right = None
      return
    else:
      self.room = None
      dataset_a, dataset_b, split, feature = find_split(dataset, split_resolution, split_threshold=0) #unused as tree is not being tuned
      self.left = Node(dataset_a, TreeDepth + 1, split_threshold=split_threshold)
      self.right = Node(dataset_b, TreeDepth + 1, split_threshold=split_threshold)
      
      #feature is the room number
      self.feature = feature
      #split is the value its split on
      self.split = split



  def clean(self):
    if(self.left.room is None):
      self.left.clean()
    if(self.right.room is None):
      self.right.clean()
    if((self.left.room is not None) and self.left.room == self.right.room):
      self.room = self.left.room
      self.left = None
      self.right = None

  def print(self):
    if self.room is not None:
      return "Room {}".format(self.room)
    else:
      return "Feature {a} < {b}".format(a=self.feature, b=round(self.split, 2))

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


def split_train_test(dataset, test_proportion=0.1):
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


def split_folds(n_instances):
    folds = []
    train = []
    test = []
    shuffled_indices = np.random.permutation(n_instances)
    n_folds = 10
    split_indices = np.array_split(shuffled_indices, n_folds)
    for i in range(n_folds):
        test = split_indices[i]
        train = np.hstack(split_indices[:i] + split_indices[i+1:])

        folds.append([train, test])

    return folds


def calc_entropy(occurences):
  """Calculates entropy of numpy array elements"""
  total = np.sum(occurences)
  entropy = np.sum(
      -(occurences / total) *
      np.log2(occurences / total,
              where=(occurences != 0)))  # does not take log of zero
  return entropy


def decision_tree_learning(training_dataset, depth):

  ##TODO: Make this the generate a tree
  pass


def find_split(dataset, split_resolution=20, split_threshold=0):
  """
  Finds the best split for the dataset by finding the best split for
  each feature and choosing the one with the highest information gain,
  then selects highest information gain out of all the features. Returns
  2 datasets after the split and the details of the split
  
  split_threshold is a float between 0 and 1, and defines "at least split_threshold % of the data 
  must be present in the split to form a split" This is to help prevent overfitting of the data.
  e.g a split threshold of 0 means a split can be only 1 entry if needed, however a split threshold of
  0.1 means that at least 10% of the data must be in the split in order to make the split

  """
  #print("ode Inputted Threshold, ", split_threshold)
  if mode == "debug": print(dataset)
  height, width = np.shape(dataset)
  categories = dataset[:height, -1:].astype(int).flatten()
  all_splits = []
  # finding best split for each feature
  for i in range(width - 1):
    column = dataset[:, i]
    toAdd = best_split_in_feature(column, categories, split_resolution, split_threshold=0) #Set to zero as this is a non-tunable tree
    #print(toAdd)
    if toAdd is not None:
      all_splits.append(toAdd)

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


def best_split_in_feature(column, categories, resolution, split_threshold=0):
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
    #print("Above: {a}, Below: {b}".format(a=sum(above), b=sum(below)))

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
