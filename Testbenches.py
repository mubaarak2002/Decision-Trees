import numpy as np
import matplotlib.pyplot as plt
from matplotlib.table import Table
import Decision_Tree_Classifier
import random
import math
import multiprocessing as mp

class Model_Comparison_TB():
    '''
    takes in as many classes as you want, and compares their performances in a plot
    Criteria to run test:
    
    constructor input
    dataset: The dataset to be split into test/train into 
    num_trials: The number of trials to run to generate the distribution of each testing method
    
    all other constructor inputs are just the class of every ML model you want to compare against eachother
        - All constructors of the different testers must take only a dataset as input, and be able to (internlly) in the constructor split it into a test/train and run the model
        - All test benches must have an evaluate_internal() function to return the test and return a list of either 1 or 0, denoting a success/failiure for every test case
    '''
  
    def __init__(self, dataset, num_trials=10, tree_depth=10, *args):
        
        Models = []
        for testMethod in args:
            Models.append(testMethod)
        means = {}
        all_samples = {}
        for index, model in enumerate(Models):
            for trial in range(num_trials):       
                if model == Decision_Tree_Classifier.Tree:
                    sample = model(dataset, tree_depth)
                else:
                    sample = model(dataset)
                results = sample.evaluate_internal()

                if sample.name() not in list(all_samples.keys()):
                    all_samples[sample.name()] = []
                    means[sample.name()] = []
                else:
                    all_samples[sample.name()].append(sample)
                
                success_rate = 0
                for i in results:
                    if i == 1:
                        success_rate += 1
                    
                means[sample.name()].append(success_rate / len(results))
                    
        self.num_folds = num_trials
        self.tree_depth = tree_depth             
        self.dataset = dataset 
        self.means = means
        self.models = Models
        self.all_samples = all_samples
        self.confusion = {}
        self.confusion_matrix()


    def plotNorm(self):
        
        plt.figure(figsize=(8, 6))
        for method, results in self.means.items():
            
            mean = np.mean(results)
            std = np.std(results)

            
            x = np.linspace(mean - 4 * std,  mean + 4 * std, 100)
            
            # Calculate the PDF values for the normal distribution
            pdf = 1 / (std * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mean) / std) ** 2)
            
            plt.plot(x, pdf, label=f"{method}, with Mean={round(mean, 2)}")
            
        plt.title("Performance Distribution of Multiple Machine Learning Models")
        plt.xlabel("Accuracy")
        plt.ylabel("Probability Concentration")
        plt.legend()
        plt.grid(True)
        plt.show()
            
    def confusion_matrix(self, train_given=None, test_given=None, num_folds=10):
        
        
        #TODO: this did not need to use indexes, probably replace later?
        newOrder =list(range(len(self.dataset)))
        random.shuffle(newOrder)
        confusion = {}
        
        for iteration_num in range(num_folds):
            test_start = iteration_num * math.floor( (len(self.dataset) / num_folds) )
            test_finish = (iteration_num + 1) * math.floor( (len(self.dataset) / num_folds) )
            test_set = []
            train_set = []
            
            for i in range(test_start, test_finish):
                test_set.append( self.dataset[ newOrder[i] ] )
                
            if test_start == 0:
                
                for i in range(test_finish, len(self.dataset)):
                    train_set.append( self.dataset[ newOrder[i] ] )
            
            else:
                for i in range(0, test_start):
                    train_set.append( self.dataset[newOrder[i]] )
                for i in range(test_finish, len(self.dataset)):
                    train_set.append(  self.dataset[newOrder[i]])
        
        
        
    
            for model in self.models:
                
                if model == Decision_Tree_Classifier.Tree:
                    specific_model = model(self.dataset, self.tree_depth, train_given = np.array(train_set), test_given = np.array(test_set))
                else:
                    specific_model = model(self.dataset, train_given = train_set, test_given = test_set)

                (labels, actual, guess) = specific_model.confusion_constructor()
                
                
                name = specific_model.name()
                if name not in list(confusion.keys()):
                    confusion[name] = (confusion_matrix(actual, guess, class_labels=labels))
                    
                else:
                    toAdd = (confusion_matrix(actual, guess, class_labels=labels))
                    for row_index, row in enumerate(toAdd):
                        
                        for col_index, cell in enumerate(row):
                            
                            confusion[name][row_index][col_index] += cell
                            #confusion[name][row_index][col_index] += cell / num_folds

        if(1):
            for name, matrix in confusion.items():     
                for rowIndex, row in enumerate(matrix):
                    for colIndex, cell in enumerate(row):
                            
                        confusion[name][rowIndex][colIndex] = cell / num_folds
                
                
        self.confusion = confusion
        #print(self.confusion)
        return self.confusion
        
    def accuracy(self): 
       
       self.accuracies = {}

       for name, matrix in self.confusion.items():
          self.accuracies[name] = accuracy_f(matrix)
       
       return self.accuracies
    
    def precision(self): 
       
       self.precisions = {}
       for name, matrix in self.confusion.items():
          self.precisions[name] = precision_f(matrix)
       return self.precisions
    
    def recall(self):
       self.recalls = {}
       for name, matrix in self.confusion.items():
          self.recalls[name] = recall_f(matrix)
       return self.recalls
    
    def f1(self):
       self.f1_scores = {}
       for name, matrix in self.confusion.items():
          self.f1_scores[name] = f1_score_f(matrix)
       return self.f1_scores

    def global_Error(self, num_folds=10):

        
        #TODO: this did not need to use indexes, probably replace later?
        newOrder =list(range(len(self.dataset)))
        random.shuffle(newOrder)
        
        global_error_sum = {}
        
        for iteration_num in range(num_folds):
            test_start = iteration_num * math.floor( (len(self.dataset) / num_folds) )
            test_finish = (iteration_num + 1) * math.floor( (len(self.dataset) / num_folds) )
            test_set = []
            train_set = []
            
            for i in range(test_start, test_finish):
                test_set.append( self.dataset[ newOrder[i] ] )
                
            if test_start == 0:
                
                for i in range(test_finish, len(self.dataset)):
                    train_set.append( self.dataset[ newOrder[i] ] )
            
            else:
                for i in range(0, test_start):
                    train_set.append( self.dataset[newOrder[i]] )
                for i in range(test_finish, len(self.dataset)):
                    train_set.append(  self.dataset[newOrder[i]])
                    
                    
            for model in self.models:
                if model == Decision_Tree_Classifier.Tree:
                    instance = model(self.dataset, self.tree_depth, train_given = np.array(train_set), test_given = np.array(test_set))
                else:
                    instance = model(self.dataset, train_given = np.array(train_set), test_given = np.array(test_set))

                success = instance.evaluate_internal()
                
                counter = 0
                for outcome  in success:
                    if outcome:
                        counter += 1
                        
                accuracy = counter / len(success)
                
                if instance.name() not in list(global_error_sum.keys()):
                    global_error_sum[instance.name()] = accuracy 
                else: 
                    global_error_sum[instance.name()] += accuracy
                    
                    
        for model_type, error_sum in global_error_sum.items():
            global_error_sum[model_type] = error_sum * (num_folds) ** -1

        self.global_error = global_error_sum
        return self.global_error

    def all_metrics(self):
        '''
        Plot tables of all the different performance evaluation metrics
        '''

        
        fig, ax = plt.subplots(len(self.models), 1, figsize=(12, 4))
        self.global_Error()
        plotNum = 0
        for model, matrix in self.confusion.items():
            
            col_labs = ["Predicted Room: {}".format(x) for x in range(1, len(matrix[0]) + 1)] 
            row_labs = ["Actually Room: {}".format(x) for x in range(1, len(matrix[0]) + 1)] 

        
            ax[plotNum].set_axis_off() 
            table = ax[plotNum].table( 
                cellText = matrix,  
                rowLabels = row_labs,  
                colLabels = col_labs, 
                rowColours =["lightblue"] * 10,  
                colColours =["lightblue"] * 10, 
                cellLoc ='center',  
                loc ='upper left')         
            
            ax[plotNum].set_title("Confusion Matrix of {a} over {c} folds with Global Error {b}%".format(a=model, b= round((1 - self.global_error[model]) * 100, 2), c=self.num_folds ), fontweight ="bold") 
            
            plotNum += 1
            
        plt.savefig('./figures/confusion_matricies.png', dpi=150)
        
        
        fig1, ax1 = plt.subplots(len(self.models), 1)
        col_labels = ["Precision", "Recall", "F1 Score"]
        
        plotNum = 0
        for model, matrix in self.confusion.items():
            row_labs = ["Room: {}".format(x) for x in range(1, len(matrix[0]) + 1)]

            
            p, macrop = precision_f(matrix)
            accuracy = accuracy_f(matrix)
            r, macro_r = recall_f(matrix)
            f, macrof = f1_score_f(matrix)
            
            
            table_data = [np.array( [ "{}%".format(round(elem * 100, 2)) for elem in p ] ), 
                          np.array( [ "{}%".format(round(elem * 100, 2)) for elem in r ] ) , 
                          np.array( [ "{}".format(round(elem * 100, 2)) for elem in f ] )]
  

            ax1[plotNum].set_axis_off() 
            table = ax1[plotNum].table( 
                cellText = np.transpose( np.array( table_data ) ),  
                rowLabels = row_labs,  
                colLabels = col_labels, 
                rowColours =["lightblue"] * 10,  
                colColours =["lightblue"] * 10, 
                cellLoc ='center',  
                loc ='upper left')    
            
            ax1[plotNum].set_title("Performance metrics of " + model, fontweight ="bold")      
            plotNum += 1
        

        plt.savefig('./figures/performance_metrics.png', dpi=150)
        
        # plt.show() 






class Depth_Hyperparameter_Tuning():
    #clean:
    #best depth: 17 , accuracy: 0.9694999999999998

    #noisy:
    #best depth: 52 , accuracy: 0.8574999999999999
    
    def __init__(self, dataset, tree_model, depth_min = 5, depth_max = 40, num_folds=10):
        
        self.dataset = dataset
        self.tree_model = tree_model
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.num_folds = num_folds
        newOrder =list(range(len(self.dataset)))
        random.shuffle(newOrder)
        self.newOrder = newOrder
        
    
    def show(self):
        
        x = []
        y = {}
        template = {"Accuracy": [], "Average Precision": [], "Average Recall": [], "Average F1 Score": []}
        for depth, matrix in self.depth.items():
            x.append(depth)
            y = template
            a = accuracy_f(matrix)
            p, macrop = precision_f(matrix)
            (recalls, macro_r) = recall_f(matrix)
            f, macro_f = f1_score_f(matrix)

            y["Accuracy"].append( a )
            y["Average Precision"].append( sum(p)/len(p) )
            y["Average Recall"].append( sum(recalls)/len(recalls) )
            y["Average F1 Score"].append( sum(f)/len(f) )
            
            
        fig, ax = plt.subplots()
        for label, data in y.items():
            ax.plot(x, data, label=label)
            
        ax.set_xlabel("Maximum Allowed Tree Depth")
        ax.set_ylabel("Performance Metric Score")
        ax.set_title("Performance of a Decision Tree Over Multiple Depths")
        ax.legend()
        ax.grid(True)
        plt.show()
        
        


    def depth_test(self, depth_num):
        depth_avgs = []
        
        for iteration_num in range(self.num_folds):
            print("Calculating Fold {a} at Depth {b}".format(a = iteration_num, b=depth_num))
            test_start = iteration_num * math.floor( (len(self.dataset) / self.num_folds) )
            test_finish = (iteration_num + 1) * math.floor( (len(self.dataset) / self.num_folds) )
            test_set = []
            train_set = []
                
            for i in range(test_start, test_finish):
                test_set.append( self.dataset[ self.newOrder[i] ] )
                    
            if test_start == 0:
                    
                for i in range(test_finish, len(self.dataset)):
                    train_set.append( self.dataset[ self.newOrder[i] ] )
                
            else:
                for i in range(0, test_start):
                    train_set.append( self.dataset[self.newOrder[i]] )
                for i in range(test_finish, len(self.dataset)):
                    train_set.append(  self.dataset[self.newOrder[i]])
                        
            instance = self.tree_model(self.dataset, depth_num, train_given=np.array(train_set), test_given=np.array(test_set))
            results = instance.evaluate_internal()
                
            sum = 0
            for result in results:
                if result == 1: sum += 1
                    
            depth_avgs.append(sum / len(results))
        return depth_avgs


    def run(self):
        print("Running hyperparameter tuning on depth paramter")
        manager = mp.Manager()
        depth = manager.dict()
        depth_num = []
        n_threads = mp.cpu_count()
        print("Starting multiprocessing on", n_threads, "threads")
        if n_threads == None: n_threads = 8
        pool = mp.Pool(n_threads)
        
        for i in range(self.depth_min, self.depth_max + 1):
            depth_num.append(i)

        depth = pool.map(self.depth_test, iterable= depth_num)
            
        pool.close()
        pool.join()

        print('Done', flush=True)

        depth_accuracy_avg = depth
        for i in range(len(depth)):
            sum = 0
            for data_point in depth[i]:
                sum += data_point
            avg = sum / len(depth[i])
            depth_accuracy_avg[i] = avg

        best_accuracy = 0
        best_index = 0
        for i in range(len(depth_accuracy_avg)):
            if depth_accuracy_avg[i] > best_accuracy:
                best_accuracy = depth_accuracy_avg[i]
                best_index = i

        print("best depth:", depth_num[best_index],", accuracy:", best_accuracy)

     
        plt.plot(depth_num, depth_accuracy_avg)
        plt.savefig("./figures/Depth_HyperParameter_Tuning.png", dpi=300)
        





# THIS IS NOT QUITE WORKING YET, SO WILL FIX LATER
class Split_Hyperparameter_Tuning():
    '''Tune the Split Parameter'''

    def __init__(self, dataset, tree_model, num_tests=20, num_folds=10):
        
        self.dataset = dataset
        self.trials = {}
        self.num_tests = num_tests
        self.num_folds = num_folds
        
        
        newOrder =list(range(len(self.dataset)))
        random.shuffle(newOrder)
        
        self.max = 0.6
        self.min = 0.0
        for test_number in range(num_tests):
            split_threshold = (test_number * 1/num_tests) * self.max + self.min
            print("Testing Threshold: ", split_threshold)
            print("\n")
            global_error_sum = 0
            for fold_iteration_num in range(num_folds):
                print("Fold: ", fold_iteration_num)
                test_start = fold_iteration_num * math.floor( (len(self.dataset) / num_folds) )
                test_finish = (fold_iteration_num + 1) * math.floor( (len(self.dataset) / num_folds) )
                test_set = []
                train_set = []
                
                for i in range(test_start, test_finish):
                    test_set.append( self.dataset[ newOrder[i] ] )
                    
                if test_start == 0:
                    
                    for i in range(test_finish, len(self.dataset)):
                        train_set.append( self.dataset[ newOrder[i] ] )
                
                else:
                    for i in range(0, test_start):
                        train_set.append( self.dataset[newOrder[i]] )
                    for i in range(test_finish, len(self.dataset)):
                        train_set.append(  self.dataset[newOrder[i]])

                instance = tree_model(self.dataset, 15, train_given = np.array(train_set), test_given = np.array(test_set), split_threshold=split_threshold)
                success = instance.evaluate_internal()
                
                counter = 0
                for outcome  in success:
                    if outcome:
                        counter += 1
                        
                accuracy = counter / len(success)
                global_error_sum += accuracy
            
            
            self.trials[split_threshold] = global_error_sum * (num_folds) ** -1      
        print(self.trials)


    def plot_performance(self):
        '''Plot Global Accuracy for different split thresholds, assuming global accuracy = 1 - global error'''
        
        
        test_data_1 = {
                0.0                : 0.9664999999999999, 
                0.024              : 0.9664999999999999, 
                0.048              : 0.9664999999999999, 
                0.072              : 0.9664999999999999, 
                0.096              : 0.9664999999999999, 
                0.12               : 0.9664999999999999, 
                0.144              : 0.9664999999999999, 
                0.16799999999999998: 0.9664999999999999, 
                0.192              : 0.9664999999999999, 
                0.216              : 0.9664999999999999, 
                0.24               : 0.9664999999999999, 
                0.264              : 0.9664999999999999, 
                0.288              : 0.9664999999999999, 
                0.312              : 0.9664999999999999, 
                0.33599999999999997: 0.9664999999999999, 
                0.36               : 0.9664999999999999, 
                0.384              : 0.9664999999999999, 
                0.408              : 0.9664999999999999, 
                0.432              : 0.9664999999999999, 
                0.45599999999999996: 0.9664999999999999}
        
        
        # This test is quite computationally expencive, so this is the results pre-stored to make runniung the test less long
        data = self.trials
        
        split_threshold = list(data.keys())
        split_accuracy = list(data.values())
        
        plt.plot(split_threshold, split_accuracy )
        plt.savefig("./figures/HyperParameter_Tuning.png", dpi=300)





def confusion_matrix(actuals, predictions, class_labels=None):
  """ Compute the confusion matrix.

  Args:
      actual (np.ndarray): the correct ground truth/gold standard labels
      prediction (np.ndarray): the predicted labels
      class_labels (np.ndarray): a list of unique class labels.
                             Defaults to the union of actual and prediction.

  Returns:
      np.array : shape (C, C), where C is the number of classes.
                 Rows are ground truth per class, columns are predictions
  """

  # if no class_labels are given, we obtain the set of unique class labels from
  # the union of the ground truth annotation and the prediction
  if class_labels is None:
      class_labels = np.unique(np.concatenate((actuals, predictions)))

  confusion = np.zeros((len(class_labels), len(class_labels)), dtype=float)

     

  for i in range(len(predictions)):
    prediction = predictions[i] - 1
    actual = actuals[i] - 1
    confusion[int(prediction)][int(actual)] += 1

  return confusion


def accuracy_f(confusion):
  """ Compute the accuracy given the confusion matrix

  Args:
      confusion (np.ndarray): shape (C, C), where C is the number of classes.
                  Rows are ground truth per class, columns are predictions

  Returns:
      float : the accuracy
  """

  if np.sum(confusion) > 0:
      return np.trace(confusion) / np.sum(confusion) # trace <= sum of elements across diagonal
  else:
      return 0.


def precision_f(confusion):
  """ Compute the precision score per class given the ground truth and predictions
  Also return the macro-averaged precision across classes.

  Returns:
      tuple: returns a tuple (precisions, macro_precision) where
          - precisions is a np.ndarray of shape (C,), where each element is the
            precision for class c
          - macro-precision is macro-averaged precision (a float)
  """
  
  # Compute the precision per class
  p = np.diag(confusion) / confusion.sum(axis = 0)
  # Compute the macro-averaged precision
  macro_p = np.mean(p)

  return (p, macro_p)


def recall_f(confusion):
  """ Compute the recall score per class given confusion matrix
  Also return the macro-averaged recall across classes.

  Returns:
      tuple: returns a tuple (recalls, macro_recall) where
          - recalls is a np.ndarray of shape (C,), where each element is the
              recall for class c
          - macro-recall is macro-averaged recall (a float)
  """

  # Compute the recall per class
  r = np.diag(confusion) / confusion.sum(axis = 1)
  # Compute the macro-averaged recall
  macro_r = np.mean(r)

  return (r, macro_r)


def f1_score_f(confusion):
  """ Compute the F1-score per class given the ground truth and predictions

  Also return the macro-averaged F1-score across classes.

  Args:
      y_gold (np.ndarray): the correct ground truth/gold standard labels
      y_prediction (np.ndarray): the predicted labels

  Returns:
      tuple: returns a tuple (f1s, macro_f1) where
          - f1s is a np.ndarray of shape (C,), where each element is the
            f1-score for class c
          - macro-f1 is macro-averaged f1-score (a float)
  """

  (precisions, macro_p) = precision_f(confusion)
  (recalls, macro_r) = recall_f(confusion)

  # just to make sure they are of the same length
  assert len(precisions) == len(recalls)
  f = 2 * (precisions * recalls) / (precisions + recalls)

  macro_f = np.mean(f)

  return (f, macro_f)

    
    
    
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

if __name__ == '__main__':
    pass
      
