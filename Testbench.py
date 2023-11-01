import numpy as np
import matplotlib.pyplot as plt
from matplotlib.table import Table
import random
import math

class Testbench():
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
  
    def __init__(self, dataset, num_trials=10, *args):
        
        Models = []
        for testMethod in args:
            Models.append(testMethod)
        means = {}
        all_samples = {}
        for index, model in enumerate(Models):
            
            for trial in range(num_trials):
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
        plt.xlabel("Success Rate")
        plt.ylabel("Probability Concentration")
        plt.legend()
        plt.grid(True)
        plt.show()
            
    def confusion_matrix(self):
        for model in self.models:
            
            specific_model = model(self.dataset)
            (labels, actual, guess) = specific_model.confusion_constructor()
            self.confusion[specific_model.name()] = (confusion_matrix(actual, guess, class_labels=labels))
        
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
          self.f1s_scores[name] = f1_score_f(matrix)
       return self.f1_scores

    def global_Error(self):
        
            
        num_folds = 10
        
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
        print(self.confusion)

        
        fig, ax = plt.subplots(3, 1, figsize=(12, 4))
        
        plotNum = 0
        for model, matrix in self.confusion.items():
            col_labs = ["Room: {}".format(x) for x in range(len(matrix[0]))] 
            row_labs = ["Room: {}".format(x) for x in range(len(matrix[0]))] 
            val3 = [["" for c in range(10)] for r in range(10)] 
        
            ax[plotNum].set_axis_off() 
            table = ax[plotNum].table( 
                cellText = matrix,  
                rowLabels = row_labs,  
                colLabels = col_labs, 
                rowColours =["palegreen"] * 10,  
                colColours =["palegreen"] * 10, 
                cellLoc ='center',  
                loc ='upper left')         
            
            ax[plotNum].set_title("Confusion Matrix of " + model, 
                        fontweight ="bold") 
            
            plotNum += 1
        plt.show() 



        plt.savefig('pyplot-table-original.png',
                    bbox_inches='tight',
                    dpi=150
                    )

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

  confusion = np.zeros((len(class_labels), len(class_labels)), dtype=int)

     

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

  (precisions, macro_p) = precision(confusion)
  (recalls, macro_r) = recall(confusion)

  # just to make sure they are of the same length
  assert len(precisions) == len(recalls)
  f = 2 * (precisions * recalls) / (precisions + recalls)

  macro_f = np.mean(f)

  return (f, macro_f)

    
    
    
        

      