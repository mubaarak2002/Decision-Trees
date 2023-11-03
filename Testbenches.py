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

        if(0):
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
            ax[plotNum].set_title("Confusion Matrix of " + model, fontweight ="bold") 
            
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
    
    def __init__(self, dataset, tree_model, depth_min = 4, depth_max = 55, num_folds=10, mode="preLoaded-noisy"):
        
        # This test takes a very long time to run, so there is some pre-determined data here
        preRun_clean_data = {
            4: np.array([[49.8,  0. ,  0.5,  0.9],
            [ 0. , 43.1,  0.2,  0. ],
            [ 0.2,  6.9, 49. ,  0.2],
            [ 0. ,  0. ,  0.3, 48.9]]), 
            
            5: np.array([[49.8,  0. ,  0.9,  0.8], 
            [ 0. , 43.8,  0.5,  0. ],
            [ 0.2,  6.2, 48.5,  0.1],
            [ 0. ,  0. ,  0.1, 49.1]]), 
            
            6: np.array([[49.8,  0. ,  0.8,  0.7], 
            [ 0. , 44.3,  0.8,  0. ],
            [ 0.2,  5.7, 48.3,  0.2],
            [ 0. ,  0. ,  0.1, 49.1]]), 
            
            7: np.array([[49.8,  0. ,  0.9,  0.7], 
            [ 0. , 45.4,  0.6,  0. ],
            [ 0.2,  4.6, 48.3,  0. ],
            [ 0. ,  0. ,  0.2, 49.3]]), 
            
            8: np.array([[49.8,  0. ,  0.8,  0.6], 
            [ 0. , 45.4,  0.7,  0. ],
            [ 0.2,  4.6, 48.3,  0.2],
            [ 0. ,  0. ,  0.2, 49.2]]), 
            
            9: np.array([[49.7,  0. ,  0.5,  0.6], 
            [ 0. , 45.2,  0.5,  0. ],
            [ 0.2,  4.8, 48.7,  0.1],
            [ 0.1,  0. ,  0.3, 49.3]]), 
            
            10: np.array([[49.8,  0. ,  0.8,  0.6],
            [ 0. , 45.3,  0.5,  0. ],
            [ 0.2,  4.7, 48.5,  0.1],
            [ 0. ,  0. ,  0.2, 49.3]]), 
            
        11: np.array([[49.7,  0. ,  0.5,  0.4],
            [ 0. , 45.5,  0.5,  0. ],
            [ 0.2,  4.5, 48.8,  0.3],
            [ 0.1,  0. ,  0.2, 49.3]]), 
            
        12: np.array([[49.7,  0. ,  0.6,  0.5],
            [ 0. , 45.7,  0.5,  0. ],
            [ 0.2,  4.3, 48.7,  0.2],
            [ 0.1,  0. ,  0.2, 49.3]]), 
            
        13: np.array([[49.7,  0. ,  0.7,  0.5],
            [ 0. , 45.5,  0.5,  0. ],
            [ 0.2,  4.5, 48.5,  0.2],
            [ 0.1,  0. ,  0.3, 49.3]]), 
            
        14: np.array([[49.7,  0. ,  0.6,  0.3],
            [ 0. , 45.7,  0.8,  0. ],
            [ 0.2,  4.3, 48.3,  0.3],
            [ 0.1,  0. ,  0.3, 49.4]]), 
            
        15: np.array([[49.7,  0. ,  0.6,  0.5],
            [ 0. , 46. ,  0.5,  0. ],
            [ 0.2,  4. , 48.6,  0.2],
            [ 0.1,  0. ,  0.3, 49.3]]), 
            
        16: np.array([[49.7,  0. ,  0.6,  0.5],
            [ 0. , 45.8,  0.5,  0. ],
            [ 0.2,  4.2, 48.6,  0.1],
            [ 0.1,  0. ,  0.3, 49.4]]), 
            
        17: np.array([[49.8,  0. ,  0.7,  0.3],
            [ 0. , 45.5,  0.5,  0. ],
            [ 0.2,  4.5, 48.7,  0.2],
            [ 0. ,  0. ,  0.1, 49.5]]), 
            
            18: np.array([[49.7,  0. ,  0.7,  0.5],
            [ 0. , 46.1,  0.7,  0. ],
            [ 0.2,  3.9, 48.3,  0.3],
            [ 0.1,  0. ,  0.3, 49.2]]), 
            
            19: np.array([[49.6,  0. ,  0.6,  0.3],
            [ 0. , 46. ,  0.9,  0. ],
            [ 0.4,  4. , 48.1,  0.3],
            [ 0. ,  0. ,  0.4, 49.4]]), 
            
            20: np.array([[49.8,  0. ,  0.8,  0.5],
            [ 0. , 46.2,  0.5,  0. ],
            [ 0.2,  3.8, 48.6,  0.2],
            [ 0. ,  0. ,  0.1, 49.3]]), 
            
            21: np.array([[49.8,  0. ,  0.7,  0.6],
            [ 0. , 45.6,  0.5,  0. ],
            [ 0.2,  4.4, 48.3,  0.1],
            [ 0. ,  0. ,  0.5, 49.3]]), 
            
            22: np.array([[49.7,  0. ,  0.6,  0.4],
            [ 0. , 45.5,  0.6,  0. ],
            [ 0.2,  4.5, 48.4,  0.5],
            [ 0.1,  0. ,  0.4, 49.1]]), 
            
            23: np.array([[49.7,  0. ,  0.6,  0.6],
            [ 0. , 45.8,  0.7,  0. ],
            [ 0.2,  4.2, 48.3,  0.3],
            [ 0.1,  0. ,  0.4, 49.1]]), 
            
            24:  np.array([[49.7,  0. ,  0.7,  0.5],
            [ 0. , 45.9,  1. ,  0. ],
            [ 0.2,  4.1, 48. ,  0.2],
            [ 0.1,  0. ,  0.3, 49.3]]), 
            
            25: np.array([[49.7,  0. ,  0.7,  0.4],
            [ 0. , 46. ,  0.6,  0. ],
            [ 0.3,  4. , 48.5,  0.3],
            [ 0. ,  0. ,  0.2, 49.3]])
       
       }
        
        
        preRun_noisy_data = {4: np.array([[ 0. ,  0. ,  0. ,  0. ],
       [ 0. ,  2.5,  0. ,  0. ],
       [37.6, 38.2, 39.9, 42. ],
       [11.4,  9. , 11.6,  7.8]]), 5: np.array([[ 0. ,  0. ,  0.1,  0. ],
       [ 4.5,  6.4,  6. ,  5.4],
       [44.5, 43.3, 45.4, 43.8],
       [ 0. ,  0. ,  0. ,  0.6]]), 6: np.array([[ 0. ,  0. ,  0.1,  0. ],
       [ 5.1,  5.9,  6.1,  4.9],
       [43.9, 43.6, 45.3, 43.5],
       [ 0. ,  0.2,  0. ,  1.4]]), 7: np.array([[ 0.1,  0.3,  0.1,  0. ],
       [ 0.1,  2.3,  0. ,  0. ],
       [48.8, 47.1, 51.4, 48.6],
       [ 0. ,  0. ,  0. ,  1.2]]), 8: np.array([[ 0.2,  0.1,  0.1,  0. ],
       [ 0.1,  2.2,  0.1,  0. ],
       [44.7, 42.4, 45.4, 44.3],
       [ 4. ,  5. ,  5.9,  5.5]]), 9: np.array([[ 0.2,  0.1,  0.1,  0. ],
       [ 0.2,  3.1,  0.1,  0. ],
       [44. , 41.4, 45. , 44.3],
       [ 4.6,  5.1,  6.3,  5.5]]), 10: np.array([[ 0. ,  0.1,  0.1,  0.2],
       [ 5.5,  6.9,  5.7,  5.2],
       [37.7, 37.5, 40. , 39. ],
       [ 5.8,  5.2,  5.7,  5.4]]), 11: np.array([[ 0. ,  0.1,  0. ,  0.1],
       [ 0.1,  3.8,  0.1,  0.2],
       [48.9, 45.8, 51.4, 47.2],
       [ 0. ,  0. ,  0. ,  2.3]]), 12: np.array([[ 0. ,  0. ,  0.1,  0.2],
       [ 4.5,  6.8,  6.7,  4.9],
       [44.5, 42.9, 44.7, 42.1],
       [ 0. ,  0. ,  0. ,  2.6]]), 13: np.array([[ 0. ,  0.1,  0.1,  0.2],
       [ 0.2,  2.8,  0.1,  0. ],
       [48.8, 46.8, 51.3, 46.8],
       [ 0. ,  0. ,  0. ,  2.8]]), 14: np.array([[ 0. ,  0. ,  0.1,  0.1],
       [ 0.3,  5.6,  0.1,  0.1],
       [48.4, 44. , 51.2, 42.3],
       [ 0.3,  0.1,  0.1,  7.3]]), 15: np.array([[ 0. ,  0.2,  0.1,  0.2],
       [ 0.3,  6.9,  0.2,  0.2],
       [47.8, 42. , 50.1, 30.2],
       [ 0.9,  0.6,  1.1, 19.2]]), 16: np.array([[ 3.7,  2.1,  6.6,  5. ],
       [ 5.2, 17.7,  6.9,  4.7],
       [39.4, 29.6, 37.7, 25.9],
       [ 0.7,  0.3,  0.3, 14.2]]), 17: np.array([[ 0.3,  0. ,  0.1,  0.4],
       [ 0.4, 14.2,  0.4,  0.5],
       [46.9, 34.7, 49.7, 26.3],
       [ 1.4,  0.8,  1.3, 22.6]]), 18: np.array([[ 3.8,  1. ,  6.3,  5.9],
       [ 0.4, 14.9,  0.3,  0.4],
       [43.1, 32.8, 43.8, 15.2],
       [ 1.7,  1. ,  1.1, 28.3]]), 19: np.array([[ 0.8,  0.2,  0.1,  0.4],
       [ 0.5, 20.1,  0.9,  0.6],
       [46.1, 28.4, 49.4, 14.1],
       [ 1.6,  1. ,  1.1, 34.7]]), 20: np.array([[ 1.5,  0.2,  0.2,  0.4],
       [ 0.3, 14.2,  0.4,  0.7],
       [45.3, 34.4, 49.7, 10.2],
       [ 1.9,  0.9,  1.2, 38.5]]), 21: np.array([[ 8.7,  0.4,  0.4,  1.2],
       [ 0.4, 24.2,  1.2,  1. ],
       [38.4, 24.1, 49. , 15.6],
       [ 1.5,  1. ,  0.9, 32. ]]), 22: np.array([[ 0.9,  0.1,  0.2,  0.4],
       [ 0.5, 22.7,  1. ,  1.1],
       [45.6, 25.8, 48.8,  6.5],
       [ 2. ,  1.1,  1.5, 41.8]]), 23: np.array([[ 6.4,  0.5,  0.6,  0.6],
       [ 0.5, 30.5,  1.2,  0.8],
       [40.3, 17.8, 48.4, 10.7],
       [ 1.8,  0.9,  1.3, 37.7]]), 24: np.array([[10. ,  0.7,  0.3,  0.9],
       [ 0.9, 32.1,  1.1,  1.2],
       [36.3, 15.9, 48.6, 10.7],
       [ 1.8,  1. ,  1.5, 37. ]]), 25: np.array([[10.1,  1. ,  0.4,  1.4],
       [ 0.9, 36.6,  1.3,  1. ],
       [36.1, 11.2, 48.5,  9.3],
       [ 1.9,  0.9,  1.3, 38.1]]), 26: np.array([[16.1,  1.2,  1.6,  3.3],
       [ 0.8, 30.5,  1.3,  1.3],
       [29.9, 17. , 47.1,  4.1],
       [ 2.2,  1. ,  1.5, 41.1]]), 27: np.array([[17.7,  1.2,  0.8,  1.5],
       [ 1. , 32.5,  1. ,  1.1],
       [28.3, 14.9, 48.1,  4.6],
       [ 2. ,  1.1,  1.6, 42.6]]), 28: np.array([[23. ,  0.8,  0.8,  2.3],
       [ 0.8, 35.7,  1.2,  1.3],
       [23.2, 12.2, 48. ,  4.3],
       [ 2. ,  1. ,  1.5, 41.9]]), 29: np.array([[14.9,  0.6,  1.2,  2.1],
       [ 0.8, 35.6,  1.1,  1.4],
       [31.2, 12.4, 47.7,  4.7],
       [ 2.1,  1.1,  1.5, 41.6]]), 30: np.array([[26.8,  1.7,  1.7,  3.1],
       [ 1.1, 34.5,  1.3,  1.4],
       [19. , 12.5, 46.7,  3.1],
       [ 2.1,  1. ,  1.8, 42.2]]), 31: np.array([[32.4,  1.4,  1.7,  3.7],
       [ 0.9, 38.1,  1.3,  1.2],
       [13.8,  9.1, 46.8,  2.4],
       [ 1.9,  1.1,  1.7, 42.5]]), 32: np.array([[34.9,  1.6,  1.8,  3.1],
       [ 0.7, 35.9,  1.4,  1.4],
       [11.3, 11. , 46.5,  2. ],
       [ 2.1,  1.2,  1.8, 43.3]]), 33: np.array([[34.9,  1.7,  2.3,  5.9],
       [ 1.2, 39.7,  1.4,  1.3],
       [10.9,  7.1, 46.3,  1.8],
       [ 2. ,  1.2,  1.5, 40.8]]), 34: np.array([[44.4,  2. ,  2.9,  5. ],
       [ 0.9, 39.2,  1.4,  1.5],
       [ 1.6,  7.3, 45.7,  1.8],
       [ 2.1,  1.2,  1.5, 41.5]]), 35: np.array([[43.9,  2.1,  2.4,  4.4],
       [ 1. , 39.1,  1.3,  1.4],
       [ 2.2,  7.3, 46. ,  1.6],
       [ 1.9,  1.2,  1.8, 42.4]]), 36: np.array([[44.2,  2.3,  2.4,  4.3],
       [ 1. , 38.6,  1.6,  1.6],
       [ 1.8,  7.7, 45.5,  1.6],
       [ 2. ,  1.1,  2. , 42.3]]), 37: np.array([[44.3,  2. ,  2.8,  4.6],
       [ 1.1, 39.9,  1.6,  1.4],
       [ 1.6,  6.8, 45.6,  1.4],
       [ 2. ,  1. ,  1.5, 42.4]]), 38: np.array([[44.1,  1.8,  2.3,  4.8],
       [ 0.9, 40.7,  1.4,  1.3],
       [ 2. ,  6. , 46.3,  1.9],
       [ 2. ,  1.2,  1.5, 41.8]]), 39: np.array([[43.9,  2.4,  2.6,  4.2],
       [ 1.1, 38.3,  1.6,  1.5],
       [ 2. ,  7.7, 45.6,  2. ],
       [ 2. ,  1.3,  1.7, 42.1]]), 40: np.array([[44.1,  2.2,  2.1,  4.2],
       [ 1.2, 38.9,  1.4,  1.3],
       [ 1.6,  7.6, 45.9,  1.9],
       [ 2.1,  1. ,  2.1, 42.4]]), 41: np.array([[44.1,  2.1,  2.5,  4.2],
       [ 1.1, 40.6,  1.4,  1.5],
       [ 1.8,  5.8, 46. ,  1.5],
       [ 2. ,  1.2,  1.6, 42.6]]), 42: np.array([[43.9,  2.2,  2.3,  5.2],
       [ 1.1, 40.5,  1.6,  1.3],
       [ 1.9,  5.8, 45.6,  1.6],
       [ 2.1,  1.2,  2. , 41.7]]), 43: np.array([[43.6,  2.1,  2.3,  3.7],
       [ 1.2, 41.8,  1.6,  1.4],
       [ 2.1,  4.7, 45.9,  1.9],
       [ 2.1,  1.1,  1.7, 42.8]]), 44: np.array([[44.1,  2.1,  2.3,  4. ],
       [ 0.8, 38.4,  1.8,  1.5],
       [ 1.9,  8. , 45.6,  1.8],
       [ 2.2,  1.2,  1.8, 42.5]]), 45: np.array([[43.8,  2.2,  2.3,  4.2],
       [ 0.9, 40. ,  1.7,  1.3],
       [ 2.1,  6.2, 45.4,  1.4],
       [ 2.2,  1.3,  2.1, 42.9]]), 46: np.array([[43.8,  2.3,  2.2,  4. ],
       [ 1. , 40.6,  1.7,  1.4],
       [ 1.9,  5.6, 45.3,  1.4],
       [ 2.3,  1.2,  2.3, 43. ]]), 47: np.array([[44.1,  2.1,  2.5,  4. ],
       [ 0.9, 40.7,  1.6,  1.5],
       [ 2. ,  5.9, 45.6,  1.6],
       [ 2. ,  1. ,  1.8, 42.7]]), 48: np.array([[44.1,  2.2,  2.7,  4.4],
       [ 1. , 40.1,  1.6,  1.5],
       [ 1.9,  6.1, 45.5,  1.4],
       [ 2. ,  1.3,  1.7, 42.5]]), 49: np.array([[44.2,  2.2,  2.5,  4.6],
       [ 1. , 40.6,  1.7,  1.6],
       [ 1.9,  5.7, 45.2,  1.8],
       [ 1.9,  1.2,  2.1, 41.8]]), 50: np.array([[43.6,  2. ,  2.4,  3.8],
       [ 1.1, 40. ,  1.6,  1.6],
       [ 2. ,  6.2, 45.5,  2. ],
       [ 2.3,  1.5,  2. , 42.4]]), 51: np.array([[43.7,  2.6,  2.5,  4.1],
       [ 0.9, 40.7,  1.7,  1.6],
       [ 2.2,  4.8, 45.2,  1.6],
       [ 2.2,  1.6,  2.1, 42.5]]), 52: np.array([[43.7,  3. ,  2.2,  5.1],
       [ 1.2, 40.1,  2.1,  1.6],
       [ 1.9,  5.5, 44.7,  1.9],
       [ 2.2,  1.1,  2.5, 41.2]]), 53: np.array([[43. ,  2.4,  2.3,  4. ],
       [ 1.3, 41.1,  1.7,  1.7],
       [ 2.3,  4.9, 45.3,  1.5],
       [ 2.4,  1.3,  2.2, 42.6]]), 54: np.array([[43.8,  2.1,  2.5,  4.6],
       [ 1.1, 40.7,  1.6,  1.6],
       [ 1.8,  5.3, 45.2,  1.4],
       [ 2.3,  1.6,  2.2, 42.2]]), 55: np.array([[43.5,  2.4,  2.8,  4.6],
       [ 1.4, 40.9,  2.2,  1.9],
       [ 1.7,  5.2, 44.3,  1.6],
       [ 2.4,  1.2,  2.2, 41.7]])}

        
        self.clean_data = preRun_clean_data
        self.noisy_data = preRun_noisy_data
        self.dataset = dataset
        depth = {}
        
        if mode == "preLoaded-noisy":
            depth = self.noisy_data
        elif mode == "preLoaded-clean":
            depth = self.clean_data
            
        else:
            print("Running Test")
            for depth_test in range(depth_min, depth_max + 1):
                print("Running Test for Depth: ", depth_test)
                test_instance = Model_Comparison_TB(dataset, num_folds, depth_test, tree_model)
                
                depth[depth_test] = test_instance.confusion_matrix()["Decision Tree Classifier"]

        self.depth = depth
        print(depth)
        
    
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
        
        


    def run(self):
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
        print(list(zip(depth_num, depth)))
        





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
      
