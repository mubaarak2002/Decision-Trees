# Machine-Learning-Decision-Trees
Group Members: <br />
Edward Chen (ec1221) <br />
Mubaarak Khan (mmk120) <br />
Omar Ben-Gacem (ob420) <br />

## Running the Project
`main.py` has been setup to take in 3 arguments `<dataset path>`, `<depth>`, and `<operating_mode>` <br />

 Use `python main.py <dataset path> <depth> <operating_mode>` to run the project in a potential mode. All possible modes are described below <br />
 e.g: <br />
 `python main.py "WIFI_db/clean_dataset.txt" 20 "metrics"`

 ### Operating Modes
 | operating_mode  | Function  |
 | --------------- |---------- |
 |  "show_tree"      |  generates an interactive MatPlotLib figure to view the tree. Note that all nodes will appear as circles, and upon clicking on the node, a popup will appear and give information on the node  |
 |  "metrics"        |  Update the tables of the Confusion Matrix and the Performance Metrics and place them in the figures file. This may take a  minutes to run. |
 | "depth_benchmark" | Update the figures regarding the performance of the decision tree for various depths. Note in this instance the depth field is ignored, and instead plots all data from 4 to 70. **NOTE**: This computation takes a long time to run, see *Reccomended Depth Parameters* For the output of this test when run in advance |
 | "normal"          | Plot the normal distribution of the accuracy for all three Classification Methods over 10 Folds  |

 ### Hyperparameter Tuning - Depth
 Instead of providing a speific `<depth>` integer argument to build the tree with, inputing `tune` as the argument will run hyperparameter tuning on the dataset first to find the optimal depth and will proceed to use that value. NOTE: This is a very resource intensive task.


## Reccomended Depth Parameters
Hyperparameter Tuning runs multiple trials (plotted below) to find the optimal depth parameter for both the clean and noisy datasets. The Plots are shown below

Clean Dataset Depth Optimisation |  Noisy Dataset Depth Optimisation
:-------------------------:|:-------------------------:
![](./figures/Depth_HyperParameter_Tuning_clean_.png)  |  ![](./figures/Depth_HyperParameter_Tuning_noisy_.png)
*Note the differing scales in the figures*


Both Plots Plateau for a certain depth. **When running the clean dataset, it is reccommended to use a depth of 20, and for the noisy dataset, it is reccomended to use a depth of 50**. This will give the best performance of the model on each dataset.

