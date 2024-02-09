# ICU_Classification
How to run
1.Run run_pipeline to fit on some data or use train_data(default) (without --test argument). Use --data_path argument to add path to data.\
2.This will create a model and a preprocessor should be saved in the same directory as finalized_preprocessor.sav and finalized_model.sav\
3.Next run run_pipeline to test on some data (with --test argument)\
4.The predictions are saved in the directory where you ran the command as probas.json file\
About model.py\
model.py contains class model that have 2 parameters - model(default = SVC(kernel="rbf", C=1, probability=True, class_weight="balanced")) and threshold(default = 0.2). In addition to standard methods, it also includes a save_probas(self, x) method that stores probabilities and a threshold in a JSON file. Also it has modified score method, which returns accuracy, sensitivity, specificity, f1 score and auc in numpy array.  
About preprocessor.py\
preprocessor.py defines Preprocessor class which has 3 methods - fit, transform and pretransform. Pretransform method fills nans with mean medical norms. Fit and transform methods apply standard scaling.\
Requirements\
Python 3.12.0\
Numpy 1.26.3\
Pandas 2.2.0\
Pip 23.2.1\
Pyarrow 15.0.0\
Scikit-Learn 1.4.0
