import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection

# Use time series variant of K-fold cross validation to generate estimates of test time performance
# Collect Data to later be used to define the final model.
def forest_param_search(train_features, train_labels, n_trials, num_trees, depths):
    # Split the data
    splitter = sklearn.model_selection.TimeSeriesSplit(n_splits = 4)

    # Define variables to hold result metrics
    MSQ_by_trial = np.zeros(n_trials)
    MAE_by_trial = np.zeros(n_trials)
    
    # Perform cross-validation on each model configuration
    model_index = 0
    for n_trees,depth in zip(num_trees,depths):
        print("Testing model {} of {}".format(model_index+1, n_trials), "num_trees:", n_trees, " and max_depth: ", depth)

        # Define error metrics arrays
        fold_index = 0
        MAEs = np.zeros(4)
        MSQs = np.zeros(4)

        # Perform cross-validation on the current model
        for train_index, test_index in splitter.split(train_features):
            # Split the data in training and validation set
            X_train, X_test = train_features[train_index], train_features[test_index]
            y_train, y_test = train_labels[train_index], train_labels[test_index]

            # Define andfit the model
            model = sklearn.ensemble.RandomForestRegressor(n_estimators= n_trees,
                                                           max_depth = depth, random_state=0)
            model = model.fit(X_train,y_train)

            # Make predictions and generate useful metrics
            y_pred = model.predict(X_test)
            msq = sklearn.metrics.mean_squared_error(y_test, y_pred)
            mae = sklearn.metrics.mean_absolute_error(y_test, y_pred)

            #Save metrics in an array for later analysis
            MSQs[fold_index] = msq
            MAEs[fold_index] = mae

            # Increase fold index
            fold_index += 1

        # Add mean error metrics for last fold to trial metrics
        MSQ_by_trial[model_index] = np.mean(MSQs)
        MAE_by_trial[model_index] = np.mean(MAEs)

        # Increase model index   
        model_index +=1

    # Parameter search completed
    print('\nDone')
    return (MSQ_by_trial, MAE_by_trial)


# Function for plotting a summary of hyperparameter effects on MSQ
def plot_param_performance(num_trees, depths, MSQ_by_trial, MAE_by_trial):
    # Plot each parameters configuration with its performance denoted by marker color
    plt.figure(figsize =(12,8))
    plt.scatter(num_trees, depths, marker = 'x', c = MSQ_by_trial, s = 100)
    plt.colorbar()
    plt.title("Average MSE over training of a Random Forest")
    plt.xlabel("Number of Estimators")
    plt.ylabel("Maximum Depth of Estimators")
    
    # Print summary of best classifier
    print("Best Classifier Attributes")
    print("Number of Estimators: ", num_trees[np.argmin(MSQ_by_trial)], "Max Depth: ", depths[np.argmin(MSQ_by_trial)])
    print("MSE of best classifer: ", MSQ_by_trial[np.argmin(MSQ_by_trial)])
    print("MAE of best classifier: ", MAE_by_trial[np.argmin(MAE_by_trial)] )