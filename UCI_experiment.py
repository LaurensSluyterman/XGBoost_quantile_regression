def main():
    import os
    import numpy as np
    import argparse
    import xgboost as xgb
    import matplotlib
    matplotlib.use('TkAgg')
    from sklearn.model_selection import KFold
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GridSearchCV
    from objective_functions import arctan_loss, smooth_pinball_loss
    from utils import scoring_loss, pinball_loss, compute_crossing_percentage, load_data, plot_quantiles
    from functools import partial
    from time import time
    from utils import pinball_loss_total, scoring_loss
    from joblib import dump, load

    os.chdir('/Users/laurens/OneDrive/Onedrivedocs/PhD/Code/2023/XGBoost_quantile_regression')
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', required=True,
                        help="name of dataset")
    parser.add_argument('-l', '--loss', required=True,
                        help="loss function")
    parser.add_argument('-nfo', '--number_of_folds_out', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('-nfi', '--number_of_folds_in', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('-k', '--number_of_taus', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('-s', '--smoothing', type=float, default=0.1,
                        help='Number of training epochs')
    args = parser.parse_args()

    # Load the data set
    dataset = args.dataset
    X, Y = load_data(dataset)

    number_of_outer_folds = args.number_of_folds_out
    number_of_inner_folds = args.number_of_folds_in
    loss = eval(args.loss)
    K = args.number_of_taus
    s = args.smoothing

    # Calculated the desired quantiles
    taus = [-0.05 + i * 1 / (K) for i in range(K, 0, -1)]

    def get_best_model(model, X, Y, param_grid):
        """Executes a grid search"""
        wrapped_model = TransformedTargetRegressor(regressor=model, transformer=StandardScaler())
        grid_search = GridSearchCV(wrapped_model, param_grid, cv=number_of_inner_folds,
                                   scoring=partial(scoring_loss, taus=taus))
        grid_search.fit(X, Y)
        return grid_search


    predicted_quantiles_original = np.zeros((len(X), K))
    predicted_quantiles_new = np.zeros((len(X), K))
    # The outer loop of the cross validation
    for i, (training_indices, test_indices) in enumerate(
            KFold(number_of_outer_folds, shuffle=True, random_state=1).split(X)):
        print(f'{i + 1}  of {number_of_outer_folds}')
        x, x_val = X[training_indices], X[test_indices]
        y, y_val = Y[training_indices], Y[test_indices]


        # We need to make targets the correct shape to make XGBoost output a vector
        targets = np.zeros((len(y), K))
        for j in range(K):
            targets[:, j] = y
        y = np.reshape(y, (len(y), 1))

        # Initialize both the original and new model
        xgb_new = xgb.XGBRegressor(objective=partial(loss, taus=taus, s=s),
                                   base_score=0,
                                   multi_strategy="multi_output_tree",
                                   min_child_weight=0,
                                   max_delta_step=0.5)

        xgb_original = xgb.XGBRegressor(objective='reg:quantileerror',
                                        quantile_alpha=taus,
                                        base_score=0,
                                        min_child_weight=0)

        # Define the grid for the hyper-parameter search
        param_grid = {
            'regressor__n_estimators': [100, 200, 400],
            'regressor__max_leaves': [400],
            'regressor__max_depth': [2, 3, 4],
            'regressor__learning_rate': [0.05],
            'regressor__gamma': [0.1, 0.25, 0.5, 1, 2.5, 5, 10],
            'regressor__lambda': [0.01, 0.1, 0.25, 0.5, 1, 2.5, 5, 10],
        }

        # Obtain the optimal model (the inner loop of cross-validation)
        start_time1 = time()
        grid_search_original = get_best_model(xgb_original, x, y, param_grid)
        end_time1 = time()

        start_time2 = time()
        grid_search_new = get_best_model(xgb_new, x, targets, param_grid)
        end_time2 = time()

        best_model_original = grid_search_original.best_estimator_
        best_model_new = grid_search_new.best_estimator_
        predicted_quantiles_original[test_indices] = best_model_original.predict(x_val)
        predicted_quantiles_new[test_indices] = best_model_new.predict(x_val)

    # Compute the metrics for both models and save the predicted quantiles
    old_crossing = compute_crossing_percentage(predicted_quantiles_original)
    new_crossing = compute_crossing_percentage(predicted_quantiles_new)

    old_width = np.mean(predicted_quantiles_original[:, 0] - predicted_quantiles_original[:, -1])
    new_width = np.mean(predicted_quantiles_new[:, 0] - predicted_quantiles_new[:, -1])

    dump(predicted_quantiles_new, f'./results/{dataset}_{s}_quantiles_new_pg2')
    dump(predicted_quantiles_original, f'./results/{dataset}_{s}_quantiles_old_pg2')

    # Report various metrics
    print(f'{dataset}')
    print(f's = {s}')
    print(f'Old crossing: {np.round(old_crossing,1)}%')
    print(f'New crossing: {np.round(new_crossing,1)}%')
    print('  ')
    uppercorrect = Y < predicted_quantiles_new[:, 0]
    lowercorrect = Y > predicted_quantiles_new[:, -1]
    print(f'New coverage: {np.round(100 * np.mean(uppercorrect * lowercorrect), 2)}%')
    uppercorrect = Y < predicted_quantiles_original[:, 0]
    lowercorrect = Y > predicted_quantiles_original[:, -1]
    print(f'Original coverage: {np.round(100 * np.mean(uppercorrect * lowercorrect), 2)}%')
    print('  ')
    print(f'Old width: {old_width}')
    print(f'New width: {new_width}')
    print('  ')
    print(f'Old time: {end_time1 - start_time1}')
    print(f'New time: {end_time2 - start_time2}')

    # Create and save a calibration plot
    plot_quantiles(predicted_quantiles_new, Y, taus, save_loc=f'./plots/{dataset}_new')
    plot_quantiles(predicted_quantiles_original, Y, taus, save_loc=f'./plots/{dataset}_old')

    # Report the optimal hyper parameter (for the last split)
    print('  ')
    print(f'Old params: {grid_search_original.best_params_}')
    print(f'New params: {grid_search_new.best_params_}')
    print('  ')
    print(f'Old pinball_loss: {pinball_loss_total(predicted_quantiles_original, Y, taus) / len(Y)}')
    print(f'New pinball_loss: {pinball_loss_total(predicted_quantiles_new, Y, taus) / len(Y)}')

if __name__ == '__main__':
    main()

