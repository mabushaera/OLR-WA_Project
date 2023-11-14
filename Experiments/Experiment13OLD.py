from sklearn.model_selection import train_test_split
from Datasets import SyntheticDS
from Hyperparameters import Hyperparameter
from Models.MiniBatchGradientDescent import MiniBatchGradientDescent
from Models.OLR_WA import OLR_WA
from Models.OnlineLassoRegression import OnlineLassoRegression
from Models.OnlinePassiveAggressive import OnlinePassiveAggressive
from Models.OnlineRidgeRegression import OnlineRidgeRegression
from Models.RLS import RLS
from Models.StochasticGradientDescent import StochasticGradientDescent
from Models.WidrowHoff import WidrowHoff
from Utils import Measures, Predictions, Constants
from Utils import Util


def experiment13Main():
    """
    This function performs an experiment to evaluate various online regression models, it measures the performance using
    R-squared on each 10 iterations (in other words, on each streamed 10 data points).
    In addition to that, this experiment plots the convergence behavior of the algorithms on each 10 points
    and stores them in the Constants.plotting_path = 'C:/data/plots/'
    Please note that the plots are only for the first 200 points, after that all algorithms already converged
    and no need to plot them

    - Dataset: Synthetic dataset with specified parameters. DS9
    - Models:
        The following algorithms are evaluated:
        - OLR_WA (Online Linear Regression with Weighted Averaging)
        - Stochastic Gradient Descent
        - Mini-Batch Gradient Descent
        - Widrow-Hoff
        - Online Ridge Regression
        - Online Lasso Regression
        - Recursive Least Squares (RLS)
        - Online Passive-Aggressive (PA)
    - Metrics: Mean R2 score

    Args:
        None (Uses predefined parameters and models)

    Returns:
        None (Prints model results with mean R2 scores and execution times)

    Generated Data Set Details:
        - Number of samples: 1000
        - Number of features: 1
        - Noise level: 20
        - Random seeds: 0
    """
    n_samples = 1000
    n_features = 1
    noise = 20
    random_seed = 0

    Util.create_directory(Constants.plotting_path)
    X, y = SyntheticDS.create_dataset(n_samples=n_samples, n_features=n_features, noise=noise, shuffle=True,
                                      random_state=random_seed)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

    # Util.print_header('OLR_WA - ONLINE REGRESSION WEIGHTED AVG - PER ITERATION')
    # w = OLR_WA.olr_wa_plot_convergence(X_train, y_train, X_test, y_test,
    #                                    Hyperparameter.olr_wa_w_base,
    #                                    Hyperparameter.olr_wa_w_inc,
    #                                    Hyperparameter.olr_wa_base_model_size0,
    #                                    Hyperparameter.olr_wa_increment_size(n_features, user_defined_val=10),
    #                                    model_name=Constants.MODEL_NAME_OLW_WA)
    #
    # predicted_y_test = Predictions._compute_predictions__(X_test, w)
    # acc = Measures.r2_score_(y_test, predicted_y_test)
    # print('OLR_WA - ONLINE REGRESSION WEIGHTED AVG - FINAL ACC: ', acc)
    #
    Util.print_header('SGD - STOCHASTIC GRADIENT DESCENT - PER ITERATION')
    w, bias = StochasticGradientDescent.stochastic_gradient_descent_plot_convergence(X_train, y_train,
                                                                                 Hyperparameter.gd_stochastic_epochs(
                                                                                     n_samples, 2),
                                                                                 Hyperparameter.gd_learning_rate,
                                                                                 X_test, y_test,
                                                                                 model_name=Constants.MODEL_NAME_SGD)
    predicted_y_test = Predictions.compute_predictions_(X_test, w, bias)
    acc = Measures.r2_score_(y_test, predicted_y_test)
    print('SGD - STOCHASTIC GRADIENT DESCENT - FINAL ACC: ', acc)

    Util.print_header('MBGD - MINI-BATCH GRADIENT DESCENT - PER ITERATION')
    w, bias = MiniBatchGradientDescent.mini_batch_stochastic_gradient_descent_plot_convergence(X_train,
                                                                               y_train,
                                                                               Hyperparameter.gd_mini_batch_epochs(
                                                                                   n_samples,
                                                                                   n_features,
                                                                                   5),
                                                                               Hyperparameter.gd_mini_batch_size(
                                                                                   n_features,
                                                                                   user_defined_val=10),
                                                                               Hyperparameter.gd_learning_rate,
                                                                               X_test, y_test,
                                                                               model_name=Constants.MODEL_NAME_MBGD)
    predicted_y_test = Predictions.compute_predictions_(X_test, w, bias)
    acc = Measures.r2_score_(y_test, predicted_y_test)
    print('MBGD - MINI-BATCH GRADIENT DESCENT - FINAL ACC: ', acc)

    # Util.print_header('LMS - WIDROW-HOFF - PER ITERATION')
    # w = WidrowHoff.widrow_hoff_plot_convergence(X_train, y_train,
    #                                             Hyperparameter.wf_learning_rate,
    #                                             X_test, y_test, model_name=Constants.MODEL_NAME_LMS)
    # y_predicted = Predictions._compute_predictions_(X_test, w)
    # acc = Measures.r2_score_(y_test, y_predicted)
    # print('LMS - WIDROW-HOFF - FINAL ACC:', acc)
    #
    # Util.print_header('ORR - ONLINE RIDGE REGRESSION - PER ITERATION')
    # w, bias = OnlineRidgeRegression.online_ridge_regression_plot_convergence(X_train, y_train,
    #                                                                  Hyperparameter.ridge_lasso_learning_rate,
    #                                                                  Hyperparameter.ridge_lasso_epochs(
    #                                                                      n_samples, 2),
    #                                                                  Hyperparameter.ridge_lasso_regularization_param,
    #                                                                  X_test, y_test,
    #                                                                  model_name=Constants.MODEL_NAME_ORR)
    # y_predicted = Predictions.compute_predictions_(X_test, w, bias)
    # acc = Measures.r2_score_(y_test, y_predicted)
    # print('ORR - ONLINE RIDGE REGRESSION - FINAL ACC:', acc)
    #
    # Util.print_header('ORR - ONLINE LASSO REGRESSION - PER ITERATION')
    # w, bias = OnlineLassoRegression.online_lasso_regression_plot_convergence(X_train, y_train,
    #                                                                  Hyperparameter.ridge_lasso_learning_rate,
    #                                                                  Hyperparameter.ridge_lasso_epochs(
    #                                                                      n_samples, 2),
    #                                                                  Hyperparameter.ridge_lasso_regularization_param,
    #                                                                  X_test, y_test,
    #                                                                  model_name=Constants.MODEL_NAME_OLR)
    # y_predicted = Predictions.compute_predictions_(X_test, w, bias)
    # acc = Measures.r2_score_(y_test, y_predicted)
    # print('ORR - ONLINE LASSO REGRESSION - FINAL ACC:', acc)
    #
    # Util.print_header('RLS - RECURSIVE LEAST SQUARES - PER ITERATION')
    # w = RLS.rls_plot_convergence(X_train, y_train,
    #                              Hyperparameter.rls_lambda_,
    #                              Hyperparameter.rls_delta,
    #                              X_test, y_test, model_name=Constants.MODEL_NAME_RLS)
    # y_predicted = Predictions.compute_predictions(X_test, w)
    # acc = Measures.r2_score_(y_test, y_predicted)
    # print('RLS - RECURSIVE LEAST SQUARES - FINAL ACC:', acc)
    #
    # Util.print_header('PA - ONLINE PASSIVE AGGRESSIVE - PER ITERATION')
    # w = OnlinePassiveAggressive.online_passive_aggressive_plot_convergence(X_train, y_train, Hyperparameter.pa_C,
    #                                                                        Hyperparameter.pa_epsilon, X_test, y_test,
    #                                                                        model_name=Constants.MODEL_NAME_PA)
    # y_predicted = Predictions.compute_predictions(X_test, w)
    # acc = Measures.r2_score_(y_test, y_predicted)
    # print('PA - ONLINE PASSIVE AGGRESSIVE - FINAL ACC:', acc)


if __name__ == '__main__':
    experiment13Main()


