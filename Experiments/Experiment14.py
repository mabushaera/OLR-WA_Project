import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from Hyperparameters import Hyperparameter
from Models.MiniBatchGradientDescent import MiniBatchGradientDescent
from Models.OLR_WA import OLR_WA
from Models.OnlineLassoRegression import OnlineLassoRegression
from Models.OnlinePassiveAggressive import OnlinePassiveAggressive
from Models.OnlineRidgeRegression import OnlineRidgeRegression
from Models.RLS import RLS
from Models.StochasticGradientDescent import StochasticGradientDescent
from Models.WidrowHoff import WidrowHoff
from Utils import Util, Constants
import warnings

warnings.filterwarnings("ignore")

def experiment14Main():
    """
    The experiment generates a figure of all models showing the convergence behavior using the MSE VS. the number of
    data points seen so far.
    Note that the experiment is conducted to take the mean results over 5 folds, 5 seeds.

    - Dataset: Synthetic dataset with specified parameters. DS10
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
        - Random seeds: Defined in the Constants module (SEEDS)
    """
    n_samples = 1000
    n_features = 2
    noise = 20
    SEEDS = Constants.SEEDS

    olr_wa_acc_list_per_seed = []
    stochastic_gradient_descent_acc_list_per_seed = []
    mini_batch_gradient_descent_acc_list_per_seed = []
    widrow_hoff_acc_list_per_seed = []
    online_ridge_regression_list_per_seed = []
    online_lasso_regression_list_per_seed = []
    rls_list_per_seed = []
    pa_list_per_seed = []

    olr_wa_general_epochs = []
    olr_wa_general_loss = []
    stochastic_gradient_descent_general_epochs = []
    stochastic_gradient_descent_general_loss = []
    mini_batch_gradient_descent_general_epochs = []
    mini_batch_gradient_descent_general_loss = []
    online_ridge_regression_general_epochs = []
    online_ridge_regression_general_loss = []
    online_lasso_regression_general_epochs = []
    online_lasso_regression_general_loss = []
    widrow_hoff_general_epochs = []
    widrow_hoff_general_loss = []
    rls_general_epochs = []
    rls_general_loss = []
    pa_general_epochs = []
    pa_general_loss = []

    # Experiments
    for seed in SEEDS:
        X, y = datasets.make_regression(n_samples=n_samples, n_features=n_features, noise=noise, shuffle=True,
                                        random_state=seed)

        # 1. OLR_WA Experiment.
        olr_wa_acc, olr_wa_epoch_list_per_seed, olr_wa_cost_list_per_seed = OLR_WA.olr_wa_regression_convergence(X, y,
                                                                                 Hyperparameter.olr_wa_w_base,
                                                                                 Hyperparameter.olr_wa_w_inc,
                                                                                 Hyperparameter.olr_wa_base_model_size0,
                                                                                 Hyperparameter.olr_wa_increment_size(
                                                                                     n_features,
                                                                                     user_defined_val=10))
        olr_wa_acc_list_per_seed.append(olr_wa_acc)
        olr_wa_general_loss = Util.sum_lists_element_wise(olr_wa_general_loss, olr_wa_cost_list_per_seed)
        olr_wa_general_epochs = Util.sum_lists_element_wise(olr_wa_general_epochs, olr_wa_epoch_list_per_seed)

        # 2. Online Linear Regression using stochastic gradient descent
        stochastic_batch_gradient_descent_acc, sgd_epoch_list_per_seed, sgd_cost_list_per_seed = \
            StochasticGradientDescent.stochastic_gradient_descent_convergence(
            X, y,
            Hyperparameter.gd_stochastic_epochs(n_samples, 2),
            Hyperparameter.gd_learning_rate)
        stochastic_gradient_descent_acc_list_per_seed.append(stochastic_batch_gradient_descent_acc)
        stochastic_gradient_descent_general_loss = Util.sum_lists_element_wise(stochastic_gradient_descent_general_loss,
                                                                               sgd_cost_list_per_seed)
        stochastic_gradient_descent_general_epochs = Util.sum_lists_element_wise(
            stochastic_gradient_descent_general_epochs, sgd_epoch_list_per_seed)

        # 3. Online Linear Regression using mini-batch gradient descent
        mini_batch_gradient_descent_acc, mbgd_epoch_list_per_seed, mbgd_cost_list_per_seed = \
            MiniBatchGradientDescent.mini_batch_gradient_descent_convergence(
            X, y,
            Hyperparameter.gd_mini_batch_epochs(n_samples, n_features, 5),
            Hyperparameter.gd_mini_batch_size2(n_features, user_defined_val=10),
            Hyperparameter.gd_learning_rate)
        mini_batch_gradient_descent_acc_list_per_seed.append(mini_batch_gradient_descent_acc)
        mini_batch_gradient_descent_general_loss = Util.sum_lists_element_wise(mini_batch_gradient_descent_general_loss,
                                                                               mbgd_cost_list_per_seed)
        mini_batch_gradient_descent_general_epochs = Util.sum_lists_element_wise(
            mini_batch_gradient_descent_general_epochs, mbgd_epoch_list_per_seed)

        # 4. Widrow-Hoff
        widrow_hoff_acc, widrow_hoff_epoch_list_per_seed, widrow_hoff_cost_list_per_seed = \
            WidrowHoff.widrow_hoff_convergence(X, y, Hyperparameter.wf_learning_rate)
        widrow_hoff_acc_list_per_seed.append(widrow_hoff_acc)
        widrow_hoff_general_loss = Util.sum_lists_element_wise(widrow_hoff_general_loss, widrow_hoff_cost_list_per_seed)
        widrow_hoff_general_epochs = Util.sum_lists_element_wise(widrow_hoff_general_epochs,
                                                                 widrow_hoff_epoch_list_per_seed)

        # 5. Online Ridge Regression
        online_ridge_regression_acc, online_ridge_regression_epoch_list_per_seed, \
            online_ridge_regression_cost_list_per_seed = OnlineRidgeRegression.online_ridge_regression_convergence(X, y,
                                                                      Hyperparameter.ridge_lasso_learning_rate,
                                                                      Hyperparameter.ridge_lasso_epochs(n_samples, 2),
                                                                      Hyperparameter.ridge_lasso_regularization_param)
        online_ridge_regression_list_per_seed.append(online_ridge_regression_acc)
        online_ridge_regression_general_loss = Util.sum_lists_element_wise(online_ridge_regression_general_loss,
                                                                           online_ridge_regression_cost_list_per_seed)
        online_ridge_regression_general_epochs = Util.sum_lists_element_wise(online_ridge_regression_general_epochs,
                                                                             online_ridge_regression_epoch_list_per_seed)

        # 6. Online Lasso Regression
        online_lasso_regression_acc, online_lasso_regression_epoch_list_per_seed, \
            online_lasso_regression_cost_list_per_seed = OnlineLassoRegression.online_lasso_regression_convergence(X, y,
                                                                      Hyperparameter.ridge_lasso_learning_rate,
                                                                      Hyperparameter.ridge_lasso_epochs(n_samples, 2),
                                                                      Hyperparameter.ridge_lasso_regularization_param)
        online_lasso_regression_list_per_seed.append(online_lasso_regression_acc)
        online_lasso_regression_general_loss = Util.sum_lists_element_wise(online_lasso_regression_general_loss,
                                                                           online_lasso_regression_cost_list_per_seed)
        online_lasso_regression_general_epochs = Util.sum_lists_element_wise(online_lasso_regression_general_epochs,
                                                                             online_lasso_regression_epoch_list_per_seed)

        # 7. RLS
        rls_acc, rls_epoch_list_per_seed, rls_cost_list_per_seed = RLS.rls_convergence(X, y, Hyperparameter.rls_lambda_,
                                                                                       Hyperparameter.rls_delta)
        rls_list_per_seed.append(rls_acc)
        rls_general_loss = Util.sum_lists_element_wise(rls_general_loss, rls_cost_list_per_seed)
        rls_general_epochs = Util.sum_lists_element_wise(rls_general_epochs, rls_epoch_list_per_seed)

        # 8. PA
        pa_acc, pa_epoch_list_per_seed, pa_cost_list_per_seed = \
            OnlinePassiveAggressive.online_passive_aggressive_convergence(X, y,
                                                                        Hyperparameter.pa_C,
                                                                        Hyperparameter.pa_epsilon)
        pa_list_per_seed.append(pa_acc)
        pa_general_loss = Util.sum_lists_element_wise(pa_general_loss, pa_cost_list_per_seed)
        pa_general_epochs = Util.sum_lists_element_wise(pa_general_epochs, pa_epoch_list_per_seed)

    # 1. Results for OLR_WA Experiment
    olr_wa_acc = np.array(olr_wa_acc_list_per_seed).mean()
    print('Final OLR_WA Acc, 5 folds, seeds averaging', olr_wa_acc)
    olr_wa_general_epochs = olr_wa_general_epochs / len(SEEDS)
    olr_wa_general_loss = olr_wa_general_loss / len(SEEDS)

    # 2. Results for Stochastic Gradient Descent Experiment
    stochastic_gradient_descent_acc = np.array(stochastic_gradient_descent_acc_list_per_seed).mean()
    print('Final Stochastic Gradient Descent Acc, 5 folds, seeds averaging', stochastic_gradient_descent_acc)
    stochastic_gradient_descent_general_epochs = stochastic_gradient_descent_general_epochs / len(SEEDS)
    stochastic_gradient_descent_general_loss = stochastic_gradient_descent_general_loss / len(SEEDS)

    # 3. Results for Mini-Batch Gradient Descent Experiment
    mini_batch_gradient_descent_acc = np.array(mini_batch_gradient_descent_acc_list_per_seed).mean()
    print('Final Mini-Batch Gradient Descent Acc, 5 folds, seeds averaging', mini_batch_gradient_descent_acc)
    mini_batch_gradient_descent_general_epochs = mini_batch_gradient_descent_general_epochs / len(SEEDS)
    mini_batch_gradient_descent_general_loss = mini_batch_gradient_descent_general_loss / len(SEEDS)

    # 4. Results for Widrow_Hoff Experiment
    widrow_acc = np.array(widrow_hoff_acc_list_per_seed).mean()
    print('Final Widrow-Hoff Acc, 5 folds, seeds averaging', widrow_acc)
    widrow_hoff_general_epochs = widrow_hoff_general_epochs / len(SEEDS)
    widrow_hoff_general_loss = widrow_hoff_general_loss / len(SEEDS)

    # 5. Results for Online Ridge Regression Experiment
    online_ridge_regression_acc = np.array(online_ridge_regression_list_per_seed).mean()
    print('Final Online Ridge Regression Acc, 5 folds, seeds averaging', online_ridge_regression_acc)
    online_ridge_regression_general_epochs = online_ridge_regression_general_epochs / len(SEEDS)
    online_ridge_regression_general_loss = online_ridge_regression_general_loss / len(SEEDS)

    # 6. Results for Online Lasso Regression Experiment
    online_lasso_regression_acc = np.array(online_lasso_regression_list_per_seed).mean()
    print('Final Online Lasso Regression Acc, 5 folds, seeds averaging', online_lasso_regression_acc)
    online_lasso_regression_general_epochs = online_lasso_regression_general_epochs / len(SEEDS)
    online_lasso_regression_general_loss = online_lasso_regression_general_loss / len(SEEDS)

    # 7. Results for RLS Experiment
    rls_acc = np.array(rls_list_per_seed).mean()
    print('Final RLS Acc, 5 folds, seeds averaging', rls_acc)
    rls_general_epochs = rls_general_epochs / len(SEEDS)
    rls_general_loss = rls_general_loss / len(SEEDS)

    # 8. Results for PA Experiment
    pa_acc = np.array(pa_list_per_seed).mean()
    print('Final Online Passive-Aggressive (PA) Acc, 5 folds, seeds averaging', pa_acc)
    pa_general_epochs = pa_general_epochs / len(SEEDS)
    pa_general_loss = pa_general_loss / len(SEEDS)

    # if n_features == 1:
    # Plotting Cost Fn Convergence:
    plt.xlabel('Number of Data Points')
    plt.ylabel('Cost')
    plt.title('Cost Function Comparison for Multiple Seeds, 5 Folds Cross Validation')

    x = np.linspace(0, 1000, 100)
    y = np.sin(x)

    # to limit the plot to  the first 20 values stored.
    plot_limit = 17

    # Plot loss function values over seen data points.
    plt.plot(olr_wa_general_epochs, olr_wa_general_loss,
             label=Constants.MODEL_NAME_OLW_WA + Constants.OLR_WA_EXP14_FORMATTED_PARAMETERS)
    plt.plot(stochastic_gradient_descent_general_epochs[:plot_limit],
             stochastic_gradient_descent_general_loss[:plot_limit],
             label=Constants.MODEL_NAME_SGD + Constants.SGD_EXP14_FORMATTED_PARAMETERS)
    plt.plot(mini_batch_gradient_descent_general_epochs[:plot_limit],
             mini_batch_gradient_descent_general_loss[:plot_limit],
             label=Constants.MODEL_NAME_MBGD + Constants.MBGD_EXP14_FORMATTED_PARAMETERS)
    plt.plot(widrow_hoff_general_epochs[:plot_limit], widrow_hoff_general_loss[:plot_limit],
             label=Constants.MODEL_NAME_LMS + Constants.LMS_EXP14_FORMATTED_PARAMETERS)
    plt.plot(online_ridge_regression_general_epochs[:plot_limit],
             online_ridge_regression_general_loss[:plot_limit],
             label=Constants.MODEL_NAME_ORR + Constants.ORR_EXP14_FORMATTED_PARAMETERS)
    plt.plot(online_lasso_regression_general_epochs[:plot_limit],
             online_lasso_regression_general_loss[:plot_limit],
             label=Constants.MODEL_NAME_OLR + Constants.OLR_EXP14_FORMATTED_PARAMETERS)
    plt.plot(rls_general_epochs[:plot_limit], rls_general_loss[:plot_limit],
             label=Constants.MODEL_NAME_RLS + Constants.RLS_EXP14_FORMATTED_PARAMETERS)
    plt.plot(pa_general_epochs[:plot_limit], pa_general_loss[:plot_limit],
             label=Constants.MODEL_NAME_PA + Constants.PA_EXP14_FORMATTED_PARAMETERS)

    # Customize x-axis ticks
    step = 50
    plt.xticks(np.arange(0, 1000, step))

    # Rest of your plotting code
    legend = plt.legend()
    for label in legend.get_texts():
        # Set the font size for the legend labels
        label.set_fontsize(8)
    plt.show()


if __name__ == '__main__':
    experiment14Main()




