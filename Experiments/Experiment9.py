import time
import numpy as np
from Datasets import SyntheticDS
from Hyperparameters import Hyperparameter
from Models.MiniBatchGradientDescent import MiniBatchGradientDescent
from Models.OLR_WA import OLR_WA
from Models.BatchRegression import BatchRegression
from Models.OnlineLassoRegression import OnlineLassoRegression
from Models.OnlinePassiveAggressive import OnlinePassiveAggressive
from Models.OnlineRidgeRegression import OnlineRidgeRegression
from Models.RLS import RLS
from Models.StochasticGradientDescent import StochasticGradientDescent
from Models.WidrowHoff import WidrowHoff
from Utils import Util, Constants


def experiment9Main():
    """
    This function performs an experiment to evaluate various online regression models in addition to the batch
    regression model using K-Fold cross-validation. This experiment is designed to measure the models performance
    on adversarial scenarios, more specifically on Time-Based scenarios where the algorithm suppose to adapt to the
    new data, in this case, the test-data is a portion on the new data.
    It calculates and displays the mean R2 scores and execution times for each model over multiple seed values.

    - Dataset: Synthetic dataset with specified parameters. DS5
    - Models:
        The following algorithms are evaluated:
        - Batch Regression (Pseudo-Inverse)
        - OLR_WA (Online Linear Regression with Weighted Averaging)
        - Stochastic Gradient Descent
        - Mini-Batch Gradient Descent
        - Widrow-Hoff
        - Online Ridge Regression
        - Online Lasso Regression
        - Recursive Least Squares (RLS)
        - Online Passive-Aggressive (PA)
    - Metrics: Mean R2 score and average execution time over multiple seeds and folds.

    Args:
        None (Uses predefined parameters and models)

    Returns:
        None (Prints model results with mean R2 scores and execution times)

    Generated Data Set Details:
        - Number of samples: 5000
        - Number of features: 20
        - Noise level: 20
        - Random seeds: Defined in the Constants module (SEEDS)
    """
    n_samples = 2500
    total_n_samples = n_samples * 2  # total n_samples from both combined datasets
    n_features = 20
    noise = 20
    SEEDS = Constants.SEEDS

    number_of_seeds = len(SEEDS)
    number_of_folds = 5
    total_runs = number_of_seeds * number_of_folds

    batch_acc_list_per_seed = []
    olr_wa_acc_list_per_seed = []
    stochastic_gradient_descent_acc_list_per_seed = []
    mini_batch_gradient_descent_acc_list_per_seed = []
    widrow_hoff_acc_list_per_seed = []
    online_ridge_regression_list_per_seed = []
    online_lasso_regression_list_per_seed = []
    rls_list_per_seed = []
    pa_list_per_seed = []

    batch_execution_time = 0
    olr_wa_execution_time = 0
    stochastic_batch_gradient_execution_time = 0
    mini_batch_gradient_descent_execution_time = 0
    widrow_hoff_execution_time = 0
    online_ridge_regression_execution_time = 0
    online_lasso_regression_execution_time = 0
    rls_execution_time = 0
    pa_execution_time = 0

    # 1. Experiments
    for seed in SEEDS:
        X2, y2 = SyntheticDS.create_dataset(n_samples=n_samples, n_features=n_features, noise=noise, shuffle=False,
                                            random_state=seed)
        X2 *= -1
        X1, y1 = SyntheticDS.create_dataset(n_samples=n_samples, n_features=n_features, noise=noise, shuffle=False,
                                            random_state=seed)

        train_percent = int(80 * n_samples / 100)
        X2_train = X2[:train_percent]
        y2_train = y2[:train_percent]
        X_test = X2[train_percent:]
        y_test = y2[train_percent:]

        X, y = Util.combine_two_datasets(X1, y1, X2_train, y2_train)

        # 1. Batch Experiment
        start_time = time.perf_counter()
        batch_acc = BatchRegression.batch_regression_adversarial(X, y, X_test, y_test)
        end_time = time.perf_counter()
        batch_execution_time += (end_time - start_time)
        batch_acc_list_per_seed.append(batch_acc)

        # 2. OLR_WA Experiment.
        start_time = time.perf_counter()
        olr_wa_acc = OLR_WA.olr_wa_regression_adversarial(X, y,
                                                          Hyperparameter.olr_wa_w_base_adv1,
                                                          Hyperparameter.olr_wa_w_inc_adv1,
                                                          Hyperparameter.olr_wa_base_model_size1,
                                                          Hyperparameter.olr_wa_increment_size(n_features,
                                                                                               user_defined_val=10),
                                                          X_test,
                                                          y_test)
        end_time = time.perf_counter()
        olr_wa_execution_time += (end_time - start_time)
        olr_wa_acc_list_per_seed.append(olr_wa_acc)

        # 3. Online Linear Regression using stochastic gradient descent
        start_time = time.perf_counter()
        stochastic_batch_gradient_descent_acc = StochasticGradientDescent.stochastic_gradient_descent_adversarial(X, y,
                                                                                  X_test,
                                                                                  y_test,
                                                                                  Hyperparameter.gd_stochastic_epochs(
                                                                                      total_n_samples,
                                                                                      2),
                                                                                  Hyperparameter.gd_learning_rate)
        end_time = time.perf_counter()
        stochastic_batch_gradient_execution_time += (end_time - start_time)
        stochastic_gradient_descent_acc_list_per_seed.append(stochastic_batch_gradient_descent_acc)

        # 4. Online Linear Regression using mini-batch gradient descent
        start_time = time.perf_counter()
        mini_batch_gradient_descent_acc = MiniBatchGradientDescent.mini_batch_gradient_descent_adversarial(X, y, X_test,
                                                                                   y_test,
                                                                                   Hyperparameter.gd_mini_batch_epochs(
                                                                                       total_n_samples,
                                                                                       n_features,
                                                                                       2),
                                                                                   Hyperparameter.gd_mini_batch_size(
                                                                                       n_features,
                                                                                       user_defined_val=10),
                                                                                   Hyperparameter.gd_learning_rate)
        end_time = time.perf_counter()
        mini_batch_gradient_descent_execution_time += (end_time - start_time)
        mini_batch_gradient_descent_acc_list_per_seed.append(mini_batch_gradient_descent_acc)

        # 5. Widrow-Hoff
        start_time = time.perf_counter()
        widrow_hoff_acc = WidrowHoff.widrow_hoff_adversarial(X, y, X_test, y_test, Hyperparameter.wf_learning_rate)
        end_time = time.perf_counter()
        widrow_hoff_execution_time += (end_time - start_time)
        widrow_hoff_acc_list_per_seed.append(widrow_hoff_acc)

        # 6. Online Ridge Regression
        start_time = time.perf_counter()
        online_ridge_regression_acc = OnlineRidgeRegression.online_ridge_regression_adversarial(X, y, X_test, y_test,
                                                                        Hyperparameter.ridge_lasso_learning_rate,
                                                                        Hyperparameter.ridge_lasso_epochs(
                                                                            n_samples, 2),
                                                                        Hyperparameter.ridge_lasso_regularization_param)
        end_time = time.perf_counter()
        online_ridge_regression_execution_time += (end_time - start_time)
        online_ridge_regression_list_per_seed.append(online_ridge_regression_acc)

        # 7. Online Lasso Regression
        start_time = time.perf_counter()
        online_lasso_regression_acc = OnlineLassoRegression.online_lasso_regression_adversarial(X, y, X_test, y_test,
                                                                        Hyperparameter.ridge_lasso_learning_rate,
                                                                        Hyperparameter.ridge_lasso_epochs(
                                                                            n_samples, 2),
                                                                        Hyperparameter.ridge_lasso_regularization_param)
        end_time = time.perf_counter()
        online_lasso_regression_execution_time += (end_time - start_time)
        online_lasso_regression_list_per_seed.append(online_lasso_regression_acc)

        # 8. RLS
        start_time = time.perf_counter()
        rls_acc = RLS.rls_adversarial(X, y, X_test, y_test, Hyperparameter.rls_lambda_, Hyperparameter.rls_delta)
        end_time = time.perf_counter()
        rls_execution_time += (end_time - start_time)
        rls_list_per_seed.append(rls_acc)

        # 9. PA
        start_time = time.perf_counter()
        pa_acc = OnlinePassiveAggressive.online_passive_aggressive_adversarial(X, y, X_test, y_test,
                                                                               Hyperparameter.pa_C,
                                                                               Hyperparameter.pa_epsilon)
        end_time = time.perf_counter()
        pa_execution_time += (end_time - start_time)
        pa_list_per_seed.append(pa_acc)

    # 1. Results for Batch Experiment:
    batch_acc = np.array(batch_acc_list_per_seed).mean()
    print('Batch (Pseudo-Inverse), 5 folds, seeds averaging. time:',
          "{:.5f} s".format(batch_execution_time / total_runs), ', R2:', "{:.5f}".format(batch_acc))

    # 2. Results for OLR_WA Experiment
    olr_wa_acc = np.array(olr_wa_acc_list_per_seed).mean()
    print('OLR_WA, 5 folds, seeds averaging. time:', "{:.5f} s".format(olr_wa_execution_time / total_runs), ', R2:',
          "{:.5f}".format(olr_wa_acc))

    # 3. Results for Stochastic Gradient Descent Experiment
    stochastic_gradient_descent_acc = np.array(stochastic_gradient_descent_acc_list_per_seed).mean()
    print('Stochastic Gradient Descent, 5 folds, seeds averaging. time:',
          "{:.5f} s".format(stochastic_batch_gradient_execution_time / total_runs), ', R2:',
          "{:.5f}".format(stochastic_gradient_descent_acc))

    # 4. Results for Mini-Batch Gradient Descent Experiment
    mini_batch_gradient_descent_acc = np.array(mini_batch_gradient_descent_acc_list_per_seed).mean()
    print('Mini-Batch Gradient Descent, 5 folds, seeds averaging. time:',
          "{:.5f} s".format(mini_batch_gradient_descent_execution_time / total_runs), ', R2:',
          "{:.5f}".format(mini_batch_gradient_descent_acc))

    # 5. Results for Widrow_Hoff Experiment:
    widrow_hoff_acc = np.array(widrow_hoff_acc_list_per_seed).mean()
    print('Widrow-Hoff, 5 folds, seeds averaging. time:', "{:.5f} s".format(widrow_hoff_execution_time / total_runs),
          ', R2:', "{:.5f}".format(widrow_hoff_acc))

    # 6. Results for Online Ridge Regression Experiment:
    online_ridge_regression_acc = np.array(online_ridge_regression_list_per_seed).mean()
    print('Online Ridge Regression, 5 folds, seeds averaging. time:',
          "{:.5f} s".format(online_ridge_regression_execution_time / total_runs), ', R2:',
          "{:.5f}".format(online_ridge_regression_acc))

    # 7. Results for Online Lasso Regression Experiment:
    online_lasso_regression_acc = np.array(online_lasso_regression_list_per_seed).mean()
    print('Online Lasso Regression, 5 folds, seeds averaging. time:',
          "{:.5f} s".format(online_lasso_regression_execution_time / total_runs), ', R2:',
          "{:.5f}".format(online_lasso_regression_acc))

    # 8. Results for RLS Experiment:
    rls_acc = np.array(rls_list_per_seed).mean()
    print('Recursive Least Squares (RLS), 5 folds, seeds averaging. time:',
          "{:.5f} s".format(rls_execution_time / total_runs), ', R2:', "{:.5f}".format(rls_acc))

    # 9. Results for PA Experiment:
    pa_acc = np.array(pa_list_per_seed).mean()
    print('Online Passive-Aggressive (PA), 5 folds, seeds averaging. time:',
          "{:.5f} s".format(pa_execution_time / total_runs), ', R2:', "{:.5f}".format(pa_acc))


if __name__ == '__main__':
    experiment9Main()



