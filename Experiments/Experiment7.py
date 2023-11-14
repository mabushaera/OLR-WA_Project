import time
import warnings
import numpy as np
from Datasets import PublicDS
from Hyperparameters import Hyperparameter
from Models.MiniBatchGradientDescent import MiniBatchGradientDescent
from Models.OLR_WA import OLR_WA
from Models.BatchRegression import BatchRegression
from Models.OnlineLassoRegression import OnlineLassoRegression
from Models.OnlinePassiveAggressive import OnlinePassiveAggressive
from Models.OnlineRidgeRegression import OnlineRidgeRegression
from Models.OnlineSVR import OnlineSVR
from Models.RLS import RLS
from Models.StochasticGradientDescent import StochasticGradientDescent
from Models.WidrowHoff import WidrowHoff
from Utils import Util, Constants

warnings.filterwarnings("ignore")


def experiment7Main():
    """
        This function performs an experiment to evaluate various online regression models in addition to the batch
        regression model using K-Fold cross-validation on a public medical cost personal dataset.
        It calculates and displays the mean R2 scores and execution times for each model over multiple seed values.

        - Dataset: King County Houses Prices
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
            - Online Support Vector Regression (OSVR)
            - Online Passive-Aggressive (PA)
        - Metrics: Mean R2 score and average execution time over multiple seeds and folds.

        Args:
            None (Uses predefined parameters and models)

        Returns:
            None (Prints model results with mean R2 scores and execution times)

        Dataset Details:
            - Dataset path: https://www.kaggle.com/datasets/minasameh55/king-country-houses-aa
              downloaded and exists in the project files under OLR_WA_Project\Datasets\Datasets_Generators_CSV\07_KCHSD\
              007_kc_house_data.csv
            - X: Features matrix: sqft_living, grade, sqft_above, sqft_living15, bedrooms, bathrooms, view, price
            - y: Target variable: Price
        """
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
    osvr_list_per_seed = []
    pa_list_per_seed = []

    batch_execution_time = 0
    olr_wa_execution_time = 0
    stochastic_batch_gradient_execution_time = 0
    mini_batch_gradient_descent_execution_time = 0
    widrow_hoff_execution_time = 0
    online_ridge_regression_execution_time = 0
    online_lasso_regression_execution_time = 0
    rls_execution_time = 0
    osvr_execution_time = 0
    pa_execution_time = 0

    path = Util.get_dataset_path('07_KCHSD\\007_kc_house_data.csv')
    X, y = PublicDS.get_king_county_house_sales_data(path)
    n_samples, n_features = X.shape
    # Experiments
    for seed in SEEDS:
        # # 1. Batch Experiment
        # start_time = time.perf_counter()
        # batch_acc = BatchRegression.batch_regression_KFold(X, y, seed)
        # end_time = time.perf_counter()
        # batch_execution_time += (end_time - start_time)
        # batch_acc_list_per_seed.append(batch_acc)
        #
        # # 2. OLR_WA Experiment.
        # start_time = time.perf_counter()
        # olr_wa_acc = OLR_WA.olr_wa_regression_KFold(X, y,
        #                                       Hyperparameter.olr_wa_w_base1,
        #                                       Hyperparameter.olr_wa_w_inc1,
        #                                       Hyperparameter.olr_wa_base_model_size1,
        #                                       Hyperparameter.olr_wa_increment_size(n_features, user_defined_val=10),
        #                                       seed)
        # end_time = time.perf_counter()
        # olr_wa_execution_time += (end_time - start_time)
        # olr_wa_acc_list_per_seed.append(olr_wa_acc)
        #
        # # 3. Online Linear Regression using stochastic gradient descent
        # start_time = time.perf_counter()
        # stochastic_batch_gradient_descent_acc = StochasticGradientDescent.stochastic_gradient_descent_KFold(X, y,
        #                                                                           Hyperparameter.gd_stochastic_epochs(
        #                                                                               n_samples, 2),
        #                                                                           Hyperparameter.gd_learning_rate3,
        #                                                                           seed)
        # end_time = time.perf_counter()
        # stochastic_batch_gradient_execution_time += (end_time - start_time)
        # stochastic_gradient_descent_acc_list_per_seed.append(stochastic_batch_gradient_descent_acc)
        #
        # # 4. Online Linear Regression using mini-batch gradient descent
        # start_time = time.perf_counter()
        # mini_batch_gradient_descent_acc = MiniBatchGradientDescent.mini_batch_gradient_descent_KFold(X, y,
        #                                                                            Hyperparameter.gd_mini_batch_epochs(
        #                                                                                n_samples, n_features, 20),
        #                                                                            Hyperparameter.gd_mini_batch_size(
        #                                                                                n_features,
        #                                                                                user_defined_val=10),
        #                                                                            Hyperparameter.gd_learning_rate,
        #                                                                            seed)
        # end_time = time.perf_counter()
        # mini_batch_gradient_descent_execution_time += (end_time - start_time)
        # mini_batch_gradient_descent_acc_list_per_seed.append(mini_batch_gradient_descent_acc)
        #
        # # 5. Widrow-Hoff
        # start_time = time.perf_counter()
        # widrow_hoff_acc = WidrowHoff.widrow_hoff_KFold(X, y,
        #                                                Hyperparameter.wf_learning_rate3
        #                                                , seed)
        # end_time = time.perf_counter()
        # widrow_hoff_execution_time += (end_time - start_time)
        # widrow_hoff_acc_list_per_seed.append(widrow_hoff_acc)
        #
        # 6. Online Ridge Regression
        # start_time = time.perf_counter()
        # online_ridge_regression_acc = OnlineRidgeRegression.online_ridge_regression_KFold(X, y,
        #                                                             Hyperparameter.ridge_lasso_learning_rate,
        #                                                             Hyperparameter.ridge_lasso_epochs(
        #                                                                 n_samples, 10),
        #                                                             Hyperparameter.ridge_lasso_regularization_param3,
        #                                                             seed)
        # online_ridge_regression_acc = OnlineRidgeRegression.online_ridge_regression_KFold(X, y,
        #                                                                                   Hyperparameter.ridge_lasso_learning_rate2,
        #                                                                                   Hyperparameter.ridge_lasso_epochs(
        #                                                                                       n_samples, 5),
        #                                                                                   Hyperparameter.ridge_lasso_regularization_param4,
        #                                                                                   seed)
        # end_time = time.perf_counter()
        # online_ridge_regression_execution_time += (end_time - start_time)
        # online_ridge_regression_list_per_seed.append(online_ridge_regression_acc)
        #
        # # 7. Online Lasso Regression
        start_time = time.perf_counter()
        online_lasso_regression_acc = OnlineLassoRegression.online_lasso_regression_KFold(X, y,
                                                                    Hyperparameter.ridge_lasso_learning_rate2,
                                                                    Hyperparameter.ridge_lasso_epochs(
                                                                        n_samples, 5),
                                                                    Hyperparameter.ridge_lasso_regularization_param5,
                                                                    seed)
        end_time = time.perf_counter()
        online_lasso_regression_execution_time += (end_time - start_time)
        online_lasso_regression_list_per_seed.append(online_lasso_regression_acc)
        #
        # # 8. RLS
        # start_time = time.perf_counter()
        # rls_acc = RLS.rls_KFold(X, y, Hyperparameter.rls_lambda_, Hyperparameter.rls_delta, seed)
        # end_time = time.perf_counter()
        # rls_execution_time += (end_time - start_time)
        # rls_list_per_seed.append(rls_acc)
        #
        # # # 9. OSVR
        # # start_time = time.perf_counter()
        # # osvr_acc = OnlineSVR.osvr(X, y, Hyperparameter.osvr_C, Hyperparameter.osvr_eps,
        # #                           Hyperparameter.osvr_kernelParam,
        # #                           Hyperparameter.osvr_bias, seed)
        # # end_time = time.perf_counter()
        # # osvr_execution_time += (end_time - start_time)
        # # osvr_list_per_seed.append(osvr_acc)

        # # 10. PA
        # start_time = time.perf_counter()
        # pa_acc = OnlinePassiveAggressive.online_passive_aggressive_KFold(X, y,
        #                                                            Hyperparameter.pa_C2,
        #                                                            Hyperparameter.pa_epsilon2,
        #                                                            seed)
        # end_time = time.perf_counter()
        # pa_execution_time += (end_time - start_time)
        # pa_list_per_seed.append(pa_acc)

    # # 1. Results for Batch Experiment:
    # batch_acc = np.array(batch_acc_list_per_seed).mean()
    # print('Batch (Pseudo-Inverse), 5 folds, seeds averaging. time:',
    #       "{:.5f} s".format(batch_execution_time / total_runs), ', R2:', "{:.5f}".format(batch_acc))
    #
    # # 2. Results for OLR_WA Experiment
    # olr_wa_acc = np.array(olr_wa_acc_list_per_seed).mean()
    # print('OLR_WA, 5 folds, seeds averaging. time:', "{:.5f} s".format(olr_wa_execution_time / total_runs), ', R2:',
    #       "{:.5f}".format(olr_wa_acc))
    #
    # # 3. Results for Stochastic Gradient Descent Experiment
    # stochastic_gradient_descent_acc = np.array(stochastic_gradient_descent_acc_list_per_seed).mean()
    # print('Stochastic Gradient Descent, 5 folds, seeds averaging. time:',
    #       "{:.5f} s".format(stochastic_batch_gradient_execution_time / total_runs), ', R2:',
    #       "{:.5f}".format(stochastic_gradient_descent_acc))
    #
    # # 4. Results for Mini-Batch Gradient Descent Experiment
    # mini_batch_gradient_descent_acc = np.array(mini_batch_gradient_descent_acc_list_per_seed).mean()
    # print('Mini-Batch Gradient Descent, 5 folds, seeds averaging. time:',
    #       "{:.5f} s".format(mini_batch_gradient_descent_execution_time / total_runs), ', R2:',
    #       "{:.5f}".format(mini_batch_gradient_descent_acc))
    #
    # # 5. Results for Widrow_Hoff Experiment:
    # widrow_hoff_acc = np.array(widrow_hoff_acc_list_per_seed).mean()
    # print('Widrow-Hoff, 5 folds, seeds averaging. time:', "{:.5f} s".format(widrow_hoff_execution_time / total_runs),
    #       ', R2:', "{:.5f}".format(widrow_hoff_acc))
    #
    # # 6. Results for Online Ridge Regression Experiment:
    # online_ridge_regression_acc = np.array(online_ridge_regression_list_per_seed).mean()
    # print('Online Ridge Regression, 5 folds, seeds averaging. time:',
    #       "{:.5f} s".format(online_ridge_regression_execution_time / total_runs), ', R2:',
    #       "{:.5f}".format(online_ridge_regression_acc))
    #
    # 7. Results for Online Lasso Regression Experiment:
    online_lasso_regression_acc = np.array(online_lasso_regression_list_per_seed).mean()
    print('Online Lasso Regression, 5 folds, seeds averaging. time:',
          "{:.5f} s".format(online_lasso_regression_execution_time / total_runs), ', R2:',
          "{:.5f}".format(online_lasso_regression_acc))
    #
    # # 8. Results for RLS Experiment:
    # rls_acc = np.array(rls_list_per_seed).mean()
    # print('Recursive Least Squares (RLS), 5 folds, seeds averaging. time:',
    #       "{:.5f} s".format(rls_execution_time / total_runs), ', R2:', "{:.5f}".format(rls_acc))
    #
    # # # 9. Results for OSVR Experiment:
    # # osvr_acc = np.array(osvr_list_per_seed).mean()
    # # print('Online Support Vector Regression, 5 folds, seeds averaging. time:',
    # #       "{:.5f} s".format(osvr_execution_time / total_runs), ', R2:', "{:.5f}".format(osvr_acc))

    # # 10. Results for PA Experiment:
    # pa_acc = np.array(pa_list_per_seed).mean()
    # print('Online Passive-Aggressive (PA), 5 folds, seeds averaging. time:',
    #       "{:.5f} s".format(pa_execution_time / total_runs), ', R2:', "{:.5f}".format(pa_acc))


if __name__ == '__main__':
    experiment7Main()
