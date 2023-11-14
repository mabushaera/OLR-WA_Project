import warnings
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
from Utils import Constants
from Utils import Util

warnings.filterwarnings("ignore")


def experiment13Main():
    """
    This function performs an experiment to evaluate various online regression models, it measures the performance using
    R-squared on each 10 iterations (in other words, on each streamed 10 data points).

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
        - Number of features: 2
        - Noise level: 20
        - Random seeds: Constants.SEEDS
    """
    n_samples = 1000
    n_features = 2
    noise = 20
    SEEDS = Constants.SEEDS

    sgd_list = []
    mbgd_list = []
    olr_wa_list = []
    widrow_hoff_list = []
    orr_list = []
    olr_list = []
    rls_list = []
    pa_list = []

    for seed in SEEDS:
        X, y = SyntheticDS.create_dataset(n_samples=n_samples, n_features=n_features, noise=noise, shuffle=True,
                                          random_state=seed)

        olr_wa_l = OLR_WA.olr_wa_convergence2(X, y,
                                              Hyperparameter.olr_wa_w_base1,
                                              Hyperparameter.olr_wa_w_inc1,
                                              Hyperparameter.olr_wa_base_model_size0,
                                              Hyperparameter.olr_wa_increment_size2(n_features, user_defined_val=10),
                                              model_name=Constants.MODEL_NAME_OLW_WA, seed=seed)

        olr_wa_list = olr_wa_list + olr_wa_l

        sgd_l = StochasticGradientDescent.stochastic_gradient_descent_convergence2(X, y,
                                                                                   Hyperparameter.gd_stochastic_epochs(
                                                                                       n_samples, 2),
                                                                                   Hyperparameter.gd_learning_rate,
                                                                                   model_name=Constants.MODEL_NAME_SGD,
                                                                                   seed=seed)

        sgd_list = sgd_list + sgd_l

        mbgd_l = MiniBatchGradientDescent.mini_batch_gradient_descent_convergence2(X,
                                                                                   y,
                                                                                   Hyperparameter.gd_mini_batch_epochs(
                                                                                       n_samples,
                                                                                       n_features,
                                                                                       5),
                                                                                   Hyperparameter.gd_mini_batch_size2(
                                                                                       n_features, user_defined_val=10),
                                                                                   Hyperparameter.gd_learning_rate,
                                                                                   model_name=Constants.MODEL_NAME_MBGD,
                                                                                   seed=seed)

        mbgd_list = mbgd_list + mbgd_l

        widrow_hoff_l = WidrowHoff.widrow_hoff_convergence2(X, y,
                                                            Hyperparameter.wf_learning_rate,
                                                            model_name=Constants.MODEL_NAME_LMS,
                                                            seed=seed)
        widrow_hoff_list = widrow_hoff_list + widrow_hoff_l

        orr_l = OnlineRidgeRegression.online_ridge_regression_convergence2(X, y,
                                                                           Hyperparameter.ridge_lasso_learning_rate,
                                                                           Hyperparameter.ridge_lasso_epochs(
                                                                               n_samples, 2),
                                                                           Hyperparameter.ridge_lasso_regularization_param,
                                                                           model_name=Constants.MODEL_NAME_ORR,
                                                                           seed=seed)
        orr_list = orr_list + orr_l

        olr_l = OnlineLassoRegression.online_lasso_regression_convergence2(X, y,
                                                                           Hyperparameter.ridge_lasso_learning_rate,
                                                                           Hyperparameter.ridge_lasso_epochs(
                                                                               n_samples, 2),
                                                                           Hyperparameter.ridge_lasso_regularization_param,
                                                                           model_name=Constants.MODEL_NAME_OLR,
                                                                           seed=seed)
        olr_list = olr_list + olr_l

        rls_l = RLS.rls_convergence2(X, y,
                                     Hyperparameter.rls_lambda_,
                                     Hyperparameter.rls_delta,
                                     model_name=Constants.MODEL_NAME_RLS,
                                     seed=seed)

        rls_list = rls_list + rls_l

        pa_l = OnlinePassiveAggressive.online_passive_aggressive_convergence2(X, y, Hyperparameter.pa_C,
                                                                              Hyperparameter.pa_epsilon,
                                                                              model_name=Constants.MODEL_NAME_PA,
                                                                              seed=seed)
        pa_list = pa_list + pa_l

    Util.print_average_of_maps(list_of_maps=olr_wa_list, model_name=Constants.MODEL_NAME_OLW_WA)
    Util.print_average_of_maps(list_of_maps=sgd_list, model_name=Constants.MODEL_NAME_SGD)
    Util.print_average_of_maps(list_of_maps=mbgd_list, model_name=Constants.MODEL_NAME_MBGD)
    Util.print_average_of_maps(list_of_maps=widrow_hoff_list, model_name=Constants.MODEL_NAME_LMS)
    Util.print_average_of_maps(list_of_maps=orr_list, model_name=Constants.MODEL_NAME_ORR)
    Util.print_average_of_maps(list_of_maps=olr_list, model_name=Constants.MODEL_NAME_OLR)
    Util.print_average_of_maps(list_of_maps=rls_list, model_name=Constants.MODEL_NAME_RLS)
    Util.print_average_of_maps(list_of_maps=pa_list, model_name=Constants.MODEL_NAME_PA)


if __name__ == '__main__':
    experiment13Main()


