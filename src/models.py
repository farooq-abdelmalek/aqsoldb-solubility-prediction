"""
Regression model definitions for the AqSolDB solubility project.

Provides five comparable baseline algorithms. Each model is wrapped
in a scikit-learn Pipeline so preprocessing is fitted only on
training data.

Model selection and evaluation are handled separately.
"""

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


def build_models(random_state: int = 42) -> dict[str, Pipeline]:
    """
    Create five untrained regression pipelines.

    Parameters
    ----------
    random_state : int
        Random seed used by stochastic estimators.

    Returns
    -------
    dict[str, Pipeline]
        Model names mapped to their scikit-learn pipelines.

    Notes
    -----
    These are initial benchmark configurations, not tuned models.

    Missing values are imputed using training-set medians.
    Ridge and SVR additionally use training-set standardization.
    Tree-based models do not require feature standardization.
    """

    estimators = {
        "Ridge Regression": Ridge(alpha=1.0),

        "SVR": SVR(
            kernel="rbf",
            C=10,
            epsilon=0.1,
            gamma="scale"
        ),

        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            max_features="sqrt",
            min_samples_leaf=2,
            n_jobs=2,
            random_state=random_state
        ),

        "XGBoost": XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=2,
            random_state=random_state,
            verbosity=0
        ),

        "LightGBM": LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            n_jobs=2,
            random_state=random_state,
            verbosity=-1
        )
    }

    pipelines = {}

    for name, estimator in estimators.items():

        # Median statistics are fitted only during model.fit().
        steps = [
            ("imputer", SimpleImputer(strategy="median"))
        ]

        # Distance-based and regularized models benefit from scaling.
        if name in ("Ridge Regression", "SVR"):
            steps.append(("scaler", StandardScaler()))

        steps.append(("model", estimator))

        pipelines[name] = Pipeline(steps)

    return pipelines
