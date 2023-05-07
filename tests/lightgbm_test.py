import pytest


@pytest.mark.skip(reason="Fatal Python error: Segmentation fault")
def test_run():
    import lightgbm as lgb
    import sklearn.datasets
    import sklearn.model_selection

    data = sklearn.datasets.load_iris(as_frame=True)
    X, y = data.data, data.target  # pylint: disable=no-member
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y)

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    params = {
        "boosting_type": "gbdt",
        "objective": "regression",
        "metric": {"l2", "l1"},
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": 0,
    }
    gbm = lgb.train(
        params,
        lgb_train,
        num_boost_round=2,
        valid_sets=lgb_eval,
        callbacks=[lgb.early_stopping(stopping_rounds=1)],
    )
    assert gbm.best_iteration == 1
