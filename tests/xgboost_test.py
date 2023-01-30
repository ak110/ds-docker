def test_run():
    import sklearn.datasets
    import sklearn.model_selection
    import xgboost

    data = sklearn.datasets.load_iris(as_frame=True)
    X, y = data.data, data.target  # pylint: disable=no-member
    X_train, X_test, y_train, _ = sklearn.model_selection.train_test_split(X, y)

    xgb = xgboost.XGBRegressor(
        n_estimators=3, tree_method="hist", enable_categorical=True
    )
    xgb.fit(X_train, y_train)
    assert xgb.predict(X_test).shape == (len(X_test),)
