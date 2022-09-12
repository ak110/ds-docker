def test_run():
    import sklearn.datasets
    import xgboost

    data = sklearn.datasets.fetch_openml(name="house_prices", as_frame=True)
    X, y = data.data, data.target  # pylint: disable=no-member

    for c in X.select_dtypes(include=object).columns:
        X[c] = X[c].astype("category")

    xgb = xgboost.XGBRegressor(
        n_estimators=3, tree_method="hist", enable_categorical=True
    )
    xgb.fit(X[:100], y[:100])
    assert xgb.predict(X[100:]).shape == (len(X[100:]),)
