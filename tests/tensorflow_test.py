import os

import numpy as np
import pytest


def test_run(tmpdir):
    import tensorflow as tf

    X_train = np.random.rand(10, 28, 28, 3)
    y_train = np.random.rand(10)

    inputs = x = tf.keras.layers.Input((28, 28, 3))
    x = tf.keras.layers.Conv2D(1, 3, padding="same")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=x)
    model.compile("adam", "mse", jit_compile=False)

    model.summary()
    tf.keras.utils.plot_model(model, str(tmpdir / "model.png"))

    model.fit(X_train, y_train, batch_size=10, epochs=2)
    model.save(str(tmpdir / "model.keras"))

    model = tf.keras.models.load_model(str(tmpdir / "model.keras"))
    assert model.predict(X_train).shape == (len(X_train), 1)


def test_load_data():
    """<https://github.com/keras-team/keras/issues/12729>"""
    import tensorflow as tf

    tf.keras.datasets.imdb.load_data()


def test_gpu():
    """GPUのテスト。環境変数GPUに従う。"""
    import tensorflow as tf

    gpu = os.environ.get("GPU")
    if gpu is None:
        pytest.skip("Environment variable 'GPU' is not defined.")
    elif gpu == "none":
        assert len(tf.config.list_physical_devices("GPU")) == 0
    else:
        assert len(tf.config.list_physical_devices("GPU")) > 0
