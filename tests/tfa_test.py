import numpy as np


def test_run():
    import tensorflow_addons as tfa

    img = np.ones((20, 30, 3), dtype=np.uint8)
    r = tfa.image.rotate(img, np.pi / 2).numpy()
    assert r[0, 0, 0] == 0  # zero padding
    assert r[10, 15, 0] == 1  # rotated


def test_run2():
    # tf.dataとの組み合わせが少し怪しい気がするので追加
    import tensorflow as tf
    import tensorflow_addons as tfa

    def f(img):
        r = tfa.image.rotate(img, np.pi / 2)
        r = tfa.image.translate(
            img,
            [1.5, 1.5],
            interpolation="bilinear",
            fill_mode="constant",
            fill_value=1,
        )
        return r

    imgs = np.ones((1, 20, 30, 3), dtype=np.uint8)
    ds = tf.data.Dataset.from_tensor_slices(imgs)
    ds = ds.map(f)
    r = next(iter(ds)).numpy()
    assert r[0, 0, 0] == 1  # constant padding
    assert r[10, 15, 0] == 1  # rotated
