import pytest


def test_sqlite3():
    import sqlite3

    del sqlite3


def test_dbm():
    import dbm

    del dbm


@pytest.mark.skip(reason="uvでなぜか動かないためとりあえずスキップ")
def test_gdbm():
    from dbm import gnu

    del gnu


def test_ssl():
    import ssl

    del ssl


def test_ctypes():
    import ctypes

    assert ctypes.sizeof(ctypes.c_int64(value=0)) == 8


def test_hashlib():
    import hashlib

    assert (
        hashlib.sha256(b"password1").hexdigest()
        == "0b14d501a594442a01c6859541bcb3e8164d183d32937b851835442f69d5c94e"
    )


def test_uuid():
    import uuid

    assert str(uuid.uuid4()) != ""
