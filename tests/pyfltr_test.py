def test_run():
    import subprocess

    subprocess.run(
        ["pyfltr", "--commands=pyupgrade,isort,black,pflake8", __file__], check=True
    )
