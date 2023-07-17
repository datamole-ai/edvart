from edvart.example_datasets import (
    dataset_auto,
    dataset_global_temp,
    dataset_meteorite_landings,
    dataset_pollution,
    dataset_titanic,
)


def test_dataset_auto():
    df = dataset_auto()
    assert df.shape == (398, 9)
    assert df.isnull().sum().sum() == 0


def test_dataset_titanic():
    df = dataset_titanic()
    assert df.shape == (891, 12)


def test_dataset_meteorite_landings():
    df = dataset_meteorite_landings()
    assert df.shape == (45716, 10)


def test_dataset_global_temp():
    df = dataset_global_temp()
    assert df.index.name == "Date"
    assert df.shape == (1644, 2)


def test_dataset_pollution():
    df = dataset_pollution()
    assert df.index.name == "date"
    assert df.shape == (43800, 8)
