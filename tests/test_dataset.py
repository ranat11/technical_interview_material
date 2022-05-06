import pytest
import os
import pandas as pd

DATA = "data"


@pytest.fixture
def data_list():
    return [os.path.join(DATA, file_name) for file_name in os.listdir(DATA)]


def file_error(column_bool, data_list):
    return ', '.join([data_list[idx] for idx, i in enumerate(column_bool) if not i])


def test_column_name(data_list):
    column_bool = []
    for file in data_list:
        df = pd.read_csv(file, index_col='ID')
        column_bool.append(list(df.columns) == [
            "height", "weight", "age", "BicepC"])

    assert all(
        column_bool), f"{file_error(column_bool, data_list)} column name not as expected"


def test_nan(data_list):
    column_bool = []
    for file in data_list:
        df = pd.read_csv(file, index_col='ID')
        column_bool.append(not df.isnull().values.any())

    assert all(
        column_bool), f"{file_error(column_bool, data_list)} value not as expected"
