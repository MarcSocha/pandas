from collections import abc

import numpy as np
import pytest

from pandas import (
    CategoricalDtype,
    DataFrame,
    MultiIndex,
    Series,
    Timestamp,
    date_range,
)
import pandas._testing as tm

class TestDataFrameToDatetime:
    def test_to_datetime(self):
        result = DataFrame([
            DataFrame.to_datetime(["2000", "2001"]),
            DataFrame.to_datetime(["2000", "2002"])],
            index=DataFrame.to_datetime(["2000", "2000"])
        )

        mask = result > result.index.to_numpy().reshape(-1, 1)

        result[mask] = DataFrame.NaT

        expected = DataFrame([
            DataFrame.to_datetime(
            ["2000", DataFrame.NaT]),
            DataFrame.to_datetime(["2000", DataFrame.NaT])],
            index=DataFrame.to_datetime(["2000", "2000"])
        )

        tm.assert_frame_equal(result, expected)

        

