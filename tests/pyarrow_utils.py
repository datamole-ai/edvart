import pytest

from edvart.data_types import PYARROW_PANDAS_BACKEND_AVAILABLE

if PYARROW_PANDAS_BACKEND_AVAILABLE:
    pyarrow_parameterize = pytest.mark.parametrize("pyarrow_dtypes", [False, True])
else:
    pyarrow_parameterize = pytest.mark.parametrize("pyarrow_dtypes", [False])
