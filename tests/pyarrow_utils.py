from edvart.data_types import PYARROW_PANDAS_BACKEND_AVAILABLE

if PYARROW_PANDAS_BACKEND_AVAILABLE:
    pyarrow_params = [True, False]
else:
    pyarrow_params = [False]
