import json
from pathlib import Path

import numpy as np
import pytest
from embcompare.export_utils import NumpyArrayEncoder


def test_json_numpy_array_encoder(tmp_path: Path):
    # Simple array serialization
    obj = np.array([0.1, 0.2, 0.3], dtype=np.float32)

    with open(tmp_path / "array.json", "w") as f:
        with pytest.raises(TypeError):
            json.dump(obj, f)

    with open(tmp_path / "array.json", "w") as f:
        json.dump(obj, f, cls=NumpyArrayEncoder)

    # Arbitrary nested dict with numpy ndarray should be serializable
    obj = {
        "arrays": [
            np.array([0.1, 0.2, 0.3], dtype=np.float32),
            np.array([0.1, 0.2, 0.3], dtype=np.float64),
            np.array([1, 2, 3], dtype=np.int8),
            np.array([1, 2, 3], dtype=np.int16),
            np.array(["a", "b", "c"]),
        ],
        "set": {1, 2, "3"},
        "normal": {"json": ["ok"]},
    }

    with open(tmp_path / "array.json", "w") as f:
        with pytest.raises(TypeError):
            json.dump(obj, f)

    with open(tmp_path / "array.json", "w") as f:
        json.dump(obj, f, cls=NumpyArrayEncoder)

    # Some object are still not serializable. We test an example with
    # NumpyArrayEncoder to get 100% test coverage
    obj = {"not serializable": map(lambda x: x, "abc")}

    with open(tmp_path / "array.json", "w") as f:
        with pytest.raises(TypeError):
            json.dump(obj, f, cls=NumpyArrayEncoder)
