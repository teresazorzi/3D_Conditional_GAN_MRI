import tempfile
import os
import pytest
from mrisyngan.train import train_and_evaluate


def test_train_and_evaluate_raises_on_missing_data():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create empty directory without class subfolders / NIfTI files
        with pytest.raises(Exception):
            train_and_evaluate({}, tmpdir, device='cpu')
