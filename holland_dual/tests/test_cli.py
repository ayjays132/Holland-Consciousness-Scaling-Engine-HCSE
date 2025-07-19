from typer.testing import CliRunner
import numpy as np
import warnings
from numpy.exceptions import ComplexWarning

from holland_dual.shared.cli import app


def test_hdq_analyze():
    runner = CliRunner()
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=ComplexWarning)
        result = runner.invoke(app, ["hdq-analyze", "--steps", "5"])
    assert result.exit_code == 0
    assert "spectral entropy" in result.stdout
