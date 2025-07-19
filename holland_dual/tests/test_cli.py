from typer.testing import CliRunner

from holland_dual.shared.cli import app


def test_hdq_analyze():
    runner = CliRunner()
    result = runner.invoke(app, ["hdq-analyze", "--steps", "5"])
    assert result.exit_code == 0
    assert "spectral entropy" in result.stdout
