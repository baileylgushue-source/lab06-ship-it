"""Check that the files a Hugging Face Space reads agree with each other."""

import json
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def readme_settings() -> dict[str, str]:
    """Return the settings block at the top of README.md as a dictionary."""
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    block = text.split("---")[1]
    settings = {}
    for line in block.strip().splitlines():
        key, _, value = line.partition(":")
        settings[key.strip()] = value.strip().strip('"')
    return settings


def requirements_lines() -> list[str]:
    """Return the non-empty lines of requirements.txt."""
    text = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if line.strip()]


def test_installed_sklearn_matches_model_info():
    info_path = ROOT / "models" / "model_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    assert version("scikit-learn") == info["scikit_learn"]


def test_installed_sklearn_matches_requirements_pin():
    assert f"scikit-learn=={version('scikit-learn')}" in requirements_lines()


def test_gradio_is_left_out_of_the_space_requirements():
    assert not any(line.startswith("gradio") for line in requirements_lines())


def test_readme_sdk_version_matches_installed_gradio():
    assert readme_settings()["sdk_version"] == version("gradio")


def test_readme_app_file_exists():
    assert (ROOT / readme_settings()["app_file"]).is_file()
