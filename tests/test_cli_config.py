"""`oneopes --config file.yaml <command>`: settings from a file, overridden by the command line."""
from pathlib import Path

import pytest
import yaml

from oneopes.cli import app, main

EXAMPLE = Path(__file__).parents[1] / "configs" / "example.yaml"
SELECTION = {"G1": [{"name": "X1", "serial": 3}, {"name": "X2", "serial": 7}],
             "G2": [{"name": "X3", "serial": 11}, {"name": "X4", "serial": 2}]}


def write_config(tmp_path, build_cv: dict) -> Path:
    """A g1_g2.yaml to build from, and a config whose build-cv section is ``build_cv``."""
    (tmp_path / "g1_g2.yaml").write_text(yaml.safe_dump(SELECTION))
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"build-cv": {"g1-g2": str(tmp_path / "g1_g2.yaml"), **build_cv}}))
    return config


def test_settings_come_from_the_config(tmp_path):
    config = write_config(tmp_path, {"out": str(tmp_path / "from_config.dat")})
    main(["--config", str(config), "build-cv"])
    assert "GROUP ATOMS=3,7" in (tmp_path / "from_config.dat").read_text()


def test_the_command_line_overrides_the_config(tmp_path):
    config = write_config(tmp_path, {"out": str(tmp_path / "from_config.dat")})
    main(["--config", str(config), "build-cv", "--out", str(tmp_path / "from_flag.dat")])
    assert (tmp_path / "from_flag.dat").exists() and not (tmp_path / "from_config.dat").exists()


def test_a_misspelt_setting_is_an_error(tmp_path):
    config = write_config(tmp_path, {"out": str(tmp_path / "out.dat"), "water-group": "GROUPA=1-9"})
    with pytest.raises(SystemExit):
        main(["--config", str(config), "build-cv"])
    assert not (tmp_path / "out.dat").exists()


def test_the_example_config_has_one_section_per_command():
    assert set(yaml.safe_load(EXAMPLE.read_text())) <= set(app)


@pytest.mark.parametrize("command, settings", yaml.safe_load(EXAMPLE.read_text()).items())
def test_every_example_setting_is_an_option_of_its_command(command, settings):
    options = {name.removeprefix("--") for argument in app[command].assemble_argument_collection()
               for name in argument.names}
    assert set(settings) <= options
