"""How `oneopes` reports bad options and inputs: a one-line error and exit code 1, before any work."""
import logging

import MDAnalysis as mda
import pytest
from cyclopts import CycloptsError, ValidationError

from ligand_systems import ligand_in_water
from oneopes.cli import app, main
from oneopes.errors import InputError


def test_a_tolerance_that_means_nothing_stops_the_run_with_one_line(tmp_path, caplog):
    """The InputError comes from building ConvergenceCriteria while the command line is read."""
    top, traj = ligand_in_water(tmp_path)
    with pytest.raises(SystemExit) as exit_info:
        main(["monitor", "--tpr", top, "--xtc", traj, "--outdir", str(tmp_path / "out"), "--n-passes", "0"])
    assert exit_info.value.code == 1
    errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    assert errors == ["error: n_passes must be at least 1, got 0."]
    assert not (tmp_path / "out").exists()


def test_an_input_problem_found_later_is_reported_the_same_way(tmp_path, caplog):
    """A ligand selection that matches nothing is only found once the topology is read."""
    top, traj = ligand_in_water(tmp_path)
    with pytest.raises(SystemExit) as exit_info:
        main(["select", "--tpr", top, "--xtc", traj, "--outdir", str(tmp_path / "out"),
              "--ligand-selection", "resname NONE"])
    assert exit_info.value.code == 1
    assert any("is empty" in r.getMessage() for r in caplog.records if r.levelno == logging.ERROR)


def test_a_trajectory_shorter_than_one_block_is_a_one_line_error(tmp_path, caplog):
    """3 frames of 1 ps, far less than one 5 ns block, and no mdrun to wait for."""
    top, traj = ligand_in_water(tmp_path)
    with pytest.raises(SystemExit) as exit_info:
        main(["monitor", "--tpr", top, "--xtc", traj, "--outdir", str(tmp_path / "out")])
    assert exit_info.value.code == 1
    errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    assert errors == ["error: mdrun ended before the first block was complete; nothing to report."]


def test_missing_mdp_files_stop_prepare_system_before_gromacs_runs(tmp_path, caplog):
    top, _ = ligand_in_water(tmp_path)
    (tmp_path / "mdp").mkdir()
    with pytest.raises(SystemExit) as exit_info:
        main(["prepare-system", "--ligand-gro", top, "--ligand-top", top, "--outdir", str(tmp_path / "build"),
              "--mdp-dir", str(tmp_path / "mdp"), "--gmx", "echo"])
    assert exit_info.value.code == 1
    errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1 and errors[0].startswith("error: Required mdp file not found:")
    assert not (tmp_path / "build").exists()


@pytest.mark.parametrize("command", ["monitor", "select"])
def test_a_trajectory_of_another_system_is_a_one_line_error(tmp_path, caplog, command):
    top, _ = ligand_in_water(tmp_path)
    other = mda.Universe.empty(10, trajectory=True)
    with mda.Writer(str(tmp_path / "other.xtc"), 10) as writer:
        writer.write(other.atoms)
    with pytest.raises(SystemExit) as exit_info:
        main([command, "--tpr", top, "--xtc", str(tmp_path / "other.xtc"), "--outdir", str(tmp_path / "out")])
    assert exit_info.value.code == 1
    errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1 and errors[0].startswith("error: cannot read top.pdb, other.xtc:")


@pytest.mark.parametrize("option, value", [("--poll-seconds", "-1"), ("--fp-round-decimals", "-1")])
def test_monitor_parameters_are_checked_when_the_command_line_is_read(tmp_path, option, value):
    top, traj = ligand_in_water(tmp_path)
    with pytest.raises(ValidationError):
        app.parse_args(["monitor", "--tpr", top, "--xtc", traj, option, value],
                       exit_on_error=False, print_error=False)


@pytest.mark.parametrize("option, value", [("--block-ns", "0"), ("--norm-tail-bins", "0")])
def test_monitor_options_are_checked_when_the_command_line_is_read(tmp_path, option, value):
    top, traj = ligand_in_water(tmp_path)
    with pytest.raises(InputError):
        app.parse_args(["monitor", "--tpr", top, "--xtc", traj, option, value],
                       exit_on_error=False, print_error=False)


@pytest.mark.parametrize("option, value", [("--box-distance-nm", "0"), ("--ion-conc", "-0.1"),
                                           ("--box-type", "sphere"), ("--maxwarn", "-1")])
def test_prepare_system_options_are_checked_when_the_command_line_is_read(tmp_path, option, value):
    top, _ = ligand_in_water(tmp_path)
    with pytest.raises(CycloptsError):
        app.parse_args(["prepare-system", "--ligand-gro", top, "--ligand-top", top, "--outdir", str(tmp_path),
                        option, value], exit_on_error=False, print_error=False)
