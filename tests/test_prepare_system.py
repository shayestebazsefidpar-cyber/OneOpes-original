"""`oneopes prepare-system` with `echo` standing in for gmx: which gmx commands run, in which order."""
import logging
import shlex

import pytest

from oneopes.cli import main
from oneopes.ligand.setup import MDP_DIR

ALL_STEPS = ["editconf", "solvate", "grompp", "genion", "grompp", "mdrun", "grompp", "mdrun", "grompp", "mdrun"]


def prepare_system(tmp_path, gmx: str, *options: str) -> None:
    gro, top = tmp_path / "ligand.gro", tmp_path / "ligand.top"
    gro.touch()
    top.write_text("; topology\n")
    main(["prepare-system", "--ligand-gro", str(gro), "--ligand-top", str(top), "--outdir", str(tmp_path / "out"),
          "--gmx", gmx, *options])


def gmx_commands(tmp_path, caplog, *options: str) -> list[list[str]]:
    """Run prepare-system with `--gmx echo`; return each gmx command it logged, without the gmx executable."""
    caplog.set_level(logging.INFO, logger="oneopes")
    prepare_system(tmp_path, "echo", *options)
    shown = [message.removeprefix("$ ") for message in caplog.messages if message.startswith("$ ")]
    return [shlex.split(command.split(" | ")[-1])[1:] for command in shown]


def test_runs_every_step_in_order_and_copies_the_topology_and_production_settings(tmp_path, caplog):
    assert [command[0] for command in gmx_commands(tmp_path, caplog)] == ALL_STEPS
    assert (tmp_path / "out" / "system.top").read_text() == "; topology\n"
    assert (tmp_path / "out" / "prod.mdp").read_text() == (MDP_DIR / "prod.mdp").read_text()


@pytest.mark.parametrize("start_from, expected", [("npt_tpr", ["grompp", "mdrun"]), ("genion", ALL_STEPS[3:])])
def test_resumes_from_a_step(tmp_path, caplog, start_from, expected):
    assert [command[0] for command in gmx_commands(tmp_path, caplog, "--start-from", start_from)] == expected


def test_only_ions_tpr_allows_a_warning_unless_asked(tmp_path, caplog):
    allowed = {command[command.index("-o") + 1]: command[command.index("-maxwarn") + 1]
               for command in gmx_commands(tmp_path, caplog) if command[0] == "grompp"}
    assert allowed == {"ions.tpr": "1", "em.tpr": "0", "nvt.tpr": "0", "npt.tpr": "0"}
    caplog.clear()
    npt = gmx_commands(tmp_path, caplog, "--maxwarn", "2", "--start-from", "npt_tpr")[0]
    assert npt[npt.index("-maxwarn") + 1] == "2"


def test_ion_concentration_is_passed_to_genion(tmp_path, caplog):
    genion = gmx_commands(tmp_path, caplog, "--ion-conc", "0.15", "--start-from", "genion")[0]
    assert genion[0] == "genion" and genion[-2:] == ["-conc", "0.15"]


def errors(caplog) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.levelno == logging.ERROR]


def test_a_failed_step_is_one_line_naming_where_to_resume(tmp_path, caplog):
    with pytest.raises(SystemExit):
        prepare_system(tmp_path, "false", "--start-from", "em_tpr")
    assert errors(caplog) == ["error: false grompp failed at step em_tpr (exit 1; see its output above). "
                              "Fix it and rerun with --start-from em_tpr."]


def test_a_missing_gmx_is_one_line(tmp_path, caplog):
    with pytest.raises(SystemExit):
        prepare_system(tmp_path, str(tmp_path / "no_gmx"))
    assert len(errors(caplog)) == 1 and errors(caplog)[0].endswith("not found; pass the gmx executable with --gmx.")
