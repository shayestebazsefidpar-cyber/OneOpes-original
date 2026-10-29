"""`oneopes prepare-system` with `echo` standing in for gmx: which gmx commands run, in which order."""
import pytest

from oneopes.cli import main

ALL_STEPS = ["editconf", "solvate", "grompp", "genion", "grompp", "mdrun", "grompp", "mdrun", "grompp", "mdrun"]


def gmx_commands(tmp_path, capsys, *options: str) -> list[list[str]]:
    """Run prepare-system with `--gmx echo`; return each gmx command it ran, without the gmx executable."""
    gro, top = tmp_path / "ligand.gro", tmp_path / "ligand.top"
    gro.touch()
    top.write_text("; topology\n")
    main(["prepare-system", "--ligand-gro", str(gro), "--ligand-top", str(top), "--outdir", str(tmp_path / "out"),
          "--gmx", "echo", *options])
    shown = [line.split(" $ ", 1)[1] for line in capsys.readouterr().out.splitlines() if " $ " in line]
    return [command.split(" | ")[-1].split()[1:] for command in shown]


def test_runs_every_step_in_order_and_copies_the_topology(tmp_path, capsys):
    assert [command[0] for command in gmx_commands(tmp_path, capsys)] == ALL_STEPS
    assert (tmp_path / "out" / "system.top").read_text() == "; topology\n"


@pytest.mark.parametrize("start_from, expected", [("npt_tpr", ["grompp", "mdrun"]), ("genion", ALL_STEPS[3:])])
def test_resumes_from_a_step(tmp_path, capsys, start_from, expected):
    assert [command[0] for command in gmx_commands(tmp_path, capsys, "--start-from", start_from)] == expected


def test_ion_concentration_is_passed_to_genion(tmp_path, capsys):
    genion = gmx_commands(tmp_path, capsys, "--ion-conc", "0.15", "--start-from", "genion")[0]
    assert genion[0] == "genion" and genion[-2:] == ["-conc", "0.15"]
