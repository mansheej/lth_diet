import sys
import warnings
from typing import Type

from lth_diet.exps import TrainExperiment as Experiment


def warning_on_one_line(
    message: str,
    category: Type[Warning],
    filename: str,
    lineno: int,
    file=None,
    line=None,
):
    # From https://stackoverflow.com/questions/26430861/make-pythons-warnings-warn-not-mention-itself
    return f"{category.__name__}: {message} (source: {filename}:{lineno})\n"


def main() -> None:
    warnings.formatwarning = warning_on_one_line
    if len(sys.argv) == 1:
        sys.argv = [sys.argv[0], "--help"]

    exp = Experiment.create(cli_args=True)
    exp.run()


if __name__ == "__main__":
    main()
