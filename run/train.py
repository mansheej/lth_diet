import sys
from lth_diet.exps import TrainExperiment as Experiment


def main() -> None:
    if len(sys.argv) == 1:
        sys.argv = [sys.argv[0], "--help"]

    exp = Experiment.create(cli_args=True)
    exp.validate()
    exp.run()


if __name__ == "__main__":
    main()
