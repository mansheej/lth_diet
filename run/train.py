import dotenv
import sys
from lth_diet.exps import TrainExperiment as Experiment
from lth_diet.utils import utils

dotenv.load_dotenv()


def main() -> None:
    if len(sys.argv) == 1:
        sys.argv = [sys.argv[0], "--help"]

    exp = Experiment.create(cli_args=True)

    if "--name" in sys.argv:
        print("Name:", exp.name, "Hash:", utils.get_hash(exp.name), sep="\n")
        sys.exit(0)

    exp.validate()
    exp.run()


if __name__ == "__main__":
    main()
