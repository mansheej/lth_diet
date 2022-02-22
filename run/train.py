import dotenv
import sys

from lth_diet.exps import TrainExperiment as Experiment
from lth_diet.utils import utils

dotenv.load_dotenv()


def main() -> None:
    if len(sys.argv) == 1:
        sys.argv = [sys.argv[0], "--help"]

    exp = Experiment.create(cli_args=True)

    print("\n" + ("=" * 88))
    print("\nEXPERIMENT NAME:\n" + exp.name)
    print("\nREPLICATE: " + str(exp.replicate))
    print("\nEXPERIMENT HASH:\n" + utils.get_hash(exp.name))

    if exp.get_name:
        print("\n" + ("=" * 88) + "\n")
        sys.exit(0)
    print("\n" + ("-" * 88) + "\n")

    exp.validate()
    exp.run()

    print("\n" + ("=" * 88) + "\n")


if __name__ == "__main__":
    main()
