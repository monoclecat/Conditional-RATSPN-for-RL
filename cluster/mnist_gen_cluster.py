from cw2 import experiment, cw_error, cluster_work
from cw2.cw_data import cw_logging


class MnistGen(experiment.AbstractExperiment):
    # ...

    def initialize(self, config: dict, rep: int, logger: cw_logging.LoggerArray) -> None:
        # Skip for Quickguide
        pass

    def run(self, config: dict, rep: int, logger: cw_logging.LoggerArray) -> None:
        # Perform your existing task
        print(1)
        # existing_project.project_main()

    def finalize(self, surrender: cw_error.ExperimentSurrender = None, crash: bool = False):
        # Skip for Quickguide
        pass


if __name__ == "__main__":
    # Give the MyExperiment Class, not MyExperiment() Object!!
    cw = cluster_work.ClusterWork(MnistGen)

    # Optional: Add loggers
    # cw.add_logger(...)

    # RUN!
    cw.run()