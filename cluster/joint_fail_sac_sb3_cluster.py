from cw2 import experiment, cw_error, cluster_work
from cw2.cw_data import cw_logging
from experiments.joint_fail_sac_sb3 import joint_failure_sac


class JointFailSacCluster(experiment.AbstractExperiment):
    # ...

    def initialize(self, config: dict, rep: int, logger: cw_logging.LoggerArray) -> None:
        # Skip for Quickguide
        pass

    def run(self, config: dict, rep: int, logger: cw_logging.LoggerArray) -> None:
        # Perform your existing task
        joint_failure_sac(**config.get('params'))
        print(1)

    def finalize(self, surrender: cw_error.ExperimentSurrender = None, crash: bool = False):
        # Skip for Quickguide
        pass


if __name__ == "__main__":
    # Give the MyExperiment Class, not MyExperiment() Object!!
    cw = cluster_work.ClusterWork(JointFailSacCluster)

    # Optional: Add loggers
    # cw.add_logger(...)

    # RUN!
    cw.run()
