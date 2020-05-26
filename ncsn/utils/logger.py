from subprocess import Popen, PIPE

from pytorch_lightning.loggers import MLFlowLogger
from mlflow.utils.mlflow_tags import *


def _terminal_output(args: [str]) -> str:
    stdout, stderr = Popen(args, stdout=PIPE).communicate()
    return stdout.decode("utf-8").strip()


__user__ = _terminal_output(["git", "config", "user.name"])
__remote__ = _terminal_output(["git", "remote"])
__remote_url__ = _terminal_output(["git", "remote", "get-url", __remote__])
__branch__ = _terminal_output(["git", "branch", "--show-current"])
__commit__ = _terminal_output(["git", "log", "-n", "1", "--oneline"]).split(" ")[0]


def custom_mlflow_logger(experiment_name: str = "", run_name: str = "", debug: bool = False):
    if debug:
        return None

    return MLFlowLogger(
        experiment_name=experiment_name,
        tags={
            MLFLOW_USER: __user__,
            MLFLOW_RUN_NAME: run_name,
            MLFLOW_GIT_REPO_URL: __remote_url__,
            MLFLOW_SOURCE_NAME: __remote_url__,
            MLFLOW_GIT_BRANCH: __branch__,
            MLFLOW_GIT_COMMIT: __commit__,
        }
    )

