import os
import subprocess
import time
import argparse
import shutil
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def should_update_local(local_commit: str, remote_commit: str) -> bool:
    return local_commit != remote_commit


def backup_config():
    """Backup the config file if it exists"""
    try:
        if os.path.exists("config/config.yml"):
            shutil.copy2("config/config.yml", "config/config.yml.backup")
            return True
    except Exception as e:
        logging.error(f"Error backing up config: {e}")
        return False


def restore_config():
    """Restore the config file from backup"""
    try:
        if os.path.exists("config/config.yml.backup"):
            shutil.copy2("config/config.yml.backup", "config/config.yml")
            os.remove("config/config.yml.backup")
            return True
    except Exception as e:
        logging.error(f"Error restoring config: {e}")
        return False


def update_local_repo(remote_commit: str):
    backup_config()

    reset_cmd = f"git reset --hard {remote_commit}"
    process = subprocess.Popen(reset_cmd.split(), stdout=subprocess.PIPE)
    _, error = process.communicate()

    if error:
        logging.error(f"Error in updating: {error}")
        return

    restore_config()

    os.system("./utils/autoupdate_validator_steps.sh")
    time.sleep(20)

    logging.info("Finished running the autoupdate steps! Ready to go 😎")


def run_auto_updater():
    logging.info("Running the autoupdater! First I'll start 'er up...")

    os.system("./utils/launch_auditor.sh")
    time.sleep(60)
    logging.info("Auditor container launched successfully!")

    while True:
        logging.info("Checking github for updates to the auditor code...")
        current_branch = subprocess.getoutput("git rev-parse --abbrev-ref HEAD")
        local_commit = subprocess.getoutput("git rev-parse HEAD")
        os.system("git fetch")
        remote_commit = subprocess.getoutput(f"git rev-parse origin/{current_branch}")

        if should_update_local(local_commit, remote_commit):
            logging.info(
                "Local repo is not up-to-date with github, there's been an update! Updating now..."
            )
            update_local_repo(remote_commit)

        else:
            logging.info("Repo is up-to-date.")

        time.sleep(60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run auto updates for a validator")

    args = parser.parse_args()

    run_auto_updater()
