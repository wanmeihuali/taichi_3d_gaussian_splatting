import os
import time
import boto3
import sagemaker
from smexperiments.experiment import Experiment
from smexperiments.trial import Trial
from smexperiments.trial_component import TrialComponent
import argparse
import json
from github import Github

if __name__ == "__main__":
    branch_name = os.environ["BRANCH_NAME"]
    github_access_token = os.environ["GITHUB_ACCESS_TOKEN"]
    pull_request_number = os.environ["PULL_REQUEST_NUMBER"]
    git_owner = os.environ["GITHUB_OWNER"]
    git_repo = os.environ["GITHUB_REPO"]
    repo_folder_name = git_repo.split("/")[-1]
    git_sha = os.environ["GITHUB_SHA"]

    short_sha = git_sha[:7]

    experiment_name = f"{branch_name}-{short_sha}-{time.strftime('%Y-%m-%d-%H-%M-%S', time.gmtime())}"
    s3_output_dir = "s3://taichi-3d-gaussian-splatting-log"


    github_client = Github(github_access_token)
    print(f"Getting pull request {pull_request_number} from {git_repo}")
    github_repo = github_client.get_repo(f"{git_repo}")
    pull_request = github_repo.get_pull(int(pull_request_number))
    pull_request.create_issue_comment(f"Running experiment on sagemaker with git sha {git_sha}")
    
    datasets = {
        "tat-truck": "config/ci_sagemaker_tat_truck.json",
        "tat-train": "config/ci_sagemaker_tat_train.json",
    }

    sagemaker_client = boto3.client("sagemaker", region_name="us-east-2")
    experiment = Experiment.create(
        experiment_name=experiment_name,
        description=f"Experiment triggered by pull request https://github.com/{git_repo}/pull/{pull_request_number}",
        sagemaker_boto_client=sagemaker_client,
    )

    # we use prebuilt docker image for training, so we need to clone the repo and install the dependencies at entrypoint
    """
    entrypoint = [
        "git",
        "clone",
        f"https://github.com/{git_repo}.git",
        "&&",
        "cd",
        repo_folder_name,
        "&&",
        "pip3", 
        "install", 
        "-r", 
        "requirements.txt",
        "&&",
        "python3",
        "gaussian_point_train.py",
    ]
    """
    
    train_job_names = []
    train_job_name_to_output_path = {}
    for dataset, config_path in datasets.items():
        trial = Trial.create(
            trial_name=f"{experiment_name}-{dataset}",
            experiment_name=experiment_name,
            sagemaker_boto_client=sagemaker_client,
        )

        train_job_name = f"{experiment_name}-{dataset}"
        train_job_names.append(train_job_name)
        full_s3_output_path = os.path.join(s3_output_dir, dataset, train_job_name)
        train_job_name_to_output_path[train_job_name] = full_s3_output_path
        with open(config_path) as f:
            train_job_config = json.load(f)
        train_job_config["TrainingJobName"] = train_job_name
        entrypoint = f"""git clone https://github.com/{git_repo}.git && \
            cd {repo_folder_name} && \
            pip3 install -r requirements.txt && \
            python3 gaussian_point_train.py --train_config {train_job_config["HyperParameters"]["train_config"]}
        """
        train_job_config["AlgorithmSpecification"]["ContainerEntrypoint"] = entrypoint
        train_job_config["OutputDataConfig"]["S3OutputPath"] = full_s3_output_path
        sagemaker_client.create_training_job(**train_job_config)
        pull_request.create_issue_comment(f"Training job {train_job_name} created")

        trial_component_name = f"{train_job_name}-aws-training-job"
        trial_component = TrialComponent.load(trial_component_name=trial_component_name)
        trial.add_trial_component(trial_component)

    # wait for training jobs to finish
    while True:
        all_jobs_completed = True
        for train_job_name in train_job_names:
            train_job_description = sagemaker_client.describe_training_job(TrainingJobName=train_job_name)
            train_job_status = train_job_description["TrainingJobStatus"]
            if train_job_status in ["Failed", "Stopped"]:
                pull_request.create_issue_comment(f"Training job {train_job_name} failed")
            elif train_job_status == "Completed":
                pull_request.create_issue_comment(f"Training job {train_job_name} completed, output path {train_job_name_to_output_path[train_job_name]}")
            else:
                all_jobs_completed = False
        if all_jobs_completed:
            break
        print("Waiting for training jobs to finish")
        time.sleep(60)


        