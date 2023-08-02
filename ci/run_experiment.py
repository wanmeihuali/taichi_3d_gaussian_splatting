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
from collections import defaultdict

JOB_URL_FORMAT = "https://us-east-2.console.aws.amazon.com/sagemaker/home?region=us-east-2#/jobs/{}"


def kv_pairs_to_markdown(kv_pairs):
    comment = "|"
    for key, value in kv_pairs.items():
        comment += f" {key} |"
    comment += "\n|"
    for key, value in kv_pairs.items():
        comment += " --- |"
    comment += "\n|"
    for key, value in kv_pairs.items():
        comment += f" {value} |"
    comment += "\n"
    return comment
        
def comment_all_metrics(train_job_name, train_job_metrics):
    comment = ""
    concern_latest_metric_names = {
        "train:iteration", 
        "train:loss", 
        "train:l1loss", 
        "train:ssimloss",
        "train:psnr",
        "train:ssim",
        "val:loss",
        "val:psnr",
        "val:ssim",
        "train:num_valid_points"
    }
    concern_max_metric_names = {
        "train:psnr",
        "train:ssim",
        "val:psnr",
        "val:ssim",
        "train:7kpsnr",
        "train:7kssim",
        "val:7kpsnr",
        "val:7kssim",
        "train:5kpsnr",
        "train:5kssim",
        "val:5kpsnr",
        "val:5kssim",
    }
    concern_iterations = {5000, 7000, 30000}
    concern_iteration_metric_names = {
        "train:loss"
        "train:l1loss", 
        "train:ssimloss",
        "train:psnr",
        "train:ssim",
        "val:psnr",
        "val:ssim",
        "train:num_valid_points"
    }
    kv_pairs = {}
    # for concern_latest_metric_name in concern_latest_metric_names:
    for concern_latest_metric_name in sorted(list(concern_latest_metric_names)):
        if concern_latest_metric_name not in train_job_metrics[train_job_name] \
            or len(train_job_metrics[train_job_name][concern_latest_metric_name]) == 0:
            continue
        latest_ts = max(train_job_metrics[train_job_name][concern_latest_metric_name].keys())
        kv_pairs[concern_latest_metric_name] = train_job_metrics[train_job_name][concern_latest_metric_name][latest_ts]
    # comment is markdown table
    if len(kv_pairs) > 0:
        comment += "## Latest Metrics\n"
        comment += kv_pairs_to_markdown(kv_pairs)
        comment += "\n"
    kv_pairs = {}
    for concern_max_metric_name in sorted(list(concern_max_metric_names)):
        if concern_max_metric_name not in train_job_metrics[train_job_name] \
            or len(train_job_metrics[train_job_name][concern_max_metric_name]) == 0:
            continue
        max_value = max(train_job_metrics[train_job_name][concern_max_metric_name].values())
        kv_pairs[concern_max_metric_name] = max_value
    if len(kv_pairs) > 0:
        comment += "## Max Metrics\n"
        comment += kv_pairs_to_markdown(kv_pairs) 
        comment += "\n"

    for concern_iteration in concern_iterations:
        if concern_iteration not in train_job_metrics[train_job_name]["train:iteration"]:
            continue
        iteration_ts = train_job_metrics[train_job_name]["train:iteration"][concern_iteration]
        kv_pairs = {}
        for concern_iteration_metric_name in concern_iteration_metric_names:
            nearest_ts = min(train_job_metrics[train_job_name][concern_iteration_metric_name].keys(), key=lambda x:abs(x-iteration_ts))
            if abs(nearest_ts - iteration_ts) > 60:
                continue
            kv_pairs[concern_iteration_metric_name] = train_job_metrics[train_job_name][concern_iteration_metric_name][nearest_ts]
        if len(kv_pairs) > 0:
            comment += f"## Iteration {concern_iteration}\n"
            comment += kv_pairs_to_markdown(kv_pairs)
            comment += "\n"
    return comment
    

if __name__ == "__main__":
    branch_name = os.environ["BRANCH_NAME"]
    github_access_token = os.environ["GITHUB_ACCESS_TOKEN"]
    pull_request_number = os.environ["PULL_REQUEST_NUMBER"]
    git_owner = os.environ["GITHUB_OWNER"]
    git_repo = os.environ["GITHUB_REPO"]
    repo_folder_name = git_repo.split("/")[-1]
    git_sha = os.environ["GITHUB_SHA"]
    image_uri = os.environ["IMAGE_URI"]
    label_name = os.environ["LABEL_NAME"]

    short_sha = git_sha[:7]

    branch_name = branch_name.replace("/", "-").replace("_", "-")
    experiment_name = f"{branch_name}-{short_sha}-{time.strftime('%y%m%d-%H%M%S', time.gmtime())}"
    # limit length of experiment name to 63 characters by truncating the branch name
    if len(experiment_name) > 63:
        experiment_name = experiment_name[63 - len(experiment_name):]
    s3_output_dir = "s3://taichi-3d-gaussian-splatting-log"


    github_client = Github(github_access_token)
    print(f"Getting pull request {pull_request_number} from {git_repo}")
    github_repo = github_client.get_repo(f"{git_repo}")
    pull_request = github_repo.get_pull(int(pull_request_number))
    # labels = [label.name for label in pull_request.labels]
    # replace "_" with "-" in labels
    # labels = [label.replace("_", "-") for label in labels]
    label_name = label_name.replace("_", "-")
    pull_request.create_issue_comment(f"Running experiment on sagemaker with git sha {git_sha}")
    
    datasets = {
        "tat-truck": {"S3Uri": "s3://nerf-dataset-collection/tanks_and_temples/truck/", "MaxRuntimeInSeconds": 4*3600},
        "tat-truck-baseline": {"S3Uri": "s3://nerf-dataset-collection/tanks_and_temples/truck_baseline/", "MaxRuntimeInSeconds": 4*3600},
        "tat-train-baseline": {"S3Uri": "s3://nerf-dataset-collection/tanks_and_temples/train_baseline/", "MaxRuntimeInSeconds": 4*3600},
        "tat-train": {"S3Uri": "s3://nerf-dataset-collection/tanks_and_temples/train/", "MaxRuntimeInSeconds": 4*3600},
        "garden": {"S3Uri": "s3://nerf-dataset-collection/Mip-NeRF360/garden_4/", "MaxRuntimeInSeconds": 4*3600},
        "stump": {"S3Uri": "s3://nerf-dataset-collection/Mip-NeRF360/stump/", "MaxRuntimeInSeconds": 4*3600},
        "bicycle": {"S3Uri": "s3://nerf-dataset-collection/Mip-NeRF360/bicycle/", "MaxRuntimeInSeconds": 4*3600},
    }
    if label_name != "need-experiment":
        # if the pull request does not have "need-experiment" label, it must have "need-experiment-<dataset>" label
        selected_datasets = [label_name[len("need-experiment-"):]]
        datasets = {dataset: datasets[dataset] for dataset in selected_datasets}

    sagemaker_client = boto3.client("sagemaker", region_name="us-east-2")
    experiment = Experiment.create(
        experiment_name=experiment_name,
        description=f"Experiment triggered by pull request https://github.com/{git_repo}/pull/{pull_request_number}",
        sagemaker_boto_client=sagemaker_client,
    )

    # we use prebuilt docker image for training, so we need to clone the repo and install the dependencies at entrypoint
    entrypoint = [
        "bash",
        "ci/entrypoint.sh",
    ]
    
    train_job_names = []
    train_job_name_to_output_path = {}
    for dataset, config_to_fill in datasets.items():
        trial = Trial.create(
            trial_name=f"{experiment_name}-{dataset}",
            experiment_name=experiment_name,
            sagemaker_boto_client=sagemaker_client,
        )

        train_job_name = f"{experiment_name}-{dataset}"
        if len(train_job_name) > 63:
            train_job_name = train_job_name[63 - len(train_job_name):]
        train_job_names.append(train_job_name)
        full_s3_output_path = os.path.join(s3_output_dir, dataset)
        train_job_name_to_output_path[train_job_name] = full_s3_output_path
        with open("config/ci_sagemaker_template.json") as f:
            train_job_config = json.load(f)
        train_job_config["TrainingJobName"] = train_job_name
        train_job_config["AlgorithmSpecification"]["TrainingImage"] = image_uri
        train_job_config["AlgorithmSpecification"]["ContainerEntrypoint"] = entrypoint
        train_job_config["InputDataConfig"][0]["DataSource"]["S3DataSource"]["S3Uri"] = config_to_fill.get("S3Uri", "")
        train_job_config["StoppingCondition"]["MaxRuntimeInSeconds"] = config_to_fill.get("MaxRuntimeInSeconds", 4*3600)
        train_job_config["StoppingCondition"]["MaxWaitTimeInSeconds"] = config_to_fill.get("MaxRuntimeInSeconds", 4*3600) * 2
        
        train_job_config["Environment"] = {
            "TRAIN_CONFIG": train_job_config["HyperParameters"]["train_config"],
        }
        train_job_config["OutputDataConfig"]["S3OutputPath"] = full_s3_output_path
        sagemaker_client.create_training_job(**train_job_config)
        pull_request.create_issue_comment(f"Training job [{train_job_name}]({JOB_URL_FORMAT.format(train_job_name)}) created")

        trial_component_name = f"{train_job_name}-aws-training-job"
        trial_component = TrialComponent.load(trial_component_name=trial_component_name)
        trial.add_trial_component(trial_component)

    # wait for training jobs to finish
    finished_jobs = set()
    final_metrics_commented_jobs = set()
    last_comment_metric_time = time.time()
    any_job_failed = False
    
    # each train job has a dict of metrics, each metric has a dict of (timestamp, value) pair
    train_job_metrics = {train_job_name: defaultdict(lambda:defaultdict(dict)) for train_job_name in train_job_names}
    while True:
        all_jobs_completed = True
        reset_last_comment_metric_time = False
        for train_job_name in train_job_names:
            train_job_description = sagemaker_client.describe_training_job(TrainingJobName=train_job_name)
            train_job_status = train_job_description["TrainingJobStatus"]
            if train_job_status in ["Failed", "Stopped"]:
                pull_request.create_issue_comment(f"Training job {train_job_name} failed")
                any_job_failed = True
            elif train_job_status == "Completed":
                if train_job_name not in finished_jobs:
                    model_url = os.path.join(train_job_name_to_output_path[train_job_name], train_job_name, "output", "model.tar.gz")
                    tensorboard_output_path = os.path.join(train_job_name_to_output_path[train_job_name], train_job_name, "output", "output.tar.gz")
                    comment = f"# Training job [{train_job_name}]({JOB_URL_FORMAT.format(train_job_name)}) completed. \n## Model url: {model_url}, \n## tensorboard output path: {tensorboard_output_path}\n"
                    pull_request.create_issue_comment(comment)
                    finished_jobs.add(train_job_name)
                if train_job_name not in final_metrics_commented_jobs:
                    comment = comment_all_metrics(
                        train_job_metrics=train_job_metrics,
                        train_job_name=train_job_name,
                    )
                    if comment != "":
                        final_metrics_commented_jobs.add(train_job_name)
                        comment = f"Training job [{train_job_name}]({JOB_URL_FORMAT.format(train_job_name)}) final metrics: \n" + comment
                        pull_request.create_issue_comment(comment)
                
            else:
                all_jobs_completed = False
            # get metrics
            for metric in train_job_description.get("FinalMetricDataList", []):
                metric_name = metric["MetricName"]
                metric_timestamp = metric["Timestamp"]
                metric_value = metric["Value"]
                train_job_metrics[train_job_name][metric_name][metric_timestamp] = metric_value
            # try comment on metrics every 10 minutes
            if time.time() - last_comment_metric_time > 600:
                # the following code is commented out because it is not working
                # Final metric data is not available until the training job is completed
                # TODO: find a way to get the real-time metrics, may be by using CloudWatch API
                """
                comment = f"Training job [{train_job_name}]({JOB_URL_FORMAT.format(train_job_name)}) metrics every 10 minutes: \n"
                comment += comment_all_metrics(
                    train_job_metrics=train_job_metrics,
                    train_job_name=train_job_name,
                )
                pull_request.create_issue_comment(comment)
                """
                reset_last_comment_metric_time = True
        if reset_last_comment_metric_time:
            last_comment_metric_time = time.time()
                
        if all_jobs_completed:
            break
        print("Waiting for training jobs to finish")
        time.sleep(60)

    for train_job_name in train_job_names:
        comment = f"Training job [{train_job_name}]({JOB_URL_FORMAT.format(train_job_name)}) final metrics: \n"
        comment += comment_all_metrics(
            train_job_metrics=train_job_metrics,
            train_job_name=train_job_name,
        )
        pull_request.create_issue_comment(comment)
    if any_job_failed:
        exit(1)

        