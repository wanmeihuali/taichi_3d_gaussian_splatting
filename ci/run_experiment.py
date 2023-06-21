import os
import sagemaker
import argparse
from github import Github

if __name__ == "__main__":
    parser = argparse.ArgumentParser("run experiment on sagemaker")
    parser.add_argument("--experiment-name", type=str, default="sagemaker-experiment")
    parser.add_argument("--s3-output-path", type=str, required=True)

    args = parser.parse_args()
    github_access_token = os.environ["GITHUB_ACCESS_TOKEN"]
    pull_request_number = os.environ["PULL_REQUEST_NUMBER"]
    git_owner = os.environ["GITHUB_OWNER"]
    git_repo = os.environ["GITHUB_REPO"]
    git_sha = os.environ["GITHUB_SHA"]
    github_client = Github(github_access_token)
    print(f"Getting pull request {pull_request_number} from {git_owner}/{git_repo}")
    github_repo = github_client.get_repo(f"{git_repo}")
    pull_request = github_repo.get_pull(int(pull_request_number))
    pull_request.create_issue_comment(f"Running experiment on sagemaker with git sha {git_sha}")
    
