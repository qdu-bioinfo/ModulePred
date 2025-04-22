import argparse
import subprocess
from utility.config import parse_args

def run_programA(args):

    subprocess.run(["python", "./utility/CN_score.py"])

def run_programB(args):

    subprocess.run(["python", "./utility/RA_score.py"])

def run_programC(args):

    subprocess.run(["python", "./utility/AA_score.py"])

def run_programE(args):

    subprocess.run(["python", "./utility/Candidate_edge.py"])


if __name__ == "__main__":
    args = parse_args()

    # 依次调用程序
    run_programA(args)
    run_programB(args)
    run_programC(args)
    run_programE(args)