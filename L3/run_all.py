import argparse
import subprocess
from utility.config import parse_args

def run_programA(args):
    # 调用第一个程序的命令
    subprocess.run(["python", "./L3/utility/CN_score.py"])

def run_programB(args):
    # 调用第二个程序的命令
    subprocess.run(["python", "./L3/utility/RA_score.py"])

def run_programC(args):
    # 调用第三个程序的命令
    subprocess.run(["python", "./L3/utility/AA_score.py"])

def run_programE(args):
    # 调用第四个程序的命令
    subprocess.run(["python", "./L3/utility/Candidate_edge.py"])


if __name__ == "__main__":
    args = parse_args()

    # 依次调用程序
    run_programA(args)
    run_programB(args)
    run_programC(args)
    run_programE(args)