import argparse
import subprocess

def run_programA():
    # 调用第一个程序的命令
    subprocess.run(["python", "./Node2vec/utility/A_e2v_walks.py"])

def run_programB():
    # 调用第二个程序的命令
    subprocess.run(["python", "./Node2vec/utility/B_learn_vecs.py"])

def run_programC():
    # 调用第三个程序的命令
    subprocess.run(["python", "./Node2vec/utility/C_dis_gene_pred.py"])

def run_programD():
    # 调用第三个程序的命令
    subprocess.run(["python", "./Node2vec/utility/D_renumber.py"])

def run_programE():
    # 调用第三个程序的命令
    subprocess.run(["python", "./Node2vec/utility/E_getMoudle_Embeeding.py"])




if __name__ == "__main__":


    # 依次调用程序
    run_programA()
    run_programB()
    run_programC()
    run_programD()
    run_programE()
