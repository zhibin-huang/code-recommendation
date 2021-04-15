import csv
import os
import subprocess
import shutil

def read_csv(filename):
    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        sources = [row['repository'] for row in reader]
        return sources


def downLoad_repo(sources):
    urls = []
    prefix = "git clone --depth=1 https://github.com/"
    for i in sources:
        dest = os.getcwd() + "/repos/" + i
        subprocess.call(prefix + i + " " + dest, shell=True) 
        get_java(dest)

cnt = 0
def get_java(path):
    if os.path.exists(path):
        print("entering " + path)
        for f in os.listdir(path):
            wholepath = os.path.join(path,f)
            if os.path.isdir(wholepath):
                get_java(wholepath)
                wholepath = os.path.join(path,f) # 递归退出，恢复路径
            if os.path.isfile(wholepath):
                if not wholepath.endswith(".java"):
                    print("removing " + wholepath)
                    os.remove(wholepath)
                else:
                    global cnt
                    print("find " + wholepath)
                    shutil.copy(wholepath, os.getcwd() + "/java/" + str(cnt) + '.java')
                    cnt+=1

s = read_csv('github_repositories.csv')
downLoad_repo(s)