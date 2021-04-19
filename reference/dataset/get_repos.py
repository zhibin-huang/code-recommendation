import csv
import os
import subprocess
import shutil
import stat

def read_csv(filename):
    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        sources = [row['repository'] for row in reader]
        return sources


def downLoad_repo(sources):
    urls = []
    prefix = "git clone --depth=1 https://github.com/"
    def readonly_handler(func, path, execinfo):
        os.chmod(path, stat.S_IWRITE)
        func(path)
    for i in sources:
        dest = os.getcwd() + "/repos/" + i
        if os.path.exists(dest):
            shutil.rmtree(dest, onerror=readonly_handler)
        subprocess.call(prefix + i + " " + dest, shell=True) 


s = read_csv('github_repositories.csv')
downLoad_repo(s)