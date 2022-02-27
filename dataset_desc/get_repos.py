import csv
import os
import subprocess
import shutil
import stat

def read_csv(filename):
    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        repositories = [row['repository'] for row in reader]
        return repositories


def downLoad_repo(repositories):
    prefix = "git clone --depth=1 https://github.com/"
    def readonly_handler(func, path, execinfo):
        os.chmod(path, stat.S_IWRITE)
        func(path)
    for r in repositories:
        dest = os.getcwd() + "/reference/repos_new/" + r
        if os.path.exists(dest):
            shutil.rmtree(dest, onerror=readonly_handler)
        subprocess.call(prefix + r + " " + dest, shell=True) 

filecnt = 0
def get_java_path(path: str, output_file):
    if os.path.exists(path):
        for wholepath in [os.path.join(path, f) for f in os.listdir(path)]:
            if os.path.isdir(wholepath):
                get_java_path(wholepath, output_file)
            elif os.path.isfile(wholepath):
                if wholepath.endswith(".java") and 'test' not in wholepath.lower():
                    relpath = os.path.relpath(wholepath)
                    output_file.write(relpath)
                    output_file.write('\n')
                    global filecnt
                    filecnt = filecnt + 1
                    print("file quantity: " + str(filecnt) + ',' + relpath)
                else:
                    os.remove(wholepath)

# repositories = read_csv(os.getcwd() + '/reference/dataset_desc/github_repositories.csv')
# downLoad_repo(repositories)
with open("reference/tmpout/records_path.txt", "w") as f:
    get_java_path("reference/repos_new/", f)