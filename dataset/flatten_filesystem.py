import os
import shutil


destination = "files/bach/flattened"
source = "files/bach/bach"

for dirname, dirnames, filenames in os.walk(source):

    # print path to all filenames.
    for filename in filenames:
        src = os.path.join(dirname, filename)
        dst = os.path.join(destination, filename)
        shutil.copyfile(src,dst)