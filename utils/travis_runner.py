#!/usr/bin/env python
"""This script manages all tasks for the TRAVIS build server."""
import glob
import subprocess as sp

if __name__ == "__main__":

    for notebook in glob.glob("*.ipynb"):
        cmd = " jupyter nbconvert --execute {}  --ExecutePreprocessor.timeout=-1".format(
            notebook
        )
        sp.check_call(cmd, shell=True)
