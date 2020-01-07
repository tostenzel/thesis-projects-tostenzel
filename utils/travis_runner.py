#!/usr/bin/env python
"""This script manages all tasks for the TRAVIS build server."""
import glob
import subprocess as sp

if __name__ == "__main__":

    sp.check_call("python python/script_uncertainty_propagation.py", shell=True)
    for notebook in glob.glob("*.ipynb"):
        cmd = " jupyter nbconvert --execute {} --ExecutePreprocessor.timeout=-1".format(
            notebook
        )
        sp.check_call(cmd, shell=True)
