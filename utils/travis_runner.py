#!/usr/bin/env python
"""This script manages all tasks for the TRAVIS build server."""
import glob
import os
import subprocess as sp

if __name__ == "__main__":
    sp.check_call(
        "pytest scrypy/tests --cov=scrypy/tests --cov-report term --cov-report xml:coverage.xml",
        shell=True,
    )
    # Commented out to save travis around 5 minutes.
    # sp.check_call("python python/jac_estimation_chol.py", shell=True)
    sp.check_call("python scrypy/script_uncertainty_propagation.py", shell=True)
    os.chdir("notebooks")
    for notebook in glob.glob("*.ipynb"):
        cmd = " jupyter nbconvert --execute {} --ExecutePreprocessor.timeout=-1".format(
            notebook
        )
        sp.check_call(cmd, shell=True)
