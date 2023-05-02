import sys
import os
from retrainer.app.app import app
from flask import render_template, request

sys.path.append(os.getcwd())


@app.errorhandler(404)
def not_found(e):
    return render_template("400.html")
