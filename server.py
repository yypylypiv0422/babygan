from flask import Flask, request, jsonify, json
import subprocess
import os
from multiprocessing import set_start_method
from time import time

try:
    set_start_method('spawn')
except:
    pass
# import app as user_src

app = Flask(__name__)


# loading all models
# user_src.init()

# Healthchecks verify that the environment is correct on Banana Serverless
@app.route('/healthcheck', methods=["GET"])
def healthcheck():
    # dependency free way to check if GPU is visible
    gpu = False
    out = subprocess.run("nvidia-smi", shell=True)
    if out.returncode == 0:  # success state on shell command
        gpu = True

    return request.json({"state": "healthy", "gpu": gpu})


# Inference POST handler at '/' is called for every http call from Banana
@app.route('/', methods=["POST"])
def inference():
    # try:
    #     model_inputs = request.json.loads(request.json)
    # except:
    #     model_inputs = request.json
    #
    # start = time()
    # output = user_src.inference(model_inputs)
    # total_time = time() - start
    # print(total_time,"----------------------total_time")
    # print(output, 'lllkkk')
    # response = jsonify(output=str(output))
    # print(response.json, 'kkkk')


    return response.json


if __name__ == '__main__':
    # app.run(debug=True, host = '0.0.0.0', port = 8000, threaded=False)
    app.run(debug=False, host='0.0.0.0', port='8000', threaded=False, use_reloader=False)