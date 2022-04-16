import os
from typing import Dict
import logging
from flask import Flask, request, jsonify, after_this_request, stream_with_context
from os.path import getsize
from flask_cors import cross_origin
import json

# from CounTr_code import crowd_counting_infer
import queue_analysis_static
import queue_analysis_live

QUEUE_ANALYSIS_UPLOAD_FOLDER = "queue_analysis_upload"
QUEUE_ANALYSIS_OUTPUT_DIR = "queue_analysis_out"

CROWD_COUNTING_UPLOAD_FOLDER = "crowd_counting_upload"
CROWD_COUNTING_OUTPUT_DIR = "crowd_counting_out"

ALLOWED_EXTENSIONS = {"mp4", "avi", "json"}

app = Flask(__name__)
app.config["QUEUE_ANALYSIS_UPLOAD_FOLDER"] = QUEUE_ANALYSIS_UPLOAD_FOLDER
app.config["CROWD_COUNTING_UPLOAD_FOLDER"] = CROWD_COUNTING_UPLOAD_FOLDER
app.config["DEBUG"] = True
CHUNK_SIZE = 8192

if not os.path.exists(QUEUE_ANALYSIS_UPLOAD_FOLDER):
    os.makedirs(QUEUE_ANALYSIS_UPLOAD_FOLDER)
if not os.path.exists(QUEUE_ANALYSIS_OUTPUT_DIR):
    os.makedirs(QUEUE_ANALYSIS_OUTPUT_DIR)
if not os.path.exists(CROWD_COUNTING_UPLOAD_FOLDER):
    os.makedirs(CROWD_COUNTING_UPLOAD_FOLDER)
if not os.path.exists(CROWD_COUNTING_OUTPUT_DIR):
    os.makedirs(CROWD_COUNTING_OUTPUT_DIR)


class Area:
    def __init__(self, uid):
        self.queue_polygon = []
        self.finish_polygon = []
        self.uid = uid


requests: Dict[str, Area] = {}
liveAnalysisInstance: Dict[str, queue_analysis_live.QueueAnalysis] = {}

if __name__ == "__main__":
    app.run(host="0.0.0.0")


def read_file_chunks(outputVideoName, outputJsonName):
    logging.debug("reading json file")
    with open(outputJsonName, "rb") as fd:
        while True:
            buf = fd.read(CHUNK_SIZE)
            if buf:
                yield buf
            else:
                break
    logging.debug("reading video file")
    with open(outputVideoName, "rb") as fd:
        while True:
            buf = fd.read(CHUNK_SIZE)
            if buf:
                yield buf
            else:
                break


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/queue-analysis/video-upload", methods=["POST"])
@cross_origin(expose_headers="json-size")
def queue_analysis_upload_video():
    logging.info(request.files)
    try:
        file = request.files["video"]
        basename, ext = os.path.splitext(file.filename)
        uid = basename
        filename = uid + ext
        if not os.path.exists(app.config["QUEUE_ANALYSIS_UPLOAD_FOLDER"]):
            os.makedirs(app.config["QUEUE_ANALYSIS_UPLOAD_FOLDER"])
        upload_file_path = os.path.join(
            app.config["QUEUE_ANALYSIS_UPLOAD_FOLDER"], filename
        )
        file.save(upload_file_path)
    except Exception as error:
        app.logger.error("Error removing or closing downloaded file handle", error)
        return jsonify(status=400)

    """
    pass the uploaded video and parameters to the pytorch model
    generate output.mp4 and output.json
    """
    logging.info(requests)
    requestArea = requests[uid]
    logging.debug(requestArea.queue_polygon)
    logging.debug(f"finish polygon: {requestArea.finish_polygon}")
    queue_analysis_static.run(
        weights="../yolo_head_detection.pt",
        source=upload_file_path,
        output_filename=uid,
        finish_area=requestArea.finish_polygon,
        queue_polygon=requestArea.queue_polygon,
        device="0",
        save_video=True,
    )
    print("analysis finished")
    os.remove(upload_file_path)
    outputVideoName = os.path.join(QUEUE_ANALYSIS_OUTPUT_DIR, uid + ".mp4")
    outputJsonName = os.path.join(QUEUE_ANALYSIS_OUTPUT_DIR, uid + ".json")

    json_size = getsize(outputJsonName)
    response = app.response_class(
        read_file_chunks(outputVideoName, outputJsonName),
        mimetype="application/octet-stream",
    )
    response.headers["json-size"] = json_size
    logging.debug(json_size)

    @after_this_request
    def remove_output(response):
        logging.debug("remove requests id")
        del requests[uid]
        return response

    return response


@app.route("/crowd-counting/video-upload", methods=["POST"])
@cross_origin(expose_headers="json-size")
def crowd_counting_upload_video():
    logging.info(request.files)
    try:
        file = request.files["video"]
        basename, ext = os.path.splitext(file.filename)
        uid = basename
        filename = uid + ext
        if not os.path.exists(app.config["CROWD_COUNTING_UPLOAD_FOLDER"]):
            os.makedirs(app.config["CROWD_COUNTING_UPLOAD_FOLDER"])
        upload_file_path = os.path.join(
            app.config["CROWD_COUNTING_UPLOAD_FOLDER"], filename
        )
        file.save(upload_file_path)
    except Exception as error:
        app.logger.error("Error removing or closing downloaded file handle", error)
        return jsonify(status=400)

    """
    pass the uploaded video and parameters to the pytorch model
    generate crowd_counting_output.mp4 and crowd_counting_output.json
    """
    logging.info(requests)
    """
    crowd_counting_infer.run(
        source=upload_file_path,
        output_dir="crowd_counting_out",
        output_filename=uid,
        device="0",
        save_video=True,
    )
    """
    os.remove(upload_file_path)
    outputVideoName = os.path.join(CROWD_COUNTING_OUTPUT_DIR, uid + ".mp4")
    outputJsonName = os.path.join(CROWD_COUNTING_OUTPUT_DIR, uid + ".json")

    json_size = getsize(outputJsonName)
    response = app.response_class(
        read_file_chunks(outputVideoName, outputJsonName),
        mimetype="application/octet-stream",
    )
    response.headers["json-size"] = json_size
    logging.debug(json_size)

    return response


@app.route("/queue-analysis/params", methods=["POST"])
@cross_origin()
def queue_analysis_upload_params():
    outputFiles = os.listdir(QUEUE_ANALYSIS_OUTPUT_DIR)
    uids = set(map(lambda filename: os.path.splitext(filename)[0], outputFiles))
    for uid in uids:
        if uid not in requests:
            outputVideoName = os.path.join(QUEUE_ANALYSIS_OUTPUT_DIR, uid + ".mp4")
            outputJsonName = os.path.join(QUEUE_ANALYSIS_OUTPUT_DIR, uid + ".json")
            logging.debug(f"remove output files with uid: {uid}")
            os.remove(outputVideoName)
            os.remove(outputJsonName)
    logging.debug(request)
    if request.method == "POST":
        logging.info(request.json)
        uid = str(request.json["videoUid"])
        if uid in requests:
            requestArea = requests[uid]
        else:
            requestArea = Area(uid)
        requestArea.queue_polygon.clear()
        requestArea.finish_polygon.clear()
        qaPoints = request.json["queueAreaPoints"]
        faPoints = request.json["finishAreaPoints"]
        for point in qaPoints:
            requestArea.queue_polygon.append(point[0])
            requestArea.queue_polygon.append(point[1])
        for point in faPoints:
            requestArea.finish_polygon.append(point[0])
            requestArea.finish_polygon.append(point[1])
        requests[uid] = requestArea
        return jsonify(succuss=True, status=200)


def hashcode(string):
    hash = 0
    if len(string) == 0:
        return hash
    for index in range(len(string)):
        chr = ord(string[index])
        hash = ((hash << 5) - hash) + chr
        hash |= 0  # Convert to 32bit integer
    return hash


@app.route("/queue-analysis/live", methods=["GET"])
@cross_origin()
def listen():
    uid = request.args.get("streamUid")
    print(liveAnalysisInstance[uid])
    logging.info("start listening with id: " + uid)
    return app.response_class(
        stream_with_context(liveAnalysisInstance[uid].run()),
        mimetype="text/event-stream",
    )


@app.route("/queue-analysis/close", methods=["POST"])
@cross_origin()
def closeConnection():
    uid = str(request.json["streamUid"])
    del liveAnalysisInstance[uid]
    logging.info("close connection with id: " + uid)
    return jsonify(succuss=True, status=200)


@app.route("/queue-analysis/register", methods=["POST"])
@cross_origin()
def register():
    logging.debug("register")
    uid = str(request.json["streamUid"])
    queueAnalysisInstance = queue_analysis_live.QueueAnalysis(
        weights="../yolo_head_detection.pt",
        source=request.json["streamUrl"],
        finish_area=request.json["finishAreaPoints"],
        queue_polygon=request.json["queueAreaPoints"],
        device="0",
    )
    liveAnalysisInstance[uid] = queueAnalysisInstance
    return jsonify(succuss=True, status=200)
