import os
from typing import Dict
import logging
from flask import Flask, request, jsonify, after_this_request
from os.path import getsize
from flask_cors import cross_origin
import queue_detect_no_velocity

UPLOAD_FOLDER = "upload"
OUTPUT_DIR = "out"

ALLOWED_EXTENSIONS = {"mp4", "avi", "json"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["DEBUG"] = True
CHUNK_SIZE = 8192


class Area:
    def __init__(self, uid):
        self.queue_polygon = []
        self.finish_polygon = []
        self.uid = uid


requests: Dict[str, Area] = {}

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


@app.route("/upload", methods=["POST"])
@cross_origin(expose_headers="json-size")
def upload_file():
    logging.info(request.files)
    try:
        file = request.files["video"]
        basename, ext = os.path.splitext(file.filename)
        uid = basename
        filename = uid + ext
        if not os.path.exists(app.config["UPLOAD_FOLDER"]):
            os.makedirs(app.config["UPLOAD_FOLDER"])
        upload_file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
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
    logging.debug(requestArea.finish_polygon)
    queue_detect_no_velocity.run(
        weights="../yolo_head_detection.pt",
        source=upload_file_path,
        output_filename=uid,
        finish_area=requestArea.finish_polygon,
        queue_polygon=requestArea.queue_polygon,
        device="0",
        save_video=True,
    )
    os.remove(upload_file_path)
    outputVideoName = os.path.join(OUTPUT_DIR, uid + ".mp4")
    outputJsonName = os.path.join(OUTPUT_DIR, uid + ".json")

    json_size = getsize(outputJsonName)
    response = app.response_class(
        read_file_chunks(outputVideoName, outputJsonName),
        mimetype="application/octet-stream",
    )
    response.headers["json-size"] = json_size
    logging.debug(json_size)

    @after_this_request
    def remove_output(response):
        logging.debug("remove output files")
        del requests[uid]
        return response

    return response


@app.route("/params", methods=["POST"])
@cross_origin()
def uploadParams():
    outputFiles = os.listdir(OUTPUT_DIR)
    uids = set(map(lambda filename: os.path.splitext(filename)[0], outputFiles))
    for uid in uids:
        if uid not in requests:
            outputVideoName = os.path.join(OUTPUT_DIR, uid + ".mp4")
            outputJsonName = os.path.join(OUTPUT_DIR, uid + ".json")
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
