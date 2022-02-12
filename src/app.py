import os
import uuid
from flask import Flask, flash, request, redirect, render_template, url_for, jsonify
from werkzeug.utils import secure_filename
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
    queue_polygon = []
    finish_polygon = []
    uid = 0


if __name__ == "__main__":
    app.run(host="0.0.0.0")


def read_file_chunks(outputVideoName, outputJsonName):
    with open(outputJsonName, "rb") as fd:
        while True:
            buf = fd.read(CHUNK_SIZE)
            print("reading json file")
            if buf:
                yield buf
            else:
                break
    with open(outputVideoName, "rb") as fd:
        while True:
            buf = fd.read(CHUNK_SIZE)
            # print("reading video file")
            if buf:
                yield buf
            else:
                break


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def upload():
    if request.method == "GET":
        return render_template("base.html")

    #     if request.method == 'POST':
    #         # check if the post request has the file part
    #         if 'file' not in request.files:
    #             flash('No file part')
    #             return redirect(request.url)
    #         file = request.files['file']
    # D        # If the user does not select a file, the browser submits an
    #         # empty file without a filename.
    #         if file.filename == '':
    #             flash('No selected file')
    #
    #         if file:
    #             filename = secure_filename(file.filename)
    #             filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    #             file.save(filepath)
    # server_main.run(source=filepath, save_img=True, device='cpu', output_dir=OUTPUT_DIR)
    # result = read_result()
    # os.remove(filepath)
    # res = first_frame(filepath)
    # return render_template('base.html', data=result[1][-1], pic=draw_pic(result[0], result[1]))
    return render_template("base.html", pic=res)
    # return render_template('base.html')


@app.route("/upload", methods=["POST"])
@cross_origin(expose_headers="json-size")
def upload_file():
    @after_this_request
    def remove_file(response):
        try:
            os.remove(filepath)
        except Exception as error:
            app.logger.error("Error removing or closing downloaded file handle", error)
        return response

    print(request.files)
    if request.method == "POST":
        if "video" in request.files:
            file = request.files["video"]
            filename = str(uuid.uuid4()) + "_" + secure_filename(file.filename)
            if not os.path.exists(app.config["UPLOAD_FOLDER"]):
                os.makedirs(app.config["UPLOAD_FOLDER"])
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
    """
    pass the uploaded video and parameters to the pytorch model
    generate output.mp4 and output.json
    """
    print(Area.queue_polygon)
    print(Area.finish_polygon)
    queue_detect_no_velocity.run(
        weights="../yolo_head_detection.pt",
        source=filepath,
        finish_area=Area.finish_polygon,
        queue_polygon=Area.queue_polygon,
        device="0",
        save_video=True,
        debug_frames=61,
    )

    outputVideoName = "./out/out.webm"
    outputJsonName = "./out/output.json"

    json_size = getsize(outputJsonName)
    response = app.response_class(
        read_file_chunks(outputVideoName, outputJsonName),
        mimetype="application/octet-stream",
    )
    response.headers["json-size"] = json_size
    print(json_size)
    print("return response success")
    return response


@app.route("/params", methods=["POST"])
@cross_origin()
def uploadParams():
    print(request)
    if request.method == "POST":
        print(request.json)
        Area.queue_polygon.clear()
        Area.finish_polygon.clear()
        Area.uid = request.json["videoUid"]
        qaPoints = request.json["queueAreaPoints"]
        faPoints = request.json["finishAreaPoints"]
        for point in qaPoints:
            Area.queue_polygon.append(point[0])
            Area.queue_polygon.append(point[1])
        for point in faPoints:
            Area.finish_polygon.append(point[0])
            Area.finish_polygon.append(point[1])
        return jsonify(succuss=True, status=200)
