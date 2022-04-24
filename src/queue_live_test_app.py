import flask
from flask_cors import cross_origin
from flask import stream_with_context, request
import time

app = flask.Flask(__name__)
import queue_analysis_live


finish_polygon = [866, 650, 1172, 473, 1281, 555, 990, 776]
queue_polygon = [717, 101, 1453, 515, 1107, 777, 416, 352]


@app.route("/listen", methods=["GET"])
@cross_origin()
def listen():
    return app.response_class(
        stream_with_context(
            queue_analysis_live.run(
                weights="../yolo_head_detection.pt",
                source=request.args.get("streamUrl"),
                finish_area=request.args.get("finishPolygon"),
                queue_polygon=request.args.get("queuePolygon"),
                device="0",
                save_video=False,
                plot_tracking=True,
            )
        ),
        mimetype="text/event-stream",
    )


@app.route("/test", methods=["GET"])
def test():
    return str(time.time())


if __name__ == "__main__":
    app.run(host="0.0.0.0")
