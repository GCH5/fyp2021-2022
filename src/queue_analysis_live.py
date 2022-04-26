import base64
import json
from pathlib import Path
import time
import cv2

import numpy as np
import torch

from shapely.geometry import Polygon
from shapely.geometry import Point
from deep_sort.tracker import Tracker
from deep_sort import nn_matching
from my_utils.encoder_torch import Extractor
from models.experimental import attempt_load
from my_utils.my_dataset import LoadStreams
from my_utils import utils
from deep_sort import detection
from my_utils.queuer import Queuer, PotentialQueuer
import logging

identity_switch_thres = 30

logging.basicConfig(filename="queue_analysis.log", level=logging.DEBUG)


class QueueAnalysis:
    def __init__(
        self,
        weights="yolov5l.pt",  # model.pt path(s)
        device="cpu",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        source="frames",  # file/dir/URL/glob, 0 for webcam
        finish_area=[866, 650, 1172, 473, 1281, 555, 990, 776],
        queue_polygon=[717, 101, 1453, 515, 1107, 777, 416, 352],  # x y x y x y x y x y
        half=False,  # use FP16 half-precision inference
    ) -> None:
        self.source = source
        self.queue_vertices = []
        for x, y in zip(*[iter(queue_polygon)] * 2):  # loop 2 coords at a time
            self.queue_vertices.append((x, y))
        self.queue_polygon = Polygon(self.queue_vertices)
        logging.debug("queue_polygon loaded")
        self.finish_vertices = []
        for x, y in zip(*[iter(finish_area)] * 2):  # loop 2 coords at a time
            self.finish_vertices.append((x, y))
        self.finish_polyogn = Polygon(self.finish_vertices)
        logging.debug("finish_polyogn loaded")

        self.device = utils.select_device(device)
        self.use_gpu = True  # device == torch.device("cuda:0")
        logging.debug(device)
        half &= self.device.type != "cpu"  # half precision only supported on CUDA

        self.model = attempt_load(weights, map_location=self.device)  # load FP32 model
        logging.debug("model loaded")
        stride = int(self.model.stride.max())  # model stride
        self.half = half
        if half:
            self.model.half()

        self.encoder = Extractor(str(Path("../model") / Path("ckpt.t7")))
        logging.debug("Loading encoder")
        max_cosine_distance = 0.2
        nn_budget = None
        metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", max_cosine_distance, nn_budget
        )
        self.tracker = Tracker(metric)
        self.dataset = LoadStreams(source, img_size=640, stride=stride)
        self.fps = self.dataset.getFps()[0]
        logging.info(f"fps: {self.fps}")
        self.current_waiting_time = 0.0
        self.queue = {}
        self.potential_queue = {}
        self.queue_time = {}
        self.frame_idx = 0

    def run(
        self,
        conf_thres=0.5,  # confidence threshold
        iou_thres=0.2,  # NMS IOU threshold
        enqueue_thres=10,
        dequeue_thres=10,
        finish_thres=1,
    ):
        logging.debug("✈Start queue analysis✈")
        for _, img, im0s, _ in self.dataset:
            self.frame_idx += 1

            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            results = self.model(img)[0]
            results = utils.non_max_suppression(results, conf_thres, iou_thres)
            if results[0].shape[0] == 0:
                continue

            det = results[0]
            det[:, :4] = utils.scale_coords(
                img.shape[2:], det[:, :4], im0s[0].shape
            ).round()
            person_ind = [i for i, cls in enumerate(det[:, -1]) if int(cls) == 0]
            xyxy = det[person_ind, :-2]  # find person only
            xywh_boxes = utils.xyxy2xywh(xyxy)
            tlwh_boxes = utils.xywh2tlwh(xywh_boxes)
            confidence = det[:, -2]
            if self.use_gpu:
                tlwh_boxes = tlwh_boxes.cpu()
                xyxy = xyxy.cpu()
            xyxy_boxes = np.array(xyxy).astype(int)

            features = self.encoder.get_features(xyxy_boxes, im0s[0])

            detections = [
                detection.Detection(bbox, confidence, "person", feature)
                for bbox, confidence, feature in zip(tlwh_boxes, confidence, features)
            ]

            # Call the tracker
            self.tracker.predict()
            self.tracker.update(detections)

            in_queueing_area = []
            queueLength = len(self.queue.keys())
            currentWaitingTime = str(round(self.current_waiting_time, 1))

            cv2.putText(
                im0s[0],
                "Queue length: {}".format(len(list(self.queue.keys()))),
                (100, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (49, 49, 247),
                2,
            )
            cv2.putText(
                im0s[0],
                "ID: {}".format(str(list(self.queue.keys()))[1:-1]),
                (100, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (49, 49, 247),
                2,
            )
            cv2.putText(
                im0s[0],
                "Last waiting time: {}".format(currentWaitingTime),
                (100, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (49, 49, 247),
                2,
            )
            cv2.putText(
                im0s[0],
                "ID, time elapsed: {}".format(str((self.queue_time))[1:-1]),
                (100, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (49, 49, 247),
                2,
            )
            pts = np.array(self.queue_vertices, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(im0s[0], [pts], True, (255, 0, 0), 2)
            pts = np.array(self.finish_vertices, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(im0s[0], [pts], True, (255, 255, 0), 2)

            for track in self.tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                track_id = track.track_id
                bbox = track.to_tlbr()
                center_x = int((bbox[0] + bbox[2]) / 2)
                center_y = int((bbox[1] + bbox[3]) / 2)
                cv2.putText(
                    im0s[0],
                    "ID: " + str(track_id),
                    (int(center_x), int(center_y)),
                    0,
                    1e-3 * im0s[0].shape[0],
                    (255, 0, 0),
                    1,
                )
                if self.queue_polygon.intersects(Point(center_x, center_y)):
                    if track_id in list(self.potential_queue.keys()):
                        start_frame = self.potential_queue[track_id].start_frame
                        self.potential_queue[track_id].accumulated_frames += 1
                        accumulated_frames = self.potential_queue[
                            track_id
                        ].accumulated_frames
                        elapse_frames = self.frame_idx - start_frame
                        if (
                            elapse_frames > enqueue_thres * self.fps
                            and accumulated_frames > elapse_frames * 0.8
                        ):
                            self.queue[track_id] = Queuer(
                                start_frame,
                                (center_x, center_y),
                                start_frame,
                                False,
                                None,
                            )
                            in_queueing_area.append(track_id)
                            del self.potential_queue[track_id]
                        cv2.rectangle(
                            im0s[0],
                            (int(bbox[0]), int(bbox[1])),
                            (int(bbox[2]), int(bbox[3])),
                            (0, 255, 0),
                            2,
                        )
                        cv2.putText(
                            im0s[0],
                            str(round((self.frame_idx - start_frame) / self.fps, 1)),
                            (int(center_x), int(bbox[1])),
                            0,
                            1e-3 * im0s[0].shape[0],
                            (0, 255, 0),
                            1,
                        )
                    elif track_id in self.queue:
                        if not self.queue[track_id].enter_finish_area_frame:
                            self.queue[track_id].last_frame = self.frame_idx
                        self.queue[track_id].position = (center_x, center_y)
                        in_queueing_area.append(track_id)
                        cv2.rectangle(
                            im0s[0],
                            (int(bbox[0]), int(bbox[1])),
                            (int(bbox[2]), int(bbox[3])),
                            (0, 0, 255),
                            2,
                        )
                        cv2.putText(
                            im0s[0],
                            str(
                                round(
                                    (self.frame_idx - self.queue[track_id].start_frame)
                                    / self.fps,
                                    1,
                                )
                            ),
                            (int(center_x), int(bbox[1])),
                            0,
                            1e-3 * im0s[0].shape[0],
                            (0, 0, 255),
                            1,
                        )

                    else:
                        addToPotential = True
                        if addToPotential:
                            self.potential_queue[track_id] = PotentialQueuer(
                                self.frame_idx, 0
                            )

            for track_id in list(self.queue.keys()):
                queueing_time = round(
                    (self.queue[track_id].last_frame - self.queue[track_id].start_frame)
                    / self.fps,
                    1,
                )
                self.queue_time[track_id] = queueing_time
                if not self.queue[
                    track_id
                ].enter_finish_area_frame and self.finish_polyogn.intersects(
                    Point(self.queue[track_id].position)
                ):
                    self.queue[track_id].enter_finish_area_frame = self.frame_idx
                if (
                    self.queue[track_id].enter_finish_area_frame
                    and not self.queue[track_id].finish_queueing
                ):
                    inside_finish_area_time = (
                        self.frame_idx - self.queue[track_id].enter_finish_area_frame
                    ) / self.fps
                    if inside_finish_area_time > finish_thres:
                        self.queue[track_id].finish_queueing = True
                if (
                    track_id not in in_queueing_area
                    and (self.frame_idx - self.queue[track_id].last_frame) / self.fps
                    > dequeue_thres
                ):
                    """
                    if queue[track_id].finish_queueing:
                        queueing_time = (queue[track_id].last_frame - queue[track_id].start_frame) / fps
                        queue_time[track_id] = queueing_time
                    """
                    self.current_waiting_time = self.queue_time[track_id]
                    del self.queue_time[track_id]
                    del self.queue[track_id]
            if (self.frame_idx) % 30 != 0:
                continue
            _, buffer = cv2.imencode(".jpg", im0s[0])
            jpg_as_text = base64.b64encode(buffer).decode("utf-8")
            data = {
                "streamUrl": self.source,
                "time": round(time.time() * 1000),
                "queueLength": queueLength,
                "currentWaitingTime": currentWaitingTime,
                "cap": jpg_as_text,
            }

            msg = f"data: {json.dumps(data)}\n\n"
            try:
                yield msg
            except GeneratorExit:
                logging.info("generator exit")
                break

        logging.debug(self.queue_time)
        logging.debug(self.current_waiting_time)
