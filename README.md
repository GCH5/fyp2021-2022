# Video Multi-Object Tracking and Queue Analysis for Smart City - Backend

> Queue analysis in one minute

**This repository contains the source code for the backend part of our FYP. The code here should be run in conjunction with the [frontend part](https://github.com/GCH5/fyp-frontend-elementui) of the project.** Your feedback and experience is welcomed in issues and discussions to make a better user experience.

## <div align="center">Quick Start Examples</div>

<details open>
<summary>Install</summary>

We recommend getting started using [Miniconda](https://conda.io/en/latest/miniconda.html). Clone repo and create the conda environment.

```bash
git clone https://github.com/GCH5/fyp2021-2022  # clone
cd fyp2021-2022
conda create -f requirements.yml  # install
```

</details>

<details open>
<summary>Run</summary>

The default API host for the frontent application is `http://localhost:5000`, so here we start the development server by

```bash
flask run -p 5000
```

</details>

The backend server is a simple [Flask](https://flask.palletsprojects.com/) server, on which the tracking module, consisting of [YOLOv5](https://github.com/ultralytics/yolov5) and [DeepSORT](https://github.com/nwojke/deep_sort) is running.

## <div align="center">Dataset</div>

[train]()(500 iamges in the train set, 100 in the test set)
[test]()

## <div align="center">Pretrained weights</div>

1. [yolo_dead_detection_best.pt](https://drive.google.com/file/d/1MLnIzWUGrnBFfb25LeGBiMriWvaTATY8/view?usp=sharing): For queue detection, detect heads only

2. [ckpt.t7](https://drive.google.com/file/d/1_qwTWdzT9dWNudpusgKavj_4elGgbkUN/view?usp=sharing): Feature extractor weights in pytorch, used in the DeepSORT tracker
