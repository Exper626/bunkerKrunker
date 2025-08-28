from roboflow import Roboflow

rf = Roboflow(api_key="x2IY0bfrKK6cFOxdpRCd")
project = rf.workspace("experiment626lab").project("demeter_dataset-8k0kc")
version = project.version(1)
dataset = version.download("yolov8")
                