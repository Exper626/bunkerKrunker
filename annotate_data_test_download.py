from roboflow import Roboflow
rf = Roboflow(api_key="9C2caCqzmEncf1AryQHq")
project = rf.workspace("hanaaexperiment626").project("demetergetstea-ciif2")
version = project.version(3)
dataset = version.download("yolov8")
                