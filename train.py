import wandb
from detectron2.data.datasets import register_coco_instances
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
import detectron2.data.transforms as T
from detectron2 import model_zoo
from detectron2.config import LazyCall as L

from CustomHooks import LossEvalHook 
import detectron2.data.transforms as T
from detectron2.data import DatasetMapper, build_detection_train_loader, build_detection_test_loader   # the default mapper
from detectron2.evaluation import COCOEvaluator

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# wandb.init(
#     project="Faster_RCNN",
#     name="DroneVsBird",
#     entity="biyotteu", 
#     sync_tensorboard=True,
# )

class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR,"inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)
    
    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1,LossEvalHook(
            cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapper(self.cfg,True)
            )
        ))
        return hooks
    
    # @classmethod
    # def build_test_loader(cls, cfg, dataset_name):
    #     return build_detection_test_loader(cfg,
    #         mapper=DatasetMapper(cfg, is_train=True, augmentations=[
    #             L(T.ResizeShortestEdge)(short_edge_length=1280, max_size=1280),
    #         ]
    #     ))

    # @classmethod
    # def build_train_loader(cls, cfg):
    #     return build_detection_train_loader(cfg,
    #         mapper=DatasetMapper(cfg, is_train=True, augmentations=[
    #             L(T.RandomFlip)(horizontal=True),  # flip first
    #             L(T.ResizeScale)(
    #                 min_scale=0.1, max_scale=2.0, target_height=1280, target_width=1280
    #             ),
    #             L(T.FixedSizeCrop)(crop_size=(1280, 1280), pad=False),
    #         ],
    #         image_format="RGB"
    #     ))

register_coco_instances("train", {}, "/home/ubuntu/dev/datasets/DroneVSBirdOrigin/challenge/coco_annotations/train_all.json", "/home/ubuntu/dev/datasets/DroneVSBirdOrigin/challenge/images")
register_coco_instances("test", {}, "/home/ubuntu/dev/datasets/DroneVSBirdOrigin/challenge/coco_annotations/val.json", "/home/ubuntu/dev/datasets/DroneVSBirdOrigin/challenge/images")

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("train",)
cfg.DATASETS.TEST = ("test",)
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 12  # This is the real "batch size" commonly known to deep learning people
# cfg.SOLVER.CHECKPOINT_PERIOD = 500
# cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
# cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
# cfg.SOLVER.STEPS = []        # do not decay learning rate
# cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  

# use this dataloader instead of the default

# cfg.DATALOADER.TRAIN.TOTAL_BATCH_SIZE = 64

# print(cfg.SOLVER.MAX_ITER)
data_length = 106466
iterations_for_one_epoch = data_length / cfg.SOLVER.IMS_PER_BATCH
# cfg.SOLVER.MAX_ITER = 1
cfg.SOLVER.CHECKPOINT_PERIOD = 500
cfg.SOLVER.MAX_ITER = int(iterations_for_one_epoch * 100)
cfg.TEST.EVAL_PERIOD = 1
print(cfg.SOLVER.MAX_ITER)
print(cfg.TEST.EVAL_PERIOD)
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
# trainer = Trainer(cfg)
trainer = DefaultTrainer(cfg)

trainer.resume_or_load(resume=False)
trainer.train()