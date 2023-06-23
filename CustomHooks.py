from detectron2.engine.hooks import HookBase
from detectron2.evaluation import inference_context
from detectron2.utils.logger import log_every_n_seconds
from detectron2.data import DatasetMapper, build_detection_test_loader
import detectron2.utils.comm as comm
import torch
import time
import datetime
import logging
import numpy as np
from detectron2.utils.visualizer import Visualizer
import random
import cv2
import wandb

class LossEvalHook(HookBase):
    def __init__(self, eval_period, model, data_loader):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader
        self.num_samples = 4
    
    def _do_loss_eval(self):
        # Copying inference_on_dataset from evaluator.py
        total = len(self._data_loader)
        num_warmup = min(5, total - 1)
            
        start_time = time.perf_counter()
        total_compute_time = 0
        losses = []
        for idx, inputs in enumerate(self._data_loader):            
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
            start_compute_time = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Loss on Validation  done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )
            loss_batch = self._get_loss(inputs)
            losses.append(loss_batch)
        mean_loss = np.mean(losses)
        self.trainer.storage.put_scalar('validation_loss', mean_loss)
        comm.synchronize()

        return losses
            
    def _get_loss(self, data):
        # How loss is calculated on train_loop 
        metrics_dict = self._model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        total_losses_reduced = sum(loss for loss in metrics_dict.values())
        return total_losses_reduced
        
        
    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            # evaluation
            # self._do_loss_eval()
            self._visualizion()
            

        self.trainer.storage.put_scalars(timetest=12)

    
    def _visualizion(self):
        # 로컬 머신에서만 시각화하기 위해 첫 번째 워커만 사용합니다.
        if comm.get_rank() > 0:
            return

        # 데이터셋과 메타데이터 가져오기
        dataset = self.trainer.data_loader.dataset

        # 무작위 샘플 선택
        random_indices = random.sample(range(len(dataset)), self.num_samples)
        samples = [dataset[idx] for idx in random_indices]

        # 이미지와 해당 GT 박스 시각화하여 Wandb에 로그 남기기
        for idx, sample in enumerate(samples):
            img = sample["image"].permute(1, 2, 0).numpy()
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            v = self.trainer.model.forward(sample["image"].unsqueeze(0).to(self.trainer.model.device))
            v = self.trainer.model.postprocess(v)[0]
            v = v.resize((img.shape[1], img.shape[0]))
            v = v.to("cpu")
            v = v.get_fields()["pred_boxes"].tensor.numpy()
            v = v.round().astype(int)
            for box in v:
                cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.imwrite("./images/{}.jpg".format(idx),img)
            # wandb.log({"visualization/sample_{}".format(idx): wandb.Image(img, caption="Sample {}".format(idx))})


