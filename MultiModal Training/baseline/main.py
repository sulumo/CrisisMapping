"""
Multitask CrisisMMD Training Script
Supports CLIP_CrisisKAN with 3 classifier heads.
"""

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import logging
import time
import random
import math
import numpy as np
import torch
import torch.nn as nn
from optimization import WarmupLinearSchedule, AdamW

from args import get_args
from trainer import Trainer
from crisismmd_dataset import CrisisMMDataset
from models_clip import CLIP_CrisisKAN

from torch.utils.data import DataLoader
from tqdm import tqdm


##############################################################
#  UTILITY: REPRODUCIBILITY
##############################################################
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


##############################################################
#  MAIN
##############################################################
if __name__ == '__main__':
    opt = get_args()
    set_seed(opt.seed if hasattr(opt, 'seed') else 42)

    device = opt.device
    num_workers = opt.num_workers
    batch_size = opt.batch_size
    learning_rate = opt.learning_rate
    NUM_EPOCHS = opt.num_epochs
    EVAL = opt.eval
    USE_TENSORBOARD = opt.use_tensorboard


    weight_decay = opt.weight_decay
    warmup_ratio = opt.warmup_ratio



    SAVE_DIR = opt.save_dir
    MODEL_NAME = opt.model_name if opt.model_name else str(int(time.time()))
    save_dir = os.path.join(SAVE_DIR, MODEL_NAME)
    os.makedirs(save_dir, exist_ok=True)

    # ----------------------------------------------------------
    # Logging
    # ----------------------------------------------------------
    logging.basicConfig(
        filename=os.path.join(save_dir, f'log_{int(time.time())}.txt'),
        level=logging.INFO
    )
    logging.info(str(opt))

    ##############################################################
    #   DATASETS (Multitask merged TSV already prepared)
    ##############################################################
    if not EVAL:
        train_set = CrisisMMDataset()
        train_set.initialize(opt, phase='train')   # task ignored now
        train_loader = DataLoader(train_set, batch_size=batch_size,
                                  shuffle=True, num_workers=num_workers)
    else:
        train_loader = None

    dev_set = CrisisMMDataset()
    dev_set.initialize(opt, phase='dev')  # task irrelevant
    dev_loader = DataLoader(dev_set, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)

    test_set = CrisisMMDataset()
    test_set.initialize(opt, phase='test')  # task irrelevant
    test_loader = DataLoader(test_set, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)


    ##############################################################
    #   MODEL (3-head multitask)
    ##############################################################
    model = CLIP_CrisisKAN(save_dir=save_dir)
    model = nn.DataParallel(model)
    model.to(device)

    ##############################################################
    #   OPTIMIZER/SCHEDULER
    ##############################################################

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay
    )


    if not EVAL:
        t_total = NUM_EPOCHS * len(train_loader)
        warmup_steps = int(t_total * warmup_ratio)

        logging.info(
            f"NUM_EPOCHS={NUM_EPOCHS}, "
            f"len(train_loader)={len(train_loader)}, "
            f"t_total={t_total}, "
            f"warmup_ratio={warmup_ratio}"
        )

        scheduler = WarmupLinearSchedule(
            optimizer,
            warmup_steps=warmup_steps,
            t_total=t_total
        )
    else:
        scheduler = None




    ##############################################################
    #   TRAINER
    ##############################################################
    trainer = Trainer(
        train_loader=train_loader,
        dev_loader=dev_loader,
        test_loader=test_loader,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        save_dir=save_dir,
        device=device,
        tensorboard=USE_TENSORBOARD,
        task_weights=None  # default class-inverse proportional weights
    )

    ##############################################################
    #   LOAD PRETRAINED MODEL IF GIVEN
    ##############################################################
    if opt.model_to_load:
        logging.info("Loading pretrained model: " + opt.model_to_load)
        model.module.load(opt.model_to_load)

    ##############################################################
    #   TRAIN + EVALUATE
    ##############################################################
    if not EVAL:
        logging.info("\n================ Training Start ================\n")
        trainer.train(num_epochs=NUM_EPOCHS)

        model.module.load_best()
        trainer.validate()
        trainer.predict()

    ##############################################################
    #   ONLY EVALUATE
    ##############################################################
    else:
        logging.info("\n================ Evaluating ================\n")
        model.module.load_best()
        trainer.validate()
        trainer.predict()

