# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import datetime
import json
import math
import os
import signal
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce, rearrange, repeat
from torch.distributions import Categorical
from torch.nn.utils.rnn import pad_sequence
from torchvision import models as torchvision_models

from croc_utils.data_transforms import *
from croc_utils.datasets import get_dataloader, get_dataset
from croc_utils.hpc import signal_handler, pin_workers_iterator
from models import vision_transformer as vits
from models.vision_transformer import CrOCHead
from utils import utils

torchvision_archs = sorted(name for name in torchvision_models.__dict__
                           if name.islower() and not name.startswith("__")
                           and callable(torchvision_models.__dict__[name]))


from croc_utils.parser import get_args_parser


def train_croc(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing data ... ============
    transform = DataAugmentationCrOC(
        global_crops_scale=args.global_crops_scale,
    )

    dataset = get_dataset(args, transform, val_or_train='train')
    data_loader = get_dataloader(args, dataset)
    signal.signal(signal.SIGTERM, partial(signal_handler, args))

    print(f"Data loaded: there are {len(dataset)} images.")

    # ============ building student and teacher networks ... ============
    # we changed the name DeiT-S for ViT-S to avoid confusions
    args.arch = args.arch.replace("deit", "vit")
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch in vits.__dict__.keys():
        student = vits.__dict__[args.arch](patch_size=args.patch_size, drop_path_rate=args.drop_path_rate,
                                           is_teacher=False)
        teacher = vits.__dict__[args.arch](patch_size=args.patch_size)
        embed_dim = student.embed_dim
    else:
        print(f"Unknow architecture: {args.arch}")

    # multi-crop wrapper handles forward with inputs of different resolutions
    student = utils.MultiCropWrapperCrOC(
        student,
        CrOCHead(embed_dim, args.out_dim, args.out_dim_c, use_bn=args.use_bn_in_head,
                 norm_last_layer=args.norm_last_layer),
        args.which_features,
    )
    teacher = utils.MultiCropWrapperCrOC(
        teacher,
        CrOCHead(embed_dim, args.out_dim, args.out_dim_c, args.use_bn_in_head),
        args.which_features,
    )
    clustering = Clustering(
        args,
        n_tokens=args.n_tokens,
        sinkhorn_lambda=args.sinkhorn_lambda,
        sinkhorn_iterations=args.sinkhorn_iterations,
        student_temp=args.student_temp,
        uniform_marginals=args.uniform_marginals,
        marginals_temp_r=args.marginals_temp_r,
        marginals_temp_c=args.marginals_temp_c,
        pos_alpha=args.pos_alpha,
        n_heads=teacher.backbone.num_heads,
    )

    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()

    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu], find_unused_parameters=False,
                                                  broadcast_buffers=False)

    # teacher and student start with the same weights
    msg = teacher_without_ddp.load_state_dict(student.module.state_dict(), strict=False)
    print('Teacher loaded with msg: {}'.format(msg))

    # there is no backpropagation through the teacher, so no need for gradients, except for the sampler
    for n, p in teacher.named_parameters():
        if 'sampler' not in n:
            p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    # ============ preparing losses ... ============
    croc_loss = CrOCLoss(
        out_dim=args.out_dim,
        out_dim_c=args.out_dim_c,
        ncrops=args.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
        warmup_teacher_temp=args.warmup_teacher_temp,
        warmup_teacher_temp_c=args.warmup_teacher_temp_c,
        teacher_temp=args.teacher_temp,
        teacher_temp_c=args.teacher_temp_c,
        warmup_teacher_temp_epochs=args.warmup_teacher_temp_epochs,
        nepochs=args.epochs,
        student_temp_c=args.student_temp_c
    ).cuda()

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )

    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1, args.epochs, len(data_loader))
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        croc_loss=croc_loss,
    )
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting DINO training !")
    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)

        # ============ training one epoch of DINO ... ============
        train_stats = train_one_epoch(student, teacher, teacher_without_ddp, croc_loss, data_loader, optimizer,
                                      lr_schedule, wd_schedule, momentum_schedule, args.alpha_s, epoch, fp16_scaler,
                                      clustering, args)

        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'croc_loss': croc_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    # clean_on_leave(args)


def train_one_epoch(student, teacher, teacher_without_ddp, croc_loss, data_loader, optimizer, lr_schedule, wd_schedule,
                    momentum_schedule, alpha_s, epoch, fp16_scaler, clustering, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)

    the_iterator = iter(data_loader)
    pin_workers_iterator(the_iterator, args)

    for it, (images, _) in enumerate(metric_logger.log_every(the_iterator, 10, header)):
        # update weight decay and learning rate according to their schedule
        crop_pos = None
        if isinstance(images[0], list):
            images, crop_pos = images
            crop_pos = [p.cuda(non_blocking=True) for p in crop_pos]

        it_ = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it_]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it_]

        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in images]

        with torch.cuda.amp.autocast(fp16_scaler is not None):
            # only the 2 global views pass through the teacher
            with torch.no_grad():
                # Get the teacher tokens
                teacher_output, teacher_last_tokens, teacher_qkv_tokens = teacher(images[:2])

                # Compute the teacher's assignments and centroids
                teacher_centroids, valid_centroids, assignments, region_count = clustering.\
                    compute_teacher_centroids(teacher_last_tokens, teacher_qkv_tokens, crop_pos)

            # Get the student tokens
            student_output, student_last_tokens, _ = student(images[2:])

            # Compute the student's assignments and centroids
            student_centroids = clustering.compute_student_centroids(assignments, student_last_tokens,
                                                                     valid_centroids)

            # Project the centroids
            with torch.no_grad():
                teacher_centroids = teacher(torch.cat(teacher_centroids), head_only=True)
            student_centroids = student(torch.cat(student_centroids), head_only=True)

            # Get the [CLS] loss
            d_loss, s_loss = croc_loss(student_output, teacher_output, epoch, student_centroids, teacher_centroids)

            # Combine the losses
            loss = d_loss + alpha_s * s_loss

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # student update
        param_norms = None
        if fp16_scaler is None:
            # Back-propagate the loss
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
            optimizer.step()
            optimizer.zero_grad()
        else:
            # Back-propagate the loss
            fp16_scaler.scale(loss).backward()

            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()
            optimizer.zero_grad()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it_]  # momentum parameter
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(d_loss=d_loss.item())
        metric_logger.update(ent=croc_loss.teacher_entropy.item())
        metric_logger.update(kl=croc_loss.kl_div.item())
        metric_logger.update(s_loss=s_loss.item())
        # metric_logger.update(r_cnt=region_count)
        metric_logger.update(ent_c=croc_loss.teacher_entropy_c.item())
        metric_logger.update(kl_c=croc_loss.kl_div_c.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class CrOCLoss(nn.Module):
    def __init__(self, out_dim, out_dim_c, ncrops, warmup_teacher_temp, warmup_teacher_temp_c, teacher_temp, teacher_temp_c,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1, student_temp_c=0.1, center_momentum=0.9, center_momentum_c=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.student_temp_c = student_temp_c
        self.center_momentum = center_momentum
        self.center_momentum_c = center_momentum_c
        self.ncrops = ncrops
        self.centroids_counter = torch.tensor(0, device='cuda')
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.register_buffer("center_c", torch.zeros(1, out_dim_c))
        # we apply a warm-up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))
        self.teacher_temp_schedule_c = np.concatenate((
            np.linspace(warmup_teacher_temp_c, teacher_temp_c, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp_c
        ))

        # Log metrics
        self.teacher_entropy = torch.tensor(0, device='cuda')
        self.teacher_entropy_c = torch.tensor(0, device='cuda')
        self.kl_div = torch.tensor(0, device='cuda')
        self.kl_div_c = torch.tensor(0, device='cuda')

    @torch.no_grad()
    def compute_metrics_dino(self, teacher_out, student_out):
        # Compute the teacher's entropy
        self.teacher_entropy = Categorical(probs=teacher_out).entropy().mean()
        dist.all_reduce(self.teacher_entropy)
        self.teacher_entropy = self.teacher_entropy / dist.get_world_size()

        # Compute the KL divergence
        self.kl_div = -torch.nn.KLDivLoss(reduction='batchmean')(student_out, teacher_out)

    @torch.no_grad()
    def compute_metrics_croc(self, student_cent, teacher_cent):
        # Compute the teacher's entropy
        self.teacher_entropy_c = Categorical(probs=teacher_cent).entropy().sum()
        dist.all_reduce(self.teacher_entropy_c)
        self.teacher_entropy_c = self.teacher_entropy_c / self.centroids_counter

        # Compute the KL divergence
        student_cent_v1, student_cent_v2 = student_cent.chunk(2)
        teacher_cent_v1, teacher_cent_v2 = teacher_cent.chunk(2)
        kl_div_1 = torch.nn.KLDivLoss(reduction='batchmean')(student_cent_v1, teacher_cent_v2)
        kl_div_2 = torch.nn.KLDivLoss(reduction='batchmean')(student_cent_v2, teacher_cent_v1)
        self.kl_div_c = -(kl_div_1 + kl_div_2) / 2.

    def forward(self, student_output, teacher_output, epoch, student_centroids=None, teacher_centroids=None):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms

        # Compute the loss on the centroids if provided
        loss_c = torch.tensor(0.).to(student_output.device)
        if student_centroids is not None and teacher_centroids is not None:
            # Sharpen the student predictions
            student_cent = student_centroids / self.student_temp_c

            # Teacher centering and sharpening
            temp = self.teacher_temp_schedule_c[epoch]
            teacher_cent = F.softmax((teacher_centroids - self.center_c) / temp, dim=-1)

            # Split the centroids view-wise
            student_cent_v1, student_cent_v2 = student_cent.chunk(2)
            teacher_cent_v1, teacher_cent_v2 = teacher_cent.chunk(2)

            # Compute the loss
            loss_c += torch.sum(-teacher_cent_v1 * F.log_softmax(student_cent_v2, dim=-1), dim=-1).mean()
            loss_c += torch.sum(-teacher_cent_v2 * F.log_softmax(student_cent_v1, dim=-1), dim=-1).mean()
            loss_c /= 2.

        # Update the centers
        self.update_center(teacher_output, teacher_centroids)

        # Update the metrics
        self.compute_metrics_dino(torch.cat(teacher_out), torch.cat(student_out))
        if student_centroids is not None and teacher_centroids is not None:
            self.compute_metrics_croc(student_cent, teacher_cent)
        return total_loss, loss_c

    @torch.no_grad()
    def update_center(self, teacher_output, teacher_centroids=None):
        """
        Update center used for teacher output.
        """
        # Image-level
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)

        # Update
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

        # Centroids-level
        if teacher_centroids is not None:
            batch_center_c = torch.sum(teacher_centroids, dim=0, keepdim=True)
            self.centroids_counter = torch.tensor(len(teacher_centroids), device='cuda')

            # Update
            dist.all_reduce(batch_center_c)
            dist.all_reduce(self.centroids_counter)
            batch_center_c = batch_center_c / self.centroids_counter
            self.center_c = self.center_c * self.center_momentum_c + batch_center_c * (1 - self.center_momentum_c)


class DataAugmentationCrOC(object):
    def __init__(self, global_crops_scale):
        color_jitter = transforms.Compose([
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # Spatial transformation
        self.spatial_transfo = MyCompose([
            RandomResizedCropWithPos(224, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC),
            MyComposeInner([RandomHorizontalFlipWithFlipBool(p=0.5)]),
        ])

        # Color transformations
        self.color_transfo1 = transforms.Compose([
            color_jitter,
            utils.GaussianBlur(1.0),
            normalize,
        ])
        self.color_transfo2 = transforms.Compose([
            color_jitter,
            utils.GaussianBlur(0.1),
            utils.Solarization(0.2),
            normalize,
        ])

    def __call__(self, image):
        # Apply the spatial transformations
        view_1_, pos_1 = self.spatial_transfo(image)
        view_2_, pos_2 = self.spatial_transfo(image)

        # The teacher's views share the same color transformation
        view_1, view_2 = self.color_transfo1(view_1_), self.color_transfo2(view_2_)
        crops = 2 * [view_1, view_2]
        crops_pos = 2 * [pos_1, pos_2]
        return crops, crops_pos


class Clustering:
    def __init__(self, args, n_tokens, sinkhorn_lambda, sinkhorn_iterations=5, student_temp=1., uniform_marginals=True,
                 marginals_temp_r=2.0, marginals_temp_c=2.0, pos_alpha=4., n_heads=6):
        self.patch_size = args.patch_size
        self.n_tokens = n_tokens
        self.student_temp = student_temp
        self.pos_alpha = pos_alpha
        self.sinkhorn_lambda = sinkhorn_lambda
        self.sinkhorn_iterations = sinkhorn_iterations
        self.uniform_marginals = uniform_marginals
        self.marginals_temp_r = marginals_temp_r
        self.marginals_temp_c = marginals_temp_c
        self.n_heads = n_heads
        self.n_centroids_max = args.n_centroids_max

    @torch.no_grad()
    def sinkhorn(self, M, r, c):
        P = torch.exp(- self.sinkhorn_lambda * M)
        P /= reduce(P, 'b n k -> b 1 1', reduction='sum')

        # Iterate over the sinkhorn algorithm
        for _ in range(self.sinkhorn_iterations):
            u = reduce(P, 'b n k -> b n 1', reduction='sum')
            P *= (r / u)
            u = reduce(P, 'b n k -> b 1 k', reduction='sum')
            P *= (c / u)
        P = torch.nan_to_num(P, nan=1e-8)
        return P, torch.sum(P * M, dim=[1, 2])

    def compute_r_marginals(self, teacher_tokens, teacher_cls_tokens, uniform_marginals):
        m, b, n, d = teacher_tokens.shape
        if uniform_marginals:
            n = m * n
            r = (torch.ones([b, n]) / n).cuda()
        else:
            r = torch.einsum('m b n d, m b d -> m b n', teacher_tokens, teacher_cls_tokens)
            r = rearrange(r, 'm b n -> b (m n)')
        return r

    def compute_c_marginals(self, r, indices, uniform_marginals, temp_r=1., temp_c=1.):
        b, n = r.shape
        k = indices.shape[1]
        if uniform_marginals:
            c = (torch.ones([b, 1, k]) / k).cuda()
            r = r[:, :, None]
        else:
            c = torch.einsum('b n, b k n -> b k', r, indices)
            c = rearrange(c / temp_c, 'b k -> b 1 k')
            c = F.softmax(c, dim=-1)
            r = rearrange(r, 'b n -> b n 1')
            r = F.softmax(r / temp_r, dim=1)
        return torch.nan_to_num(r), torch.nan_to_num(c)

    def compute_student_centroids(self, assignments, student_last_tokens, valid_centroids):
        # Compute the average norm of the centroids
        cls_norm = student_last_tokens[:, 0, :].norm(dim=-1).mean()

        # Normalize the tokens
        student_last_tokens = rearrange(student_last_tokens[:, 1:], '(m b) n d -> m b n d', m=2)

        # Compute the student centroids using the teacher assignments
        student_last_tokens = repeat(student_last_tokens, 'm b n d -> m (b h) n d', h=self.n_heads)
        centroids = torch.einsum('m b n d, m b n k -> m b k d', student_last_tokens, assignments)
        centroids = rearrange(centroids, 'm b k d -> m (b k) d')

        # Split the centroids view-wise
        centroids_v1, centroids_v2 = centroids.unbind()
        centroids_v1, centroids_v2 = centroids_v1[valid_centroids], centroids_v2[valid_centroids]

        # Re-normalize the centroids
        centroids_v1 = cls_norm * F.normalize(centroids_v1, dim=-1)
        centroids_v2 = cls_norm * F.normalize(centroids_v2, dim=-1)
        return centroids_v1, centroids_v2

    def compute_teacher_centroids(self, teacher_last_tokens, teacher_qkv_tokens, crop_pos):
        # Compute the average norm of the centroids
        cls_norm = teacher_last_tokens[:, 0, :].norm(dim=-1).mean()

        # Obtain the joint representation of the teacher tokens
        teacher_qkv_tokens = rearrange(teacher_qkv_tokens, '(m b) n d -> m b n d', m=2)
        teacher_last_tokens = rearrange(teacher_last_tokens[:, 1:], '(m b) n d -> m b n d', m=2)
        teacher_last_tokens = repeat(teacher_last_tokens, 'm b n d -> m (b h) n d', h=self.n_heads)

        # Normalize the tokens
        teacher_qkv_tokens = F.normalize(teacher_qkv_tokens, dim=-1)
        teacher_cls_tokens, teacher_qkv_tokens = teacher_qkv_tokens[:, :,  0], teacher_qkv_tokens[:, :, 1:]

        # Compute the random distribution
        r = self.compute_r_marginals(teacher_qkv_tokens, teacher_cls_tokens, uniform_marginals=self.uniform_marginals)
        p = F.softmax(r / self.marginals_temp_r, dim=1)
        teacher_qkv_tokens = rearrange(teacher_qkv_tokens, 'm b n d -> b (m n) d', m=2)
        b, n, _ = teacher_qkv_tokens.shape

        # Get the indices as one-hot
        indices = p.multinomial(num_samples=self.n_tokens, replacement=False)
        indices = rearrange(indices, 'b k -> (b k)')
        indices = torch.eye(n).to(teacher_qkv_tokens.device)[indices].to(teacher_qkv_tokens.device)
        indices = rearrange(indices, '(b k) n -> b k n', b=b)

        # --------------------------------------------------------------------------------
        # Compute the initial marginals
        n_regions = self.n_tokens
        tokens_marginals, centroids_marginals = self.compute_c_marginals(
            r, indices, uniform_marginals=self.uniform_marginals, temp_r=self.marginals_temp_r,
            temp_c=self.marginals_temp_c
        )

        ############################################## POS ENC 1 START #####################################################
        # Patchify positional encodings
        positions = rearrange(torch.stack(crop_pos[:2]), 'm b d (r i) (c j) -> m b d (r c) (i j)', i=self.patch_size,
                              j=self.patch_size).mean(dim=-1)
        positions = rearrange(positions, "m b d n -> b (m n) d")

        # Compute the query positions using the same sampling indices
        positions = repeat(positions, 'b n d -> (b h) n d', h=self.n_heads)
        centroids_positions = torch.einsum('b n d, b k n -> b k d', positions, indices)

        # Compute the distance matrix
        distances_p = torch.sqrt(torch.sum((centroids_positions[:, None, :, :] - positions[:, :, None, :]) ** 2, dim=-1))
        distances_p /= distances_p.amax(dim=(-1, -2))[:, None, None]
        distance_normalized_p = distances_p

        # Set the initial centroids
        centroids = torch.einsum('b n d, b k n -> b k d', teacher_qkv_tokens, indices)

        # Iterate until the number of regions is 2
        costs, assignments_full, indices_full = [], [], []
        while (n_regions >= 2):
            centroids = F.normalize(centroids, dim=-1)
            assignments = torch.einsum('b n d, b k d -> b n k', teacher_qkv_tokens, centroids)
            b, n, k = assignments.shape

            # Get the cost
            M = - assignments + self.pos_alpha * distance_normalized_p
            M = (M - M.min()) / (M.max() - M.min())

            # Compute the transportation plan and the distance
            assignments, transportation_cost = self.sinkhorn(
                M=M,
                r=tokens_marginals,
                c=centroids_marginals,
            )

            # Compute the current clustering cost
            cost = transportation_cost

            # Store the assignments normalized column-wise and view-wise
            costs.append(cost)
            assignments_ = rearrange(assignments, 'b (m n) k -> m b n k', m=2)
            assignment_v1, assignment_v2 = assignments_.unbind()
            assignment_v1 = assignment_v1 / assignment_v1.sum(dim=-2, keepdim=True)
            assignment_v2 = assignment_v2 / assignment_v2.sum(dim=-2, keepdim=True)
            assignments_ = torch.cat([assignment_v1, assignment_v2], dim=1)
            assignments_full.append(assignments_)
            indices_full.append(indices)

            if n_regions == 2:
                break

            # Update the centroids
            normalized_assignments = assignments / assignments.sum(dim=1, keepdim=True)
            centroids = torch.einsum('b n d, b n k -> b k d', teacher_qkv_tokens, normalized_assignments)

            # Find the centroids to merge
            centroids_similarity = torch.einsum('b i d, b j d -> b i j', centroids, centroids)
            centroids_similarity -= 2 * torch.eye(n_regions).to(centroids_similarity.device)
            centroids_similarity = rearrange(centroids_similarity, 'b i j -> b (i j)')
            merge_index = torch.argmax(centroids_similarity, dim=-1)
            i = torch.div(merge_index, n_regions, rounding_mode='floor')
            j = merge_index % n_regions

            # Find the representative for the new centroid
            b_indices = torch.arange(0, b).to(i.device)
            b_k_indices_i = tuple(torch.stack([b_indices, i]).tolist())
            b_k_indices_j = tuple(torch.stack([b_indices, j]).tolist())
            new_indices = (indices[b_k_indices_i] + indices[b_k_indices_j])[:, None, :]

            # Compute the new indices
            kept_indices = repeat(torch.arange(0, k), 'k -> b k', b=b).to(i.device)
            i_repeat = repeat(i, 'b -> b k', k=k).to(i.device)
            j_repeat = repeat(j, 'b -> b k', k=k).to(i.device)
            kept_indices = torch.logical_and(kept_indices != i_repeat, kept_indices != j_repeat)
            indices = rearrange(indices[kept_indices], '(b k) n -> b k n', b=b)
            indices = torch.cat([indices, new_indices], dim=1)

            # Merge the assignments
            assignments = rearrange(assignments, 'b n k -> b k n')
            new_assignments = (assignments[b_k_indices_i] + assignments[b_k_indices_j])[:, None, :]
            kept_assignments = rearrange(assignments[kept_indices], '(b k) n -> b k n', b=b)
            assignments = torch.cat([kept_assignments, new_assignments], dim=1)
            assignments = rearrange(assignments, 'b k n -> b n k')

            # Update the centroids
            normalized_assignments = assignments / assignments.sum(dim=1, keepdim=True)
            centroids = torch.einsum('b n d, b n k -> b k d', teacher_qkv_tokens, normalized_assignments)

            # Update the positions distance cost
            centroids_positions = torch.einsum('b n d, b n k -> b k d', positions, normalized_assignments)

            # Compute the distance matrix
            distances_p = torch.sqrt(
                torch.sum((centroids_positions[:, None, :, :] - positions[:, :, None, :]) ** 2, dim=-1))
            distances_p /= distances_p.amax(dim=(-1, -2))[:, None, None]
            distance_normalized_p = distances_p

            # Update the number of regions
            n_regions -= 1

            # Compute the new marginals
            centroids_marginals = torch.einsum('b n, b n k -> b k', tokens_marginals.squeeze(),
                                               normalized_assignments).unsqueeze(1)
            centroids_marginals /= centroids_marginals.sum(dim=-1, keepdim=True)

        # Stack the costs
        costs = torch.stack(costs, dim=-1)

        # Retrieve the best assignments
        stop_index = - (self.n_centroids_max - 1)
        costs = costs[:, stop_index:]
        optimal_ks = costs.argmin(dim=-1)
        optimal_ks = optimal_ks + (self.n_tokens - self.n_centroids_max)

        assignments = [assignments_full[j][i].transpose(0, 1) for i, j in enumerate(optimal_ks.tolist())]
        assignments = pad_sequence(assignments, batch_first=True).transpose(1, 2)

        # ============================ Split the cluster view-wise ===========================================
        # Each token belongs to a single cluster
        hard_assignments = torch.max(assignments, dim=-1, keepdim=True).values
        hard_assignments = repeat(hard_assignments, 'b n 1 -> b n k', k=assignments.shape[-1])
        hard_assignments = (assignments == hard_assignments).float()
        masked_assignments = assignments * hard_assignments

        # Split the assignments view-wise
        masked_assignments = rearrange(masked_assignments, 'b (m n) k -> m b n k', m=2)
        assignments = masked_assignments

        # Compute the centroids of each view and normalize the assignments
        centroids = torch.einsum('m b n d, m b n k -> m b k d', teacher_last_tokens, assignments)
        centroids = rearrange(centroids, 'm b k d -> m (b k) d')
        centroids_v1, centroids_v2 = centroids.unbind()

        # Discard a cluster if it's empty in either view
        masked_assignments_v1, masked_assignments_v2 = rearrange(masked_assignments, 'm b n k -> m (b k) n').unbind()
        valid_centroids_v1 = set(masked_assignments_v1.sum(dim=-1).nonzero().squeeze().tolist())
        valid_centroids_v2 = set(masked_assignments_v2.sum(dim=-1).nonzero().squeeze().tolist())
        valid_centroids = list(valid_centroids_v1.intersection(valid_centroids_v2))
        centroids_v1, centroids_v2 = centroids_v1[valid_centroids], centroids_v2[valid_centroids]

        # # The centroids must have approximately the same norm as the [CLS] tokens
        centroids_v1 = cls_norm * F.normalize(centroids_v1, dim=-1)
        centroids_v2 = cls_norm * F.normalize(centroids_v2, dim=-1)

        # Count the average number of regions
        region_count = self.n_tokens - optimal_ks.float().mean().item()
        return (centroids_v1, centroids_v2), valid_centroids, assignments, region_count


if __name__ == '__main__':
    parser = argparse.ArgumentParser('CrOC', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.epochs <= 2:
        args.warmup_teacher_temp_epochs = 0
        args.warmup_epochs = 0
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_croc(args)
