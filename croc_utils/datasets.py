import os
import tarfile
import time
from functools import partial

import torch
import torchvision.datasets
from torchvision.datasets.coco import CocoDetection


def get_tensor_binary_instance_labels(label, nb_categories=91):
    tmp = torch.zeros(nb_categories)
    for obj in label:
        tmp[obj["category_id"]] = 1
    return tmp


def collate_fn(data):
    images, targets = zip(*data)
    if isinstance(images, tuple):
        images, crop_pos = zip(*images)
    if isinstance(images[0], list):
        images = [torch.stack(b) for b in zip(*images)]
        crop_pos = [torch.stack(b) for b in zip(*crop_pos)]
    elif isinstance(images[0], torch.Tensor):
        images = torch.stack(images)
        crop_pos = torch.stack(crop_pos)
    else:
        raise NotImplemented

    if type(targets[0]) == torch.Tensor:
        targets = torch.stack(targets)
    return (images, crop_pos), targets


def untar_to_dst(args, src):
    assert (args.untar_path != "")

    if args.untar_path[0] == '$':
        args.untar_path = os.environ[args.untar_path[1:]]
    start_copy_time = time.time()

    if int(args.gpu) == 0:
        with tarfile.open(src, 'r') as f:
            f.extractall(args.untar_path)

        print('Time taken for untar:', time.time() - start_copy_time)

    try:
        torch.distributed.barrier()
        time.sleep(5)
    except RuntimeError:
        pass


def get_dataset(args, transform, target_transform=lambda x: x, val_or_train='train', wrapper=None):
    if len(args.untar_path) > 0 and args.untar_path[0] == '$':
        args.untar_path = os.environ[args.untar_path[1:]]

    if args.dataset_type == 'imagefolder':
        return torchvision.datasets.ImageFolder(args.data_path, transform=transform)

    elif args.dataset_type == 'imagenet1k':
        if args.imagenet1k_path.split('.')[-1] == 'tar':
            untar_to_dst(args, args.imagenet1k_path)
            root_dir = os.path.join(args.untar_path, args.imagenet1k_path.split('/')[-1].split('.')[0])
        else:
            root_dir = args.imagenet1k_path

        assert ('ILSVRC2012_img_train' in os.listdir(root_dir))
        assert ('ILSVRC2012_img_val' in os.listdir(root_dir))
        return torchvision.datasets.ImageFolder(os.path.join(root_dir, 'ILSVRC2012_img_{}'.format(val_or_train)),
                                                transform=transform)

    elif args.dataset_type == 'coco':
        if args.coco_path.split('.')[-1] == 'tar':
            print(args.gpu, 'NEED TO UNTAR COCO TO LOCATION')
            untar_to_dst(args, args.coco_path)
            root_dir = os.path.join(args.untar_path, args.coco_path.split('/')[-1].split('.')[0])
        else:
            print(args.gpu, 'NO TAR USED')
            root_dir = args.coco_path
        print(root_dir)
        print(os.listdir(root_dir))
        assert ('images' in os.listdir(root_dir))
        assert ('annotations' in os.listdir(root_dir))

        keyword_args = {
            "root": os.path.join(root_dir, 'images/{}2017'.format(val_or_train)),
            "annFile": os.path.join(root_dir, 'annotations/instances_{}2017.json'.format(val_or_train)),
            "transform": transform,
            "target_transform": target_transform
        }
        if wrapper:
            return wrapper(**keyword_args)

        return CocoDetection(**keyword_args)
    elif args.dataset_type == 'coco+':
        from datasets.dummy_dataset import DummyDataset
        print(args.gpu, 'NEED TO UNTAR COCO TO LOCATION')

        # Prepare the train query
        untar_to_dst(args, args.coco_path)
        root_dir = os.path.join(args.untar_path, args.coco_path.split('/')[-1].split('.')[0])
        query_train = os.path.join(root_dir, 'images/train2017/*.jpg')

        # Prepare the train query
        unlabeled_path = os.path.join(*(args.coco_path.split('/')[:-1] + ['unlabeled2017.tar']))
        unlabeled_path = '/' + unlabeled_path
        untar_to_dst(args, unlabeled_path)
        root_dir = os.path.join(args.untar_path, unlabeled_path.split('/')[-1].split('.')[0])
        query_unlabeled = os.path.join(root_dir, '*.jpg')
        queries = [query_train, query_unlabeled]
        return DummyDataset(image_queries=queries, transform=transform)
    else:
        raise NotImplemented


def get_dataloader(args, dataset, **kwargs):
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    keyword_args = {
        'sampler': sampler,
        'batch_size': args.batch_size_per_gpu,
        'num_workers': args.num_workers,
        'pin_memory': True
    }
    if 'drop_last' not in kwargs:
        keyword_args['drop_last'] = True
    if type(dataset) == torchvision.datasets.coco.CocoDetection:
        keyword_args['collate_fn'] = collate_fn
    return torch.utils.data.DataLoader(dataset, **keyword_args, **kwargs)
