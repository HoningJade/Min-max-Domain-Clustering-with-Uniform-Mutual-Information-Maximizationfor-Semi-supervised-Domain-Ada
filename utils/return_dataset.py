import os
import torch
from torchvision import transforms
from loaders.data_list import Imagelists_VISDA, return_classlist
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from collections import Counter
import numpy as np
from tqdm import tqdm


class ResizeImage():
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))


def return_uniform_sampler(label_list, num_class=126):
    """
        Return a inverse frequency weighted(therefore uniform)
        data sampler based on given datalist
    """
    count_dict = Counter(label_list)
    count_dict_full = {k: 0 for k in range(num_class)}
    for k, v in count_dict.items():
        count_dict_full[k] = v
    count_dict_sorted = {k: v for k, v in sorted(
        count_dict_full.items(), key=lambda item: item[0])}
    class_sample_count = np.array(list(count_dict_sorted.values()))
    class_sample_count = class_sample_count / class_sample_count.max()
    class_sample_count += 1e-8
    weights = 1 / torch.Tensor(class_sample_count)

    sample_weights = [weights[l] for l in label_list]
    sample_weights = torch.DoubleTensor(np.array(sample_weights))
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    return sampler


def return_pred(G, F1, target_loader_unl_random):
    G.eval()
    F1.eval()

    preds = []
    with torch.no_grad():
        for img, _ in tqdm(target_loader_unl_random):
            img = img.cuda()
            out = F1(G(img))
            preds.append(torch.argmax(out, dim=1).squeeze())
        preds = torch.cat(preds, dim=0)
    sampler = return_uniform_sampler(preds)
    return preds, sampler


def return_src_dist(args):
    base_path = './data/txt/%s' % args.dataset
    image_set_file_s = \
        os.path.join(base_path,
                     'labeled_source_images_' +
                     args.source + '.txt')
    class_cnt = {}
    with open(image_set_file_s) as f:
        label_list = []
        for ind, x in enumerate(f.readlines()):
            label = int(x.split(' ')[-1])
            if label not in class_cnt.keys():
                class_cnt[label] = 0
            class_cnt[label] += 1
    src_dist = np.zeros(len(class_cnt))
    total = sum(list(class_cnt.values()))
    for k, v in class_cnt.items():
        src_dist[k] = v / total
    return src_dist


def return_dataset(args):
    base_path = './data/txt/%s' % args.dataset
    root = './data/%s/' % args.dataset
    image_set_file_s = \
        os.path.join(base_path,
                     'labeled_source_images_' +
                     args.source + '.txt')
    image_set_file_t = \
        os.path.join(base_path,
                     'labeled_target_images_' +
                     args.target + '_%d.txt' % (args.num))
    image_set_file_t_val = \
        os.path.join(base_path,
                     'validation_target_images_' +
                     args.target + '_3.txt')
    image_set_file_unl = \
        os.path.join(base_path,
                     'unlabeled_target_images_' +
                     args.target + '_%d.txt' % (args.num))

    if args.net == 'alexnet':
        crop_size = 227
    else:
        crop_size = 224
    data_transforms = {
        'train': transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            ResizeImage(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    source_dataset = Imagelists_VISDA(image_set_file_s, root=root,
                                      transform=data_transforms['train'])
    target_dataset = Imagelists_VISDA(image_set_file_t, root=root,
                                      transform=data_transforms['val'])
    target_dataset_val = Imagelists_VISDA(image_set_file_t_val, root=root,
                                          transform=data_transforms['val'])
    target_dataset_unl = Imagelists_VISDA(image_set_file_unl, root=root,
                                          transform=data_transforms['val'])
    target_dataset_test = Imagelists_VISDA(image_set_file_unl, root=root,
                                           transform=data_transforms['test'])
    class_list = return_classlist(image_set_file_s)
    print("%d classes in this dataset" % len(class_list))

    if args.uniform_sampling:
        source_sampler = return_uniform_sampler(source_dataset.labels)
        target_sampler = return_uniform_sampler(target_dataset.labels)
    else:
        # equivalent to shuffle=True
        source_sampler = SubsetRandomSampler(np.arange(len(source_dataset.imgs)))
        target_sampler = SubsetRandomSampler(np.arange(len(target_dataset.imgs)))

    # sampler is contradict to shuffle, so set shuffle to False
    source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=args.bs,
                                                num_workers=3, shuffle=False,
                                                drop_last=True,
                                                sampler=source_sampler)
    target_loader = \
        torch.utils.data.DataLoader(target_dataset,
                                    batch_size=min(args.bs, len(target_dataset)),
                                    num_workers=3, shuffle=False, 
                                    drop_last=True,
                                    sampler=target_sampler)
    target_loader_val = \
        torch.utils.data.DataLoader(target_dataset_val,
                                    batch_size=min(args.bs,
                                                   len(target_dataset_val)),
                                    num_workers=3,
                                    shuffle=True, drop_last=True)
    target_loader_unl = \
        torch.utils.data.DataLoader(target_dataset_unl,
                                    batch_size=args.bs * 2, num_workers=3,
                                    shuffle=True, drop_last=True)
    target_loader_test = \
        torch.utils.data.DataLoader(target_dataset_test,
                                    batch_size=args.bs * 2, num_workers=3,
                                    shuffle=True, drop_last=True)
    return source_loader, target_loader, target_loader_unl, \
        target_loader_val, target_loader_test, class_list, target_dataset_unl


def return_dataset_test(args):
    base_path = './data/txt/%s' % args.dataset
    root = './data/%s/' % args.dataset
    image_set_file_s = os.path.join(base_path, args.source + '_all' + '.txt')
    image_set_file_test = os.path.join(base_path,
                                       'unlabeled_target_images_' +
                                       args.target + '_%d.txt' % (args.num))
    if args.net == 'alexnet':
        crop_size = 227
    else:
        crop_size = 224
    data_transforms = {
        'test': transforms.Compose([
            ResizeImage(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    target_dataset_unl = Imagelists_VISDA(image_set_file_test, root=root,
                                          transform=data_transforms['test'],
                                          test=True)
    class_list = return_classlist(image_set_file_s)
    print("%d classes in this dataset" % len(class_list))
    target_loader_unl = \
        torch.utils.data.DataLoader(target_dataset_unl,
                                    batch_size=args.bs * 2, num_workers=3,
                                    shuffle=False, drop_last=False)
    return target_loader_unl, class_list
