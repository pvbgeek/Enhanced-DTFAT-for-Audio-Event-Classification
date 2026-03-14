# -*- coding: utf-8 -*-
# @Time    : 6/11/21 12:57 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : run.py

import argparse
import os
import ast
import pickle
import sys
import time
import torch
from torch.utils.data import WeightedRandomSampler
basepath = os.path.dirname(os.path.dirname(sys.path[0]))
sys.path.append(basepath)

import numpy as np
from traintest import train, validate

print("I am process %s, running on %s: starting (%s)" % (os.getpid(), os.uname()[1], time.asctime()))

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data-train", type=str, default='', help="training data json")
parser.add_argument("--data-val", type=str, default='', help="validation data json")
parser.add_argument("--data-eval", type=str, default='', help="evaluation data json")
parser.add_argument("--label-csv", type=str, default='', help="csv with class labels")
parser.add_argument("--n_class", type=int, default=527, help="number of classes")
parser.add_argument("--model", type=str, default='ast', help="the model used")
parser.add_argument("--dataset", type=str, default="audioset", help="the dataset used")

parser.add_argument("--exp-dir", type=str, default="", help="directory to dump experiments")
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument("--optim", type=str, default="adam", help="training optimizer", choices=["sgd", "adam"])
parser.add_argument('-b', '--batch-size', default=12, type=int, metavar='N', help='mini-batch size')
parser.add_argument('-w', '--num-workers', default=32, type=int, metavar='NW', help='# of workers for dataloading (default: 32)')
parser.add_argument("--n-epochs", type=int, default=1, help="number of maximum training epochs")
parser.add_argument("--lr_patience", type=int, default=2, help="how many epoch to wait to reduce lr if mAP doesn't improve")

parser.add_argument("--n-print-steps", type=int, default=100, help="number of steps to print statistics")
parser.add_argument('--save_model', help='save the model or not', type=ast.literal_eval)

parser.add_argument('--freqm', help='frequency mask max length', type=int, default=0)
parser.add_argument('--timem', help='time mask max length', type=int, default=0)
parser.add_argument("--mixup", type=float, default=0, help="how many (0-1) samples need to be mixup during training")
parser.add_argument("--bal", type=str, default=None, help="use balanced sampling or not")
parser.add_argument("--fstride", type=int, default=10, help="soft split freq stride, overlap=patch_size-stride")
parser.add_argument("--tstride", type=int, default=10, help="soft split time stride, overlap=patch_size-stride")
parser.add_argument('--imagenet_pretrain', help='if use ImageNet pretrained audio spectrogram transformer model', type=ast.literal_eval, default='True')
parser.add_argument('--audioset_pretrain', help='if use ImageNet and audioset pretrained audio spectrogram transformer model', type=ast.literal_eval, default='False')

parser.add_argument("--dataset_mean", type=float, default=-4.2677393, help="the dataset spectrogram mean")
parser.add_argument("--dataset_std", type=float, default=4.5689974, help="the dataset spectrogram std")
parser.add_argument("--audio_length", type=int, default=1024, help="the dataset spectrogram std")
parser.add_argument('--noise', help='if augment noise', type=ast.literal_eval, default='False')

parser.add_argument("--metrics", type=str, default=None, help="evaluation metrics", choices=["acc", "mAP"])
parser.add_argument("--loss", type=str, default=None, help="loss function", choices=["BCE", "CE"])
parser.add_argument('--warmup', help='if warmup the learning rate', type=ast.literal_eval, default='False')
parser.add_argument("--lrscheduler_start", type=int, default=2, help="which epoch to start reducing the learning rate")
parser.add_argument("--lrscheduler_step", type=int, default=1, help="how many epochs as step to reduce the learning rate")
parser.add_argument("--lrscheduler_decay", type=float, default=0.5, help="the learning rate decay rate at each step")

parser.add_argument('--wa', help='if weight averaging', type=ast.literal_eval, default='False')
parser.add_argument('--wa_start', type=int, default=1, help="which epoch to start weight averaging the checkpoint model")
parser.add_argument('--wa_end', type=int, default=5, help="which epoch to end weight averaging the checkpoint model")
parser.add_argument("--seed", type=int, default=0, help="random seed for reproducibility")

# ---- New arguments ----
parser.add_argument("--model_variant", type=str, default='baseline',
                    choices=['baseline', 'attn_pool', 'hybrid_stem', 'multiscale', 'all_combined'],
                    help="which model variant to use")
parser.add_argument('--reverse_aug', help='if use reverse time augmentation', type=ast.literal_eval, default='False')
parser.add_argument('--freeze', help='if freeze backbone for stage 1 (epochs 1-10)', type=ast.literal_eval, default='False')
parser.add_argument("--resume_path", type=str, default=None, help="path to checkpoint model to resume training from")
# -----------------------

args = parser.parse_args()

import random
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ---- Model import based on variant ----
if args.model_variant == 'baseline':
    from the_new_audio_model import get_timm_pretrained_model
    print(f"Using baseline model")
else:
    from the_new_audio_model_enhanced import get_enhanced_model
    print(f"Using enhanced model variant: {args.model_variant}")
# ---------------------------------------

# ---- Dataloader import based on reverse_aug ----
if args.reverse_aug:
    import dataloader_enhanced as dataloader
    print("Using enhanced dataloader with reverse time augmentation")
else:
    import dataloader
    print("Using standard dataloader")
# ------------------------------------------------

# transformer based model
if args.model == 'ast':
    print('now train a audio spectrogram transformer model')

    audio_conf = {'num_mel_bins': 128, 'target_length': args.audio_length, 'freqm': args.freqm, 'timem': args.timem,
                  'mixup': args.mixup, 'dataset': args.dataset, 'mode': 'train', 'mean': args.dataset_mean,
                  'std': args.dataset_std, 'noise': args.noise}
    val_audio_conf = {'num_mel_bins': 128, 'target_length': args.audio_length, 'freqm': 0, 'timem': 0, 'mixup': 0,
                      'dataset': args.dataset, 'mode': 'evaluation', 'mean': args.dataset_mean,
                      'std': args.dataset_std, 'noise': False}

    if args.bal == 'bal':
        print('balanced sampler is being used')
        samples_weight = np.loadtxt(args.data_train[:-5]+'_weight.csv', delimiter=',')
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)
        train_loader = torch.utils.data.DataLoader(
            dataloader.AudiosetDataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf),
            batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=True)
    else:
        print('balanced sampler is not used')
        train_loader = torch.utils.data.DataLoader(
            dataloader.AudiosetDataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf),
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        dataloader.AudiosetDataset(args.data_val, label_csv=args.label_csv, audio_conf=val_audio_conf),
        batch_size=args.batch_size*2, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # ---- Instantiate model based on variant ----
    if args.model_variant == 'baseline':
        audio_model = get_timm_pretrained_model(n_classes=args.n_class, imgnet=args.imagenet_pretrain)
    else:
        audio_model = get_enhanced_model(n_classes=args.n_class, imgnet=args.imagenet_pretrain,
                                         variant=args.model_variant)
    # --------------------------------------------

print("\nCreating experiment directory: %s" % args.exp_dir)
os.makedirs(os.path.join(args.exp_dir, "models"), exist_ok=True)

with open("%s/args.pkl" % args.exp_dir, "wb") as f:
    pickle.dump(args, f)

# ---- Resume/checkpoint logic ----
args.resume_optim_path = None
if args.resume_path is not None:
    if os.path.exists(args.resume_path):
        print(f'Loading checkpoint model from: {args.resume_path}')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(args.resume_path, map_location=device)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        audio_model.load_state_dict(new_state_dict, strict=False)
        print("Checkpoint model loaded successfully.")

        optim_path = args.resume_path.replace('best_audio_model.pth', 'best_optim_state.pth')
        if os.path.exists(optim_path):
            args.resume_optim_path = optim_path
    else:
        print(f'Warning: resume_path {args.resume_path} not found. Training from scratch.')
# ---------------------------------

print('Now starting training for {:d} epochs'.format(args.n_epochs))
train(audio_model, train_loader, val_loader, args)

# for speechcommands dataset, evaluate the best model on validation set on the test set
if args.dataset == 'speechcommands':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sd = torch.load(args.exp_dir + '/models/best_audio_model.pth', map_location=device)
    audio_model = torch.nn.DataParallel(audio_model)
    audio_model.load_state_dict(sd)

    stats, _ = validate(audio_model, val_loader, args, 'valid_set')
    val_acc = stats[0]['acc']
    val_mAUC = np.mean([stat['auc'] for stat in stats])
    print('---------------evaluate on the validation set---------------')
    print("Accuracy: {:.6f}".format(val_acc))
    print("AUC: {:.6f}".format(val_mAUC))

    eval_loader = torch.utils.data.DataLoader(
        dataloader.AudiosetDataset(args.data_eval, label_csv=args.label_csv, audio_conf=val_audio_conf),
        batch_size=args.batch_size*2, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    stats, _ = validate(audio_model, eval_loader, args, 'eval_set')
    eval_acc = stats[0]['acc']
    eval_mAUC = np.mean([stat['auc'] for stat in stats])
    print('---------------evaluate on the test set---------------')
    print("Accuracy: {:.6f}".format(eval_acc))
    print("AUC: {:.6f}".format(eval_mAUC))
    np.savetxt(args.exp_dir + '/eval_result.csv', [val_acc, val_mAUC, eval_acc, eval_mAUC])