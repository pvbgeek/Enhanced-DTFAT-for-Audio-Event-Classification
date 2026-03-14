# -*- coding: utf-8 -*-
# @File    : dataloader_enhanced.py
# Enhanced dataloader with Reverse Time Augmentation
# Based on dataloader.py by Yuan Gong (MIT)
#
# Reverse Time Augmentation Strategy:
#   Training  : Dataset is DOUBLED — every sample appears once forward + once reversed
#               Same label values for both directions
#               Model guaranteed to see every sample both ways every epoch
#   Evaluation: Forward only — matches real-world deployment conditions
#               Ensures fair comparison with baselines (David's 0.487 etc.)

import csv
import json
import soundfile as sf
import torchaudio
import numpy as np
import torch
import torch.nn.functional
from torch.utils.data import Dataset
import random


def make_index_dict(label_csv):
    index_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            index_lookup[row['mid']] = row['index']
    return index_lookup


def make_name_dict(label_csv):
    name_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            name_lookup[row['index']] = row['display_name']
    return name_lookup


def lookup_list(index_list, label_csv):
    label_list = []
    table = make_name_dict(label_csv)
    for item in index_list:
        label_list.append(table[item])
    return label_list


def preemphasis(signal, coeff=0.97):
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])


def load_audio(filename):
    """
    Load audio using soundfile (avoids torchaudio backend/torchcodec issues).
    Returns: waveform [1, samples] tensor, sample_rate int
    """
    waveform_np, sr = sf.read(filename, always_2d=True)  # [samples, channels]
    waveform = torch.from_numpy(waveform_np.T).float()   # [channels, samples]
    # Mix down to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform, sr


class AudiosetDataset(Dataset):
    def __init__(self, dataset_json_file, audio_conf, label_csv=None):
        """
        Dataset that manages audio recordings.
        :param audio_conf: Dictionary containing the audio loading and preprocessing settings
        :param dataset_json_file
        """
        self.datapath = dataset_json_file
        with open(dataset_json_file, 'r') as fp:
            data_json = json.load(fp)

        self.audio_conf = audio_conf
        print('---------------the {:s} dataloader---------------'.format(self.audio_conf.get('mode')))
        self.melbins = self.audio_conf.get('num_mel_bins')
        self.freqm = self.audio_conf.get('freqm')
        self.timem = self.audio_conf.get('timem')
        print('now using following mask: {:d} freq, {:d} time'.format(self.audio_conf.get('freqm'), self.audio_conf.get('timem')))
        self.mixup = self.audio_conf.get('mixup')
        print('now using mix-up with rate {:f}'.format(self.mixup))
        self.dataset = self.audio_conf.get('dataset')
        print('now process ' + self.dataset)
        self.norm_mean = self.audio_conf.get('mean')
        self.norm_std = self.audio_conf.get('std')
        self.skip_norm = self.audio_conf.get('skip_norm') if self.audio_conf.get('skip_norm') else False
        if self.skip_norm:
            print('now skip normalization (use it ONLY when you are computing the normalization stats).')
        else:
            print('use dataset mean {:.3f} and std {:.3f} to normalize the input.'.format(self.norm_mean, self.norm_std))
        self.noise = self.audio_conf.get('noise')
        if self.noise == True:
            print('now use noise augmentation')

        self.mode = self.audio_conf.get('mode', 'train')

        # ---------------------------------------------------------------
        # Reverse Time Augmentation — Dataset Doubling Strategy
        #
        # Training:   double the dataset
        #             first half  → reversed=False (forward audio)
        #             second half → reversed=True  (reversed audio)
        #             Same label values for both
        #             Every sample seen both ways every epoch
        #
        # Evaluation: no reversing — matches real-world deployment
        #             fair comparison with all baselines
        # ---------------------------------------------------------------
        raw_data = data_json['data']

        if self.mode == 'train':
            # Tag each entry with its direction
            forward  = [dict(d, reversed=False) for d in raw_data]
            reversed_ = [dict(d, reversed=True)  for d in raw_data]
            self.data = forward + reversed_
            print(f'Reverse Time Augmentation: dataset DOUBLED '
                  f'{len(raw_data)} → {len(self.data)} samples '
                  f'(all forward + all reversed, same labels)')
        else:
            # Evaluation: forward only
            self.data = [dict(d, reversed=False) for d in raw_data]
            print('Reverse Time Augmentation DISABLED (evaluation mode — forward only)')

        self.index_dict = make_index_dict(label_csv)
        self.label_num = len(self.index_dict)
        print('number of classes is {:d}'.format(self.label_num))

    def _resample_wav(self, wav, orig_sr, req_sr):
        if orig_sr == req_sr:
            return wav
        resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=req_sr)
        return resampler(wav)

    def _apply_reverse(self, waveform, is_reversed):
        """
        Deterministically reverse waveform based on sample flag.
        No randomness — every reversed sample is always reversed.
        """
        if is_reversed:
            return torch.flip(waveform, dims=[-1])
        return waveform

    def _wav2fbank(self, filename, is_reversed, filename2=None, is_reversed2=False):
        SAMPLE_RATE = 32000

        if filename2 is None:
            waveform, sr = load_audio(filename)
            waveform = self._resample_wav(waveform, sr, SAMPLE_RATE)
            sr = SAMPLE_RATE

            # Apply deterministic reversal before fbank computation
            waveform = self._apply_reverse(waveform, is_reversed)

            waveform = waveform - waveform.mean()
        else:
            waveform1, sr1 = load_audio(filename)
            waveform1 = self._resample_wav(waveform1, sr1, SAMPLE_RATE)

            waveform2, sr2 = load_audio(filename2)
            waveform2 = self._resample_wav(waveform2, sr2, SAMPLE_RATE)

            # Apply reversal independently to each waveform
            waveform1 = self._apply_reverse(waveform1, is_reversed)
            waveform2 = self._apply_reverse(waveform2, is_reversed2)

            waveform1 = waveform1 - waveform1.mean()
            waveform2 = waveform2 - waveform2.mean()

            if waveform1.shape[1] != waveform2.shape[1]:
                if waveform1.shape[1] > waveform2.shape[1]:
                    temp_wav = torch.zeros(1, waveform1.shape[1])
                    temp_wav[0, 0:waveform2.shape[1]] = waveform2
                    waveform2 = temp_wav
                else:
                    waveform2 = waveform2[0, 0:waveform1.shape[1]]

            mix_lambda = np.random.beta(10, 10)
            mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
            waveform = mix_waveform - mix_waveform.mean()
            sr = SAMPLE_RATE

        fbank = torchaudio.compliance.kaldi.fbank(
            waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
            window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=10
        )

        target_length = self.audio_conf.get('target_length')
        n_frames = fbank.shape[0]
        p = target_length - n_frames

        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]

        if filename2 is None:
            return fbank, 0
        else:
            return fbank, mix_lambda

    def __getitem__(self, index):
        datum = self.data[index]
        is_reversed = datum.get('reversed', False)

        if random.random() < self.mixup:
            # For mixup partner, sample randomly from full dataset
            # Partner inherits its own reversed flag
            mix_sample_idx = random.randint(0, len(self.data) - 1)
            mix_datum = self.data[mix_sample_idx]
            is_reversed2 = mix_datum.get('reversed', False)

            fbank, mix_lambda = self._wav2fbank(
                datum['wav'], is_reversed,
                mix_datum['wav'], is_reversed2
            )
            label_indices = np.zeros(self.label_num)
            for label_str in datum['labels'].split(','):
                label_indices[int(self.index_dict[label_str])] += mix_lambda
            for label_str in mix_datum['labels'].split(','):
                label_indices[int(self.index_dict[label_str])] += 1.0 - mix_lambda
            label_indices = torch.FloatTensor(label_indices)
        else:
            label_indices = np.zeros(self.label_num)
            fbank, mix_lambda = self._wav2fbank(datum['wav'], is_reversed)
            for label_str in datum['labels'].split(','):
                label_indices[int(self.index_dict[label_str])] = 1.0
            label_indices = torch.FloatTensor(label_indices)

        # SpecAug — not applied for eval set
        freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
        timem = torchaudio.transforms.TimeMasking(self.timem)
        fbank = torch.transpose(fbank, 0, 1)
        fbank = fbank.unsqueeze(0)
        if self.freqm != 0:
            fbank = freqm(fbank)
        if self.timem != 0:
            fbank = timem(fbank)
        fbank = fbank.squeeze(0)
        fbank = torch.transpose(fbank, 0, 1)

        if not self.skip_norm:
            fbank = (fbank - self.norm_mean) / (self.norm_std * 2)

        if self.noise == True:
            fbank = fbank + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10
            fbank = torch.roll(fbank, np.random.randint(-10, 10), 0)

        mix_ratio = min(mix_lambda, 1 - mix_lambda) / max(mix_lambda, 1 - mix_lambda)

        fbank = fbank[None, :, :]  # [1, 1024, 128]

        return fbank, label_indices

    def __len__(self):
        return len(self.data)









#############################################################################################################################################################################################################################################################################
# # -*- coding: utf-8 -*-
# # @File    : dataloader_enhanced.py
# # Enhanced dataloader with Reverse Time Augmentation
# # Based on dataloader.py by Yuan Gong (MIT)
# # Enhancement: Reverse Time Augmentation added for training only
# # Fix: uses soundfile directly for audio loading (torchaudio.load backend issue)

# import csv
# import json
# import soundfile as sf
# import torchaudio
# import numpy as np
# import torch
# import torch.nn.functional
# from torch.utils.data import Dataset
# import random

# def make_index_dict(label_csv):
#     index_lookup = {}
#     with open(label_csv, 'r') as f:
#         csv_reader = csv.DictReader(f)
#         for row in csv_reader:
#             index_lookup[row['mid']] = row['index']
#     return index_lookup

# def make_name_dict(label_csv):
#     name_lookup = {}
#     with open(label_csv, 'r') as f:
#         csv_reader = csv.DictReader(f)
#         for row in csv_reader:
#             name_lookup[row['index']] = row['display_name']
#     return name_lookup

# def lookup_list(index_list, label_csv):
#     label_list = []
#     table = make_name_dict(label_csv)
#     for item in index_list:
#         label_list.append(table[item])
#     return label_list

# def preemphasis(signal, coeff=0.97):
#     return np.append(signal[0], signal[1:] - coeff * signal[:-1])

# def load_audio(filename):
#     """
#     Load audio using soundfile (avoids torchaudio backend issues).
#     Returns: waveform [1, samples] tensor, sample_rate int
#     """
#     waveform_np, sr = sf.read(filename, always_2d=True)  # [samples, channels]
#     waveform = torch.from_numpy(waveform_np.T).float()   # [channels, samples]
#     # Mix to mono if stereo
#     if waveform.shape[0] > 1:
#         waveform = waveform.mean(dim=0, keepdim=True)
#     return waveform, sr


# class AudiosetDataset(Dataset):
#     def __init__(self, dataset_json_file, audio_conf, label_csv=None):
#         """
#         Dataset that manages audio recordings
#         :param audio_conf: Dictionary containing the audio loading and preprocessing settings
#         :param dataset_json_file
#         """
#         self.datapath = dataset_json_file
#         with open(dataset_json_file, 'r') as fp:
#             data_json = json.load(fp)

#         self.data = data_json['data']
#         self.audio_conf = audio_conf
#         print('---------------the {:s} dataloader---------------'.format(self.audio_conf.get('mode')))
#         self.melbins = self.audio_conf.get('num_mel_bins')
#         self.freqm = self.audio_conf.get('freqm')
#         self.timem = self.audio_conf.get('timem')
#         print('now using following mask: {:d} freq, {:d} time'.format(self.audio_conf.get('freqm'), self.audio_conf.get('timem')))
#         self.mixup = self.audio_conf.get('mixup')
#         print('now using mix-up with rate {:f}'.format(self.mixup))
#         self.dataset = self.audio_conf.get('dataset')
#         print('now process ' + self.dataset)
#         self.norm_mean = self.audio_conf.get('mean')
#         self.norm_std = self.audio_conf.get('std')
#         self.skip_norm = self.audio_conf.get('skip_norm') if self.audio_conf.get('skip_norm') else False
#         if self.skip_norm:
#             print('now skip normalization (use it ONLY when you are computing the normalization stats).')
#         else:
#             print('use dataset mean {:.3f} and std {:.3f} to normalize the input.'.format(self.norm_mean, self.norm_std))
#         self.noise = self.audio_conf.get('noise')
#         if self.noise == True:
#             print('now use noise augmentation')

#         # ---- Reverse Time Augmentation ----
#         self.mode = self.audio_conf.get('mode', 'train')
#         self.reverse_prob = 0.5
#         if self.mode == 'train':
#             print(f'Reverse Time Augmentation ENABLED with probability {self.reverse_prob}')
#         else:
#             print('Reverse Time Augmentation DISABLED (evaluation mode)')
#         # -----------------------------------

#         self.index_dict = make_index_dict(label_csv)
#         self.label_num = len(self.index_dict)
#         print('number of classes is {:d}'.format(self.label_num))

#     def _resample_wav(self, wav, orig_sr, req_sr):
#         if orig_sr == req_sr:
#             return wav
#         resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=req_sr)
#         return resampler(wav)

#     def _apply_reverse_aug(self, waveform):
#         """
#         Reverse Time Augmentation: flip waveform along time axis.
#         Only applied during training with probability self.reverse_prob.
#         Spectral content is preserved — only temporal direction changes.
#         """
#         if self.mode == 'train' and random.random() < self.reverse_prob:
#             waveform = torch.flip(waveform, dims=[-1])
#         return waveform

#     def _wav2fbank(self, filename, filename2=None):
#         SAMPLE_RATE = 32000

#         if filename2 is None:
#             waveform, sr = load_audio(filename)
#             waveform = self._resample_wav(waveform, sr, SAMPLE_RATE)
#             sr = SAMPLE_RATE

#             # Apply Reverse Time Augmentation before fbank
#             waveform = self._apply_reverse_aug(waveform)

#             waveform = waveform - waveform.mean()
#         else:
#             waveform1, sr1 = load_audio(filename)
#             waveform1 = self._resample_wav(waveform1, sr1, SAMPLE_RATE)

#             waveform2, sr2 = load_audio(filename2)
#             waveform2 = self._resample_wav(waveform2, sr2, SAMPLE_RATE)

#             # Apply Reverse Time Augmentation independently to each waveform
#             waveform1 = self._apply_reverse_aug(waveform1)
#             waveform2 = self._apply_reverse_aug(waveform2)

#             waveform1 = waveform1 - waveform1.mean()
#             waveform2 = waveform2 - waveform2.mean()

#             if waveform1.shape[1] != waveform2.shape[1]:
#                 if waveform1.shape[1] > waveform2.shape[1]:
#                     temp_wav = torch.zeros(1, waveform1.shape[1])
#                     temp_wav[0, 0:waveform2.shape[1]] = waveform2
#                     waveform2 = temp_wav
#                 else:
#                     waveform2 = waveform2[0, 0:waveform1.shape[1]]

#             mix_lambda = np.random.beta(10, 10)
#             mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
#             waveform = mix_waveform - mix_waveform.mean()
#             sr = SAMPLE_RATE

#         fbank = torchaudio.compliance.kaldi.fbank(
#             waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
#             window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=10
#         )

#         target_length = self.audio_conf.get('target_length')
#         n_frames = fbank.shape[0]
#         p = target_length - n_frames

#         if p > 0:
#             m = torch.nn.ZeroPad2d((0, 0, 0, p))
#             fbank = m(fbank)
#         elif p < 0:
#             fbank = fbank[0:target_length, :]

#         if filename2 is None:
#             return fbank, 0
#         else:
#             return fbank, mix_lambda

#     def __getitem__(self, index):
#         if random.random() < self.mixup:
#             datum = self.data[index]
#             mix_sample_idx = random.randint(0, len(self.data) - 1)
#             mix_datum = self.data[mix_sample_idx]
#             fbank, mix_lambda = self._wav2fbank(datum['wav'], mix_datum['wav'])
#             label_indices = np.zeros(self.label_num)
#             for label_str in datum['labels'].split(','):
#                 label_indices[int(self.index_dict[label_str])] += mix_lambda
#             for label_str in mix_datum['labels'].split(','):
#                 label_indices[int(self.index_dict[label_str])] += 1.0 - mix_lambda
#             label_indices = torch.FloatTensor(label_indices)
#         else:
#             datum = self.data[index]
#             label_indices = np.zeros(self.label_num)
#             fbank, mix_lambda = self._wav2fbank(datum['wav'])
#             for label_str in datum['labels'].split(','):
#                 label_indices[int(self.index_dict[label_str])] = 1.0
#             label_indices = torch.FloatTensor(label_indices)

#         # SpecAug (not applied for eval set)
#         freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
#         timem = torchaudio.transforms.TimeMasking(self.timem)
#         fbank = torch.transpose(fbank, 0, 1)
#         fbank = fbank.unsqueeze(0)
#         if self.freqm != 0:
#             fbank = freqm(fbank)
#         if self.timem != 0:
#             fbank = timem(fbank)
#         fbank = fbank.squeeze(0)
#         fbank = torch.transpose(fbank, 0, 1)

#         if not self.skip_norm:
#             fbank = (fbank - self.norm_mean) / (self.norm_std * 2)

#         if self.noise == True:
#             fbank = fbank + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10
#             fbank = torch.roll(fbank, np.random.randint(-10, 10), 0)

#         mix_ratio = min(mix_lambda, 1 - mix_lambda) / max(mix_lambda, 1 - mix_lambda)

#         fbank = fbank[None, :, :]  # [1, 1024, 128]

#         return fbank, label_indices

#     def __len__(self):
#         return len(self.data)