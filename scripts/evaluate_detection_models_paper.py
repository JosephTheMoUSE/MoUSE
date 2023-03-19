import sys
import pytorch_lightning as pl
import logging
from argparse import ArgumentParser
import os
from collections import defaultdict
from functools import partial
from itertools import chain
from copy import deepcopy

import numpy as np
import pandas as pd
import seaborn as sns
import glob
import re
import torchaudio
from skimage import segmentation
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

from mouse.nn_detection.neural_network import find_USVs as find_USVs_nn
from mouse.segmentation.GAC import find_USVs as find_USVs_gac, eroded_level_set_generator
from mouse.utils.sound_util import spectrogram as generate_spectrogram, SpectrogramData, clip_spectrogram
from mouse.utils.data_util import load_table, SqueakBox
from mouse.utils.metrics import (intersection_over_union_global,
                                 detection_precision,
                                 detection_recall,
                                 f_beta,
                                 coverage,
                                 intersection_over_union_elementwise)
from mouse.denoising import noise_gate_filter
from tqdm import tqdm
from pathlib import Path

pl._logger.setLevel(logging.ERROR)
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


def _find_closest_idx(arr, item):
    return np.argmin(np.abs(arr - item))


def _clean_labels(label):
    return label if label != 'junk' else 'background'


def eval_nn_detection(spec, model_name, thresholds=(-1, 0.1)):
    return {
        f'{model_name}({th})':
        find_USVs_nn(spec_data=spec,
                     model_name=model_name,
                     cache_dir=Path('.'),
                     batch_size=1,
                     confidence_threshold=th,
                     tqdm_kwargs=dict(position=1,
                                      leave=False,
                                      desc=f'{model_name}({th})')) for th in thresholds
    }


def eval_gac_detection(spec, param_sets):
    res = {}
    for idx, params in enumerate(param_sets, 1):
        filter_18 = params['filter_18']
        del params['filter_18']
        if 'denoise_fn' in params:
            s = params['denoise_fn'](spec)
            del params['denoise_fn']
        else:
            s = spec
        preds = find_USVs_gac(s,
                              tqdm_kwargs=dict(position=1,
                                               leave=False,
                                               desc=f'GAC-{idx}'),
                              **params)
        preds = [
            p for p in preds
            if (p.freq_end - p.freq_start) < 100 and (p.t_end - p.t_start) < 5000
        ]
        if filter_18:
            preds = [
                p for p in preds
                if (p.freq_end + p.freq_start) / 2 >= spec.freq_to_pixels(18000)
            ]
        res[f'GAC-{idx}'] = preds
    return res


def denoise_spec(
    spec: SpectrogramData,
    boxes=None,
    file=None,
    noise_len=1.0,
    n_grad_freq=3,
    n_grad_time=3,
    n_std_thresh=1.0,
    noise_decrease=0.5,
):
    spec = deepcopy(spec)
    spans = []
    boxes = sorted(boxes, key=lambda b: b.t_start)
    current_end = 0
    min_noise_sample_len = spec.time_to_pixels(noise_len)
    for box in boxes:
        d = current_end - box.t_start
        if d >= min_noise_sample_len:
            spans.append([current_end, box.t_start])
            current_end = box.t_end
        elif d >= 0:
            current_end = box.t_end
        elif current_end < box.t_end:
            current_end = box.t_end
    if spans:
        try:
            weights = np.array([
                np.mean(np.quantile(spec.spec[:, s:e], 0.75, axis=1)) for s,
                e in spans if e - s > 1
            ])
            span = spans[np.random.choice(len(spans), p=weights / np.sum(weights))]
        except:
            span = [0, spec.times.shape[0] - 1]
            noise_decrease = noise_decrease / 2
            warnings.warn(
                f"Error occurred while searching for noise. For file {file}\n"
                "Span will be chosen randomly. Decreasing `noise_decrease` by half")
    else:
        span = [0, spec.times.shape[0] - 1]
        noise_decrease = noise_decrease / 2
        warnings.warn(
            f"Can't find sufficiently long span without calls. For file {file}\n"
            "Span will be chosen randomly. Decreasing `noise_decrease` by half")

    if span[1] - span[0] > min_noise_sample_len:
        span[0] = np.random.randint(span[0], span[1] - min_noise_sample_len + 1)
        span[1] = span[0] + min_noise_sample_len

    noise_spec = clip_spectrogram(spec=spec,
                                  t_start=spec.times[span[0]],
                                  t_end=spec.times[span[1]])
    noise_gate_filter(
        spectrogram=spec,
        noise_spectrogram=noise_spec,
        n_grad_freq=n_grad_freq,
        n_grad_time=n_grad_time,
        n_std_thresh=n_std_thresh,
        noise_decrease=noise_decrease,
    )
    spec.spec = spec.spec.float()
    return spec


def iou_dict_to_class(iou_dict):
    result = []
    for (squeak, intersections) in iou_dict.items():
        if len(intersections.values()) > 0:
            idx = np.argmax(intersections.values())
            result.append((squeak.label, list(intersections.keys())[idx].label))
        else:
            result.append((squeak.label, "noise"))
    return result


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--test_data_dir', default='./test/')
    parser.add_argument('--nn_model', default='f-rcnn-custom')
    parser.add_argument('--model_thresholds',
                        nargs='+',
                        type=float,
                        default=[0., 0.3, 0.5, 0.7])
    parser.add_argument('--calculate_roc', action='store_true')  # todo
    args = parser.parse_args()

    np.random.seed(0)

    annotation_file = glob.glob(os.path.join(args.test_data_dir, '*.txt'))
    assert len(annotation_file) == 1, f"there should be 1 annonation .txt file in directory {args.test_data_dir} " \
                                      f"got {len(annotation_file)}"

    annotation_file = annotation_file[0]
    annotations = load_table(annotation_file)
    annotations.rename(
        columns=lambda x: re.sub('_\(.*\)', '', x).replace(' ', '_').lower(),
        inplace=True)
    #audio_files = annotations.begin_file.unique()
    audio_files = [
        'ch1-2020-11-23_13-01-52_0000013.wav',
        'ch1-2020-11-23_13-05-52_0000017.wav',
        'ch1-2020-11-26_12-40-23_0000001.wav',
        'ch1-2020-11-20_11-43-10_0000006.wav',
        'ch1-2020-11-23_12-48-55_0000001.wav',
        'ch1-2020-11-24_10-53-15_0000005.wav'
    ]
    audio_files = sorted(audio_files)

    annotations.usv_type = annotations.usv_type.map(_clean_labels)

    results = defaultdict(lambda: defaultdict(float))
    results_per_class = defaultdict(lambda: defaultdict(float))

    y_true = defaultdict(list)
    y_pred = defaultdict(list)

    for file in tqdm(audio_files, position=0):

        waveform, sample_rate = torchaudio.load(os.path.join(args.test_data_dir, file))

        spec = generate_spectrogram(waveform, sample_rate=sample_rate)

        usv_type = 'usv_type' if 'usv_type' in annotations.keys() else 'note'
        file_annotations = annotations.loc[
            annotations.begin_file == file,
            ['begin_time', 'end_time', 'low_freq', 'high_freq', usv_type]]

        ground_truth = []
        for idx, (begin_time, end_time, low_freq, high_freq,
                  cls) in file_annotations.iterrows():
            if cls == 'background':
                continue
            time_start_index = _find_closest_idx(spec.times, begin_time)
            time_end_index = _find_closest_idx(spec.times, end_time)

            freq_start_index = _find_closest_idx(spec.freqs, low_freq)
            freq_end_index = _find_closest_idx(spec.freqs, high_freq)

            gt_box = SqueakBox(freq_start=freq_start_index,
                               freq_end=freq_end_index,
                               t_start=time_start_index,
                               t_end=time_end_index,
                               label=cls)
            ground_truth.append(gt_box)

        param_sets = [{
            'preprocessing_fn':
                partial(
                    segmentation.inverse_gaussian_gradient,
                    sigma=7.0,
                    alpha=200.0,
                ),
            'level_set':
                eroded_level_set_generator(threshold=0.75),
            'balloon':
                -1,
            'threshold':
                0.8,
            'smoothing':
                3,
            'iterations':
                15,
            'filter_18':
                False
        },
                      {
                          'preprocessing_fn':
                              partial(
                                  segmentation.inverse_gaussian_gradient,
                                  sigma=7.0,
                                  alpha=200.0,
                              ),
                          'level_set':
                              eroded_level_set_generator(threshold=0.75),
                          'balloon':
                              -1,
                          'threshold':
                              0.8,
                          'smoothing':
                              3,
                          'iterations':
                              15,
                          'filter_18':
                              True
                      },
                      {
                          'preprocessing_fn':
                              partial(
                                  segmentation.inverse_gaussian_gradient,
                                  sigma=7.0,
                                  alpha=200.0,
                              ),
                          'denoise_fn':
                              partial(denoise_spec,
                                      boxes=ground_truth,
                                      file=file,
                                      noise_len=1.0,
                                      n_grad_freq=3,
                                      n_grad_time=3,
                                      n_std_thresh=1.0,
                                      noise_decrease=0.5),
                          'level_set':
                              eroded_level_set_generator(threshold=0.75),
                          'balloon':
                              -1,
                          'threshold':
                              0.8,
                          'smoothing':
                              3,
                          'iterations':
                              15,
                          'filter_18':
                              False
                      },
                      {
                          'preprocessing_fn':
                              partial(
                                  segmentation.inverse_gaussian_gradient,
                                  sigma=7.0,
                                  alpha=200.0,
                              ),
                          'denoise_fn':
                              partial(denoise_spec,
                                      boxes=ground_truth,
                                      file=file,
                                      noise_len=1.0,
                                      n_grad_freq=3,
                                      n_grad_time=3,
                                      n_std_thresh=1.0,
                                      noise_decrease=0.5),
                          'level_set':
                              eroded_level_set_generator(threshold=0.75),
                          'balloon':
                              -1,
                          'threshold':
                              0.8,
                          'smoothing':
                              3,
                          'iterations':
                              15,
                          'filter_18':
                              True
                      },
                      {
                          'preprocessing_fn':
                              partial(
                                  segmentation.inverse_gaussian_gradient,
                                  sigma=4.0,
                                  alpha=200.0,
                              ),
                          'denoise_fn':
                              partial(denoise_spec,
                                      boxes=ground_truth,
                                      file=file,
                                      noise_len=1.0,
                                      n_grad_freq=3,
                                      n_grad_time=3,
                                      n_std_thresh=1.0,
                                      noise_decrease=0.5),
                          'level_set':
                              eroded_level_set_generator(threshold=0.8),
                          'balloon':
                              -1,
                          'threshold':
                              0.8,
                          'smoothing':
                              2,
                          'iterations':
                              15,
                          'filter_18':
                              False
                      },
                      {
                          'preprocessing_fn':
                              partial(
                                  segmentation.inverse_gaussian_gradient,
                                  sigma=4.0,
                                  alpha=200.0,
                              ),
                          'denoise_fn':
                              partial(denoise_spec,
                                      boxes=ground_truth,
                                      file=file,
                                      noise_len=1.0,
                                      n_grad_freq=3,
                                      n_grad_time=3,
                                      n_std_thresh=1.0,
                                      noise_decrease=0.5),
                          'level_set':
                              eroded_level_set_generator(threshold=0.8),
                          'balloon':
                              -1,
                          'threshold':
                              0.8,
                          'smoothing':
                              2,
                          'iterations':
                              15,
                          'filter_18':
                              True
                      }]

        nn_model_preds = eval_nn_detection(spec=spec,
                                           model_name=args.nn_model,
                                           thresholds=args.model_thresholds)

        preds_gac = eval_gac_detection(spec=spec, param_sets=param_sets)

        for m, preds in chain(nn_model_preds.items(), preds_gac.items()):
            results[m]['iou'] += intersection_over_union_global(preds,
                                                                ground_truth) * 100
            cov_pred_by_gt, cov_gt_by_pred = coverage(preds, ground_truth)
            results[m]['cov_pred_by_gt'] += cov_pred_by_gt * 100
            results[m]['cov_gt_by_pred'] += cov_gt_by_pred * 100
            for tr in [0.1, 0.5, 0.8]:
                pr = detection_precision(ground_truth, preds, threshold=tr) * 100
                rc, rc_per_class = detection_recall(ground_truth, preds, threshold=tr)
                rc *= 100

                results[m][f'precision@{tr*100}'] += pr
                results[m][f'recall@{tr*100}'] += rc
                results[m][f'f1@{tr*100}'] += f_beta(pr, rc, 1.)
                for label, rc in rc_per_class.items():
                    results_per_class[m][f'{label}@{tr * 100}'] += rc * 100

        for m, preds in nn_model_preds.items():
            iou_elementwise = intersection_over_union_elementwise(preds, ground_truth)
            hits = iou_dict_to_class(iou_elementwise)
            y_true[m] += [h[1] for h in hits]
            y_pred[m] += [h[0] for h in hits]

    res = pd.DataFrame(results)
    res /= len(audio_files)
    res_per_class = pd.DataFrame(results_per_class)
    res_per_class /= len(audio_files)

    print('Global results')
    print(res)

    print('Results per class')
    print(res_per_class)

    os.makedirs('results', exist_ok=True)
    res.to_csv('results/results.csv')
    res_per_class.to_csv('results/results_per_class.csv')

    accs = {}
    labels = ["tr", "oc", "mc", "sh", "fl", "trc", "22khz"]
    for m in y_pred:
        cf_matrix = confusion_matrix(y_true[m], y_pred[m], labels=labels)
        cf_matrix = pd.DataFrame(cf_matrix, index=labels, columns=labels)
        accs[m] = {'accuracy': accuracy_score(y_true[m], y_pred[m]) * 100}
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(cf_matrix, linewidths=1, annot=True, ax=ax, fmt='g', cmap="Blues")
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(f'results/cf_matrix_{m}.png')

    accs = pd.DataFrame(accs)

    print(accs)
    accs.to_csv('results/accuracy.csv')
