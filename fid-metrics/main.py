# import hydra
# import numpy as np
# import torch
# from omegaconf import DictConfig, OmegaConf
# from rich.progress import track

# from fid_metrics import (
#     ImageDataset,
#     ImageSequenceDataset,
#     VideoDataset,
#     build_inception,
#     build_inception3d,
#     calculate_fid,
#     is_image_dir_path,
#     is_video_path,
#     postprocess_i2d_pred,
# )


# # def build_loaders(type, paths, cfg):
# #     dls = []
# #     for path in paths:
# #         bs = cfg.batch_size
# #         dataset_cfgs = cfg.get('dataset')

# #         if is_video_path(path):
# #             if type == 'fid':
# #                 if dataset_cfgs:
# #                     dataset_cfgs = dict(dataset_cfgs)
# #                     dataset_cfgs['sequence_length'] = bs
# #                 else:
# #                     dataset_cfgs = {'sequence_length': bs}
# #                 bs = 1
# #             C = VideoDataset
# #         elif is_image_dir_path(path):
# #             C = ImageDataset if type == 'fid' else ImageSequenceDataset
# #         else:
# #             raise NotImplementedError

# #         dataset = C(path, **dataset_cfgs) if dataset_cfgs else C(path)
# #         dl = torch.utils.data.DataLoader(dataset, bs, shuffle=True, num_workers=cfg.num_workers)
# #         dls.append(dl)
# #     return dls

# def build_loaders(type, paths, cfg):
#     """
#     Builds data loaders for a list of paths based on the dataset type (image/video).

#     Args:
#         type (str): Type of the dataset ('fid' or other).
#         paths (list): List of file paths (image directories or video files).
#         cfg (dict): Configuration dictionary with 'batch_size', 'num_workers', and 'dataset' settings.

#     Returns:
#         list: A list of PyTorch DataLoader objects.
#     """
#     dls = []
#     shuffle = cfg.get("shuffle", True)  # Default shuffle to True if not specified

#     for path in paths:
#         # Validate that path is a valid string
#         if not isinstance(path, str) or not path:
#             print(f"Warning: Invalid path '{path}'. Skipping...")
#             continue
        
#         bs = cfg.get("batch_size", 1)  # Fallback to batch size of 1 if not provided
#         dataset_cfgs = cfg.get("dataset", {})  # Fallback to an empty dictionary
#         dataset_cfgs = dict(dataset_cfgs)  # Ensure it's mutable

#         if is_video_path(path):
#             if type == "fid":
#                 # For FID, sequence length is the batch size, and batch size is 1
#                 dataset_cfgs["sequence_length"] = bs
#                 bs = 1
#             C = VideoDataset
#         elif is_image_dir_path(path):
#             # Select dataset class based on type
#             C = ImageDataset if type == "fid" else ImageSequenceDataset
#         else:
#             # Skip invalid paths and print a warning
#             print(f"Warning: Path '{path}' is neither a valid video nor image directory. Skipping...")
#             continue

#         # Create the dataset and DataLoader
#         try:
#             dataset = C(path, **dataset_cfgs) if dataset_cfgs else C(path)
#             dl = torch.utils.data.DataLoader(
#                 dataset, 
#                 batch_size=bs, 
#                 shuffle=shuffle, 
#                 num_workers=cfg.get("num_workers", 0)  # Default to 0 workers
#             )
#             dls.append(dl)
#         except Exception as e:
#             print(f"Error loading data from path '{path}': {e}")
#             continue

#     if not dls:
#         raise ValueError("No valid data loaders were created. Please check the provided paths and configurations.")
    
#     return dls



# def build_model(type, cfg):
#     if type == 'fid':
#         return build_inception(cfg.dims)
#     elif type == 'fvd':
#         return build_inception3d(cfg.path)
#     else:
#         raise NotImplementedError


# @hydra.main(config_path='/content/drive/MyDrive/NVIDIA_Assignment/fid-metrics/configs', config_name='config', version_base=None)
# def main(cfg: DictConfig):
#     print(OmegaConf.to_yaml(cfg))

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f'Using device: {device}')

#     for metric_cfgs in cfg.metrics:
#         type = metric_cfgs.type
#         dls = build_loaders(type, cfg.paths, metric_cfgs.data)
#         model = build_model(type, metric_cfgs.model).to(device).eval()

#         feats = [[], []]
#         for i, dl in enumerate(dls):
#             if cfg.get('num_iters'):
#                 seq = range(cfg.num_iters // metric_cfgs.data.batch_size)
#             else:
#                 seq = range(len(dl))
#             dl = iter(dl)

#             for _ in track(seq, description=f'{type}_{i}'):
#                 x = next(dl).to(device)
#                 if type == 'fid' and x.dim() == 5:
#                     x = x.squeeze(0).transpose(0, 1)
#                 elif type == 'fvd':
#                     x = x * 2 - 1
#                 with torch.no_grad():
#                     if type == 'fid':
#                         pred = model(x)
#                         pred = postprocess_i2d_pred(pred)
#                     elif type == 'fvd':
#                         pred = model.extract_features(x)
#                         pred = pred.squeeze(3).squeeze(3).mean(2)
#                 feats[i].append(pred.cpu().numpy())
#             feats[i] = np.concatenate(feats[i], axis=0)
#         fid = calculate_fid(*feats)
#         print(f'{type.upper()}: {fid}')


# if __name__ == '__main__':
#     main()

import hydra
import argparse
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from rich.progress import track

from fid_metrics import (
    ImageDataset,
    ImageSequenceDataset,
    VideoDataset,
    build_inception,
    build_inception3d,
    calculate_fid,
    is_image_dir_path,
    is_video_path,
    postprocess_i2d_pred,
)


def build_loaders(type, paths, cfg):
    dls = []
    shuffle = cfg.get("shuffle", True)  # Default shuffle to True if not specified

    for path in paths:
        if not isinstance(path, str) or not path:
            print(f"Warning: Invalid path '{path}'. Skipping...")
            continue

        bs = cfg.get("batch_size", 1)  # Default batch size to 1
        dataset_cfgs = cfg.get("dataset", {})  # Default to an empty dictionary
        dataset_cfgs = dict(dataset_cfgs)  # Ensure it's mutable

        if is_video_path(path):
            if type == "fid":
                dataset_cfgs["sequence_length"] = bs
                bs = 1
            C = VideoDataset
        elif is_image_dir_path(path):
            C = ImageDataset if type == "fid" else ImageSequenceDataset
        else:
            print(f"Warning: Path '{path}' is neither a valid video nor image directory. Skipping...")
            continue

        try:
            dataset = C(path, **dataset_cfgs) if dataset_cfgs else C(path)
            dl = torch.utils.data.DataLoader(
                dataset, 
                batch_size=bs, 
                shuffle=shuffle, 
                num_workers=cfg.get("num_workers", 0)  # Default workers to 0
            )
            dls.append(dl)
        except Exception as e:
            print(f"Error loading data from path '{path}': {e}")
            continue

    if not dls:
        raise ValueError("No valid data loaders were created. Please check the provided paths and configurations.")
    
    return dls


def build_model(type, cfg):
    if type == 'fid':
        return build_inception(cfg.dims)
    elif type == 'fvd':
        return build_inception3d(cfg.path)
    else:
        raise NotImplementedError


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", nargs="+", help="List of video/image paths to process")
    args, _ = parser.parse_known_args()  # Ignore unknown args (Hydra will handle them)
    return args


@hydra.main(config_path='/content/drive/MyDrive/NVIDIA_Assignment/fid-metrics/configs', config_name='config', version_base=None)
def main(cfg: DictConfig):
    # Parse arguments for dynamic path override
    args = parse_args()

    # Override paths in configuration if provided
    if args.paths:
        cfg.paths = args.paths

    print(OmegaConf.to_yaml(cfg))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    for metric_cfgs in cfg.metrics:
        type = metric_cfgs.type
        dls = build_loaders(type, cfg.paths, metric_cfgs.data)
        model = build_model(type, metric_cfgs.model).to(device).eval()

        feats = [[], []]
        for i, dl in enumerate(dls):
            if cfg.get('num_iters'):
                seq = range(cfg.num_iters // metric_cfgs.data.batch_size)
            else:
                seq = range(len(dl))
            dl = iter(dl)

            for _ in track(seq, description=f'{type}_{i}'):
                x = next(dl).to(device)
                if type == 'fid' and x.dim() == 5:
                    x = x.squeeze(0).transpose(0, 1)
                elif type == 'fvd':
                    x = x * 2 - 1
                with torch.no_grad():
                    if type == 'fid':
                        pred = model(x)
                        pred = postprocess_i2d_pred(pred)
                    elif type == 'fvd':
                        pred = model.extract_features(x)
                        pred = pred.squeeze(3).squeeze(3).mean(2)
                feats[i].append(pred.cpu().numpy())
            feats[i] = np.concatenate(feats[i], axis=0)
        fid = calculate_fid(*feats)
        print(f'{type.upper()}: {fid}')


if __name__ == '__main__':
    main()
