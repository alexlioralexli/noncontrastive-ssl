import os
import torch
import argparse
import os.path as osp

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pretrained', type=str, help='path to simsiam pretrained checkpoint')
    parser.add_argument('--type', type=str, default='svd')
    parser.add_argument('--normalize', action="store_true")
    args = parser.parse_args()

    # load the data
    folder = osp.join(osp.dirname(args.pretrained))
    data = torch.load(args.pretrained)
    reprs = data['representations']
    reprs = reprs.reshape(-1, reprs.shape[-1])

    norms = torch.linalg.norm(reprs, dim=1)
    if args.normalize:
        normed_reprs = reprs / (1e-6 + norms.unsqueeze(1))
    else:
        normed_reprs = reprs
    normed_reprs -= normed_reprs.mean(dim=0, keepdims=True)
    if args.type == 'svd':
        stds = torch.svd(normed_reprs).S
    elif args.type == 'std':
        stds = torch.std(normed_reprs, dim=0)
    else:
        raise NotImplementedError

    # save norms and std
    normalize_str = 'normalized_' if args.normalize else ''
    fname = f"{os.path.basename(args.pretrained).split('.')[0]}_{normalize_str}{args.type}.pt"
    # torch.save(dict(norms=norms, stds=stds), osp.join(folder, fname))
    torch.save(dict(stds=stds), osp.join(folder, fname))
