# for now, just do it between the imagenet train set
# for now, just do distance
# later, can do cosine similarity
import os
import json
import torch
import argparse
import torch.nn.functional as F
from tqdm import trange

device = 'cuda:0'


def main():
    parser = argparse.ArgumentParser(description='Find the nearest neighbors')
    parser.add_argument('--folder', type=str, required=True)
    parser.add_argument('--source', type=str, required=True)
    parser.add_argument('--target', type=str, required=True)
    parser.add_argument('--metric', default='l2', type=str)
    parser.add_argument('--recompute', action='store_true')

    args = parser.parse_args()

    if args.metric == 'l2':
        dist = l2
    elif args.metric == 'cos':
        dist = cosine_dist

    source_data = torch.load(os.path.join(args.folder, args.source))
    source_reprs = source_data['representations'].to(device)
    source_labels = source_data['targets']
    source_len = len(source_reprs)

    source_equals_target = args.target == args.source
    if args.target == args.source:
        target_reprs = source_reprs
        target_labels = source_labels
        target_len = source_len
    else:
        target_data = torch.load(os.path.join(args.folder, args.target))
        target_reprs = target_data['representations'].to(device)
        target_labels = target_data['targets']
        target_len = len(target_reprs)

    # save the nearest neighbors, if needed
    # fname might need to change, depending on eval datasets in the future
    fname = f"knn_stats_{args.metric}_{args.source.split('_')[0]}_{args.target.split('_')[0]}.pt"
    if not os.path.exists(os.path.join(args.folder, fname)) or args.recompute:
        print('Computing nearest neighbors')
        n_neighbors = 128
        distances = torch.zeros(target_len, n_neighbors)
        indices = torch.zeros(target_len, n_neighbors, dtype=torch.int64)
        batch_size = 256
        with torch.inference_mode():
            for i in trange(0, target_len, batch_size):
                curr_batch = target_reprs[i:i + batch_size]
                batch_dist = dist(curr_batch, source_reprs)
                top_dist, top_indices = batch_dist.topk(n_neighbors, largest=False)
                distances[i:i + batch_size] = top_dist.detach().cpu()
                indices[i:i + batch_size] = top_indices.detach().cpu()

        torch.save(dict(distances=distances,
                        indices=indices),
                   os.path.join(args.folder, fname))
    else:
        print('Loading saved nearest neighbors')
        knn_data = torch.load(os.path.join(args.folder, fname))
        indices = knn_data['indices']

    # compute the kNN accuracies
    if source_equals_target:
        indices = indices[:, 1:]  # if source == target, closest neighbor is self

    accs = []
    for k in trange(1, 256):
        accs.append(get_knn_acc(source_labels, target_labels, indices, k=k))
    accs = torch.tensor(accs)
    best_k = torch.argmax(accs).item() + 1
    print(f'kNN accuracy: {accs[best_k - 1]:.4f}, at k={best_k}')
    acc_fname = f"knn_accs_{args.metric}_{args.source.split('_')[0]}_{args.target.split('_')[0]}.pt"
    summary_fname = f"knn_summary_{args.metric}_{args.source.split('_')[0]}_{args.target.split('_')[0]}.json"
    torch.save(accs, os.path.join(args.folder, acc_fname))
    with open(os.path.join(args.folder, summary_fname), 'w') as fp:
        json.dump(dict(acc=accs[best_k - 1].item(), best_k=best_k), fp)


def l2(batch, source_reprs):
    return torch.linalg.norm(batch.unsqueeze(1) - source_reprs.unsqueeze(0), dim=2)


# def cosine_dist(batch, source_reprs):
#     return 1 - F.cosine_similarity(batch.unsqueeze(1), source_reprs.unsqueeze(0), dim=2)

def cosine_dist(batch, source_reprs):
    batch = batch / (1e-8 + torch.linalg.norm(batch, dim=1, keepdim=True))
    source_reprs = source_reprs / (1e-8 + torch.linalg.norm(source_reprs, dim=1, keepdim=True))
    return 1 - torch.einsum('ik,jk->ij', batch, source_reprs)

def knn_correct(source_labels, target_labels, indices, k=1):
    nearest_neighbors = source_labels[indices[:, :k]]
    preds, _ = torch.mode(nearest_neighbors, dim=1)
    return preds == target_labels


def get_knn_acc(source_labels, target_labels, indices, k=1):
    return knn_correct(source_labels, target_labels, indices, k=k).sum() / len(indices)


if __name__ == '__main__':
    main()
