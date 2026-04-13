import torch
from pytorch_metric_learning.utils import inference
from pytorch_metric_learning import samplers, losses, distances
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import VSLNet


def train(train_dataset, val_dataset, train_labels, val_labels, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_sampler = samplers.MPerClassSampler(
        labels = train_labels,
        m=args.train['videos'],
        length_before_new_iter=len(train_dataset),
    )

    train_loader = DataLoader(train_dataset, batch_size=int(args.train['labels'] * args.train['videos']), sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=int(args.train['labels'] * args.train['videos']), sampler=train_sampler)

    model = VSLNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.train['lr'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train['epochs'])

    loss_fnc = losses.SupConLoss().to(device)
    knn_func = inference.CustomKNN(distances.CosineSimilarity())
    calculator = AccuracyCalculator(k=1, knn_func=knn_func)

    best_val = 0.0
    for epoch in range(1, args.train['epochs'] + 1):
        model.train()
        total_loss = 0.0
        for data, target in tqdm(train_loader, desc=f"Epoch {epoch}/{args.train['epochs']}"):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            embeddings = model(data)




