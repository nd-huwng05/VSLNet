import os
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms, Compose
from tqdm import tqdm
from dataset.data_preprocessing import RandomTemporalCrop, RandomPoseScale, RandomPoseNoise, TemporalInterpolatePose, PoseJoinSelect, PoseNormalize
from dataset.vsl_dataset import VSLPoseDataset
from models.metrics import SupervisedContrastiveLoss, calculate_metrics
from models.vsl_net import VSLContrastiveNet

def train(args):
    print("Mode training...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Setting device {device} successfully!")
    print(f"Preparing data loaders...")
    train_transforms = Compose([
        PoseJoinSelect(),
        RandomTemporalCrop(frames=64),
        PoseNormalize(),
        RandomPoseScale(0.8, 1.2),
        RandomPoseNoise(std=0.01)
    ])

    val_transforms = Compose([
        PoseJoinSelect(),
        TemporalInterpolatePose(frames=64),
        PoseNormalize()
    ])

    train_dataset = VSLPoseDataset(root_dir=args.DATA_PATH, split='train', transform=train_transforms)
    val_dataset = VSLPoseDataset(root_dir=args.DATA_PATH, split='val', transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=args.BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.BATCH_SIZE, shuffle=False, drop_last=False, num_workers=4)
    print(f"Loader dataset ready!")
    print(f"Preparing models...")
    model = VSLContrastiveNet(vocab_size=args.VOCAB_SIZE, embedding_size=args.EMBEDDING_SIZE).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.LR), weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.EPOCHS)
    criterion = SupervisedContrastiveLoss().to(device)
    print(f"Preparing models successfully!")
    if not os.path.exists(os.path.join(args.CHECKPOINT)): os.makedirs(os.path.join(args.CHECKPOINT))

    start_epoch = 0
    best_val_r1 = 0.0
    if os.path.exists(os.path.join(args.CHECKPOINT, 'lasted.pth')) and args.RESUME:
        print(f"Resuming training...")
        print(f"Loading checkpoint from {os.path.join(args.CHECKPOINT, 'lasted.pth')}")
        checkpoint = torch.load(os.path.join(args.CHECKPOINT, 'lasted.pth'), map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_r1 = checkpoint['best_val_r1']
        print("Load checkpoint successfully!")

    print(f"Start training...")
    for epoch in range(start_epoch, args.EPOCHS):
        model.train()
        train_loss = 0.0
        train_r1 = 0.0

        pbar_train = tqdm(train_loader, desc=f"[Training] Epoch {epoch}/{args.EPOCHS}")
        for pose, label in pbar_train:
            videos = pose.to(device)
            labels = label.to(device)
            optimizer.zero_grad()
            logits_v, logits_t = model(videos, labels)
            loss = criterion(logits_v, logits_t, labels)
            metrics = calculate_metrics(logits_v, logits_t)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_r1 += metrics['V2T_R1']

            pbar_train.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{metrics['V2T_R1'] * 100:.1f}%"
            })

        avg_train_loss = train_loss / len(train_loader)
        avg_train_r1 = train_r1 / len(train_loader)
        model.eval()
        val_loss = 0.0
        val_metrics = {k: 0.0 for k in
                       ['V2T_R1', 'V2T_R5', 'V2T_R10', 'V2T_Rank', 'T2V_R1', 'T2V_R5', 'T2V_R10', 'T2V_Rank']}
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc="Val  ", leave=False)
            for pose, label in pbar_val:
                videos, labels = pose.to(device), label.to(device)

                logits_v, logits_t = model(videos, labels)
                loss = criterion(logits_v, logits_t, labels)
                metrics = calculate_metrics(logits_v, logits_t)

                val_loss += loss.item()
                for k in metrics:
                    val_metrics[k] += metrics[k]

                pbar_val.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{metrics['V2T_R1'] * 100:.1f}%"
                })

        num_val_batches = len(val_loader)
        avg_val_loss = val_loss / num_val_batches
        for k in val_metrics:
            val_metrics[k] /= num_val_batches

        scheduler.step()
        print(f"[Train] Loss: {avg_train_loss:.4f} | Acc/R@1: {avg_train_r1 * 100:.2f}%")
        print(f"[Val]  Loss: {avg_val_loss:.4f} | Acc/R@1: {val_metrics['V2T_R1'] * 100:.2f}%")
        print(f"[Metrics] R@5: {val_metrics['V2T_R5'] * 100:.1f}% | R@10: {val_metrics['V2T_R10'] * 100:.1f}% | MeanRank: {val_metrics['V2T_Rank']:.2f}")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_r1': best_val_r1
        }, os.path.join(args.CHECKPOINT, 'lasted.pth'))

        if val_metrics['V2T_R1'] > best_val_r1:
            best_val_r1 = val_metrics['V2T_R1']
            torch.save(model.state_dict(), os.path.join(args.CHECKPOINT, 'best.pth'))
            print(f"[+] Accuracy improved! Saved new best model to '{os.path.join(args.CHECKPOINT, 'best.pth')}' (Acc: {best_val_r1 * 100:.2f}%)")