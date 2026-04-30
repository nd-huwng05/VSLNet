import os

import pandas as pd
import torch
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose
from tqdm import tqdm
import torch.nn.functional as F
from dataset.data_preprocessing import RandomTemporalCrop, RandomPoseScale, RandomPoseNoise, TemporalInterpolatePose, \
    PoseJoinSelect, PoseNormalize
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

    train_loader = DataLoader(train_dataset, batch_size=args.BATCH_SIZE, shuffle=True, drop_last=True, num_workers=args.WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=args.BATCH_SIZE, shuffle=False, drop_last=False, num_workers=args.WORKERS)
    print(f"Loader dataset ready!")
    print(f"Preparing models...")
    model = VSLContrastiveNet(vocab_size=args.VOCAB_SIZE, embedding_size=args.EMBEDDING_SIZE).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.LR), weight_decay=0.05)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=5)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=args.EPOCHS - 5, eta_min=1e-5)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[5])
    criterion = SupervisedContrastiveLoss().to(device)
    print(f"Preparing models successfully!")
    if not os.path.exists(os.path.join(args.CHECKPOINT)): os.makedirs(os.path.join(args.CHECKPOINT))

    log_dir = os.path.join(args.TENSORBOARD)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard log directory set to: {log_dir}")

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
        train_conf = 0.0

        pbar_train = tqdm(train_loader, desc=f"[Training] Epoch {epoch}/{args.EPOCHS}")
        for pose, label in pbar_train:
            videos = pose.to(device)
            labels = label.to(device)
            optimizer.zero_grad()
            logits_v, logits_t = model(videos, labels)
            loss = criterion(logits_v, logits_t, labels)
            metrics = calculate_metrics(logits_v, logits_t, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_r1 += metrics['V2T_R1']

            probs_v = F.softmax(logits_v, dim=1)
            mask = torch.eq(labels.view(-1, 1), labels.view(1, -1)).float()
            conf = (probs_v * mask).sum(dim=1).mean().item()
            train_conf += conf

            pbar_train.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{metrics['V2T_R1'] * 100:.1f}%",
                'conf': f"{conf * 100:.1f}%"
            })

        avg_train_loss = train_loss / len(train_loader)
        avg_train_r1 = train_r1 / len(train_loader)
        avg_train_conf = train_conf / len(train_loader)

        model.eval()
        val_loss = 0.0
        val_conf = 0.0
        csv_data = []
        val_metrics = {k: 0.0 for k in
                       ['V2T_R1', 'V2T_R5', 'V2T_R10', 'V2T_Rank', 'T2V_R1', 'T2V_R5', 'T2V_R10', 'T2V_Rank']}
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc="[Validating]", leave=False)
            all_text_indices = torch.arange(args.VOCAB_SIZE).to(device)
            text_embeddings_full = model.text_encoder(all_text_indices)
            text_embeddings_full = F.normalize(text_embeddings_full, p=2, dim=-1)
            for pose, label in pbar_val:
                videos, labels = pose.to(device), label.to(device)

                logits_v, logits_t = model(videos, labels)
                loss = criterion(logits_v, logits_t, labels)
                metrics = calculate_metrics(logits_v, logits_t, labels)

                val_loss += loss.item()
                for k in metrics:
                    val_metrics[k] += metrics[k]

                probs_v = F.softmax(logits_v, dim=1)
                mask = torch.eq(labels.view(-1, 1), labels.view(1, -1)).float()
                conf = (probs_v * mask).sum(dim=1).mean().item()
                val_conf += conf

                video_embedding = model.video_encoder(videos)
                video_embedding = F.normalize(video_embedding, p=2, dim=-1)

                logit_scale = model.logit_scale.exp()
                similarities = video_embedding @ text_embeddings_full.T
                logits_full = logit_scale * similarities
                probs_full = F.softmax(logits_full, dim=-1)

                max_probs, preds = probs_full.max(dim=-1)
                for i in range(len(labels)):
                    true_lbl = labels[i].item()
                    pred_lbl = preds[i].item()
                    global_conf = max_probs[i].item()
                    is_correct = bool(true_lbl == pred_lbl)

                    csv_data.append({
                        'LABEL': true_lbl,
                        'PREDICTED': pred_lbl,
                        'CONFIDENCE': round(global_conf, 4),
                        'CORRECT': is_correct
                    })

                pbar_val.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{metrics['V2T_R1'] * 100:.1f}%",
                    'conf': f"{conf * 100:.1f}%"
                })

        num_val_batches = len(val_loader)
        avg_val_loss = val_loss / num_val_batches
        avg_val_conf = val_conf / num_val_batches
        for k in val_metrics:
            val_metrics[k] /= num_val_batches

        df_results = pd.DataFrame(csv_data)
        csv_out_path = os.path.join(args.CHECKPOINT, 'val_confidence_analysis.csv')
        df_results.to_csv(csv_out_path, index=False)
        avg_conf_wrong = df_results[df_results['CORRECT'] == False]['CONFIDENCE'].mean()

        scheduler.step()
        print(
            f"[Train] Loss: {avg_train_loss:.4f} | Acc/R@1: {avg_train_r1 * 100:.2f}% | Conf: {avg_train_conf * 100:.2f}%")
        print(
            f"[Val]  Loss: {avg_val_loss:.4f} | Acc/R@1: {val_metrics['V2T_R1'] * 100:.2f}% | Conf: {avg_val_conf * 100:.2f}% | Conf wrong {avg_conf_wrong * 100:.2f}%\n" if pd.notna(
                avg_conf_wrong) else "[-] Global Conf (Wrong)   : N/A\n")
        print(
            f"[Metrics] R@5: {val_metrics['V2T_R5'] * 100:.1f}% | R@10: {val_metrics['V2T_R10'] * 100:.1f}% | MeanRank: {val_metrics['V2T_Rank']:.2f}")

        writer.add_scalars('Loss', {'Train': avg_train_loss, 'Val': avg_val_loss}, epoch)
        writer.add_scalars('Accuracy_R1', {'Train': avg_train_r1 * 100, 'Val': val_metrics['V2T_R1'] * 100}, epoch)
        writer.add_scalars('Confidence', {'Train': avg_train_conf * 100, 'Val': avg_val_conf * 100}, epoch)
        writer.add_scalar('Metrics/Val_R5', val_metrics['V2T_R5'] * 100, epoch)
        writer.add_scalar('Metrics/Val_R10', val_metrics['V2T_R10'] * 100, epoch)
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Hyperparameters/LearningRate', current_lr, epoch)

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
            print(
                f"[+] Accuracy improved! Saved new best model to '{os.path.join(args.CHECKPOINT, 'best.pth')}' (Acc: {best_val_r1 * 100:.2f}%)")
    writer.close()
