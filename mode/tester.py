import os

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from tqdm import tqdm
import torch.nn.functional as F
from dataset.data_preprocessing import TemporalInterpolatePose, PoseJoinSelect, PoseNormalize
from dataset.vsl_dataset import VSLPoseDataset
from models.metrics import SupervisedContrastiveLoss, calculate_metrics
from models.vsl_net import VSLContrastiveNet


def test(args):
    print("Mode testing...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Setting device {device} successfully!")

    print("Preparing test data loader...")
    test_transforms = Compose([
        PoseJoinSelect(),
        TemporalInterpolatePose(frames=64),
        PoseNormalize()
    ])

    test_dataset = VSLPoseDataset(root_dir=args.DATA_PATH, split='test', transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=args.BATCH_SIZE, shuffle=False, drop_last=False, num_workers=4)
    print("Test loader ready!")

    print("Preparing model for testing...")
    model = VSLContrastiveNet(vocab_size=args.VOCAB_SIZE, embedding_size=args.EMBEDDING_SIZE).to(device)
    criterion = SupervisedContrastiveLoss().to(device)

    best_model_path = os.path.join(args.CHECKPOINT, 'best.pth')
    if os.path.exists(best_model_path):
        print(f"Loading best checkpoint from {best_model_path}")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print("Loaded checkpoint successfully!")
    else:
        print(f"[!] Warning: No checkpoint found at {best_model_path}. Evaluating with random weights.")

    print("Start testing...")
    model.eval()
    test_loss = 0.0
    test_conf = 0.0
    csv_data = []
    test_metrics = {k: 0.0 for k in
                    ['V2T_R1', 'V2T_R5', 'V2T_R10', 'V2T_Rank', 'T2V_R1', 'T2V_R5', 'T2V_R10', 'T2V_Rank']}

    with torch.no_grad():
        all_text_indices = torch.arange(args.VOCAB_SIZE).to(device)
        text_embeddings_full = model.text_encoder(all_text_indices)
        text_embeddings_full = F.normalize(text_embeddings_full, p=2, dim=-1)

        pbar_test = tqdm(test_loader, desc="[Testing]")
        for pose, label in pbar_test:
            videos = pose.to(device)
            labels = label.to(device)

            logits_v, logits_t = model(videos, labels)
            loss = criterion(logits_v, logits_t, labels)
            metrics = calculate_metrics(logits_v, logits_t, labels)

            test_loss += loss.item()
            for k in metrics:
                test_metrics[k] += metrics[k]

            probs_v = F.softmax(logits_v, dim=1)
            mask = torch.eq(labels.view(-1, 1), labels.view(1, -1)).float()
            conf = (probs_v * mask).sum(dim=1).mean().item()
            test_conf += conf

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

            pbar_test.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{metrics['V2T_R1'] * 100:.1f}%",
                'conf': f"{conf * 100:.1f}%"
            })

    num_test_batches = len(test_loader)
    avg_test_loss = test_loss / num_test_batches
    avg_test_conf = test_conf / num_test_batches
    for k in test_metrics:
        test_metrics[k] /= num_test_batches

    df_results = pd.DataFrame(csv_data)
    csv_out_path = os.path.join(args.CHECKPOINT, 'test_confidence_analysis.csv')
    df_results.to_csv(csv_out_path, index=False)

    avg_conf_correct = df_results[df_results['CORRECT'] == True]['CONFIDENCE'].mean()
    avg_conf_wrong = df_results[df_results['CORRECT'] == False]['CONFIDENCE'].mean()

    print("\n" + "=" * 50)
    print(" " * 17 + "TEST RESULTS")
    print("=" * 50)
    print(f"Average Loss: {avg_test_loss:.4f}")
    print(f"Batch Confidence: {avg_test_conf * 100:.2f}%\n")  # <--- In ra đúng như code cũ
    print(f"[-] Saved Confidence CSV to: {csv_out_path}")
    print(f"[-] Global Conf (Correct) : {avg_conf_correct * 100:.2f}%" if pd.notna(
        avg_conf_correct) else "[-] Global Conf (Correct) : N/A")
    print(f"[-] Global Conf (Wrong)   : {avg_conf_wrong * 100:.2f}%\n" if pd.notna(
        avg_conf_wrong) else "[-] Global Conf (Wrong)   : N/A\n")

    print("--- Video to Text Retrieval (V2T) ---")
    print(f"Accuracy (R@1) : {test_metrics['V2T_R1'] * 100:.2f}%")
    print(f"Recall@5       : {test_metrics['V2T_R5'] * 100:.2f}%")
    print(f"Recall@10      : {test_metrics['V2T_R10'] * 100:.2f}%")
    print(f"Mean Rank      : {test_metrics['V2T_Rank']:.2f}\n")

    print("--- Text to Video Retrieval (T2V) ---")
    print(f"Accuracy (R@1) : {test_metrics['T2V_R1'] * 100:.2f}%")
    print(f"Recall@5       : {test_metrics['T2V_R5'] * 100:.2f}%")
    print(f"Recall@10      : {test_metrics['T2V_R10'] * 100:.2f}%")
    print(f"Mean Rank      : {test_metrics['T2V_Rank']:.2f}")
    print("=" * 50)

    return test_metrics
