import os
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from tqdm import tqdm
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
    test_metrics = {k: 0.0 for k in
                    ['V2T_R1', 'V2T_R5', 'V2T_R10', 'V2T_Rank', 'T2V_R1', 'T2V_R5', 'T2V_R10', 'T2V_Rank']}

    with torch.no_grad():
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

            pbar_test.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{metrics['V2T_R1'] * 100:.1f}%"
            })

    num_test_batches = len(test_loader)
    avg_test_loss = test_loss / num_test_batches
    for k in test_metrics:
        test_metrics[k] /= num_test_batches

    print("\n" + "=" * 50)
    print(" " * 17 + "TEST RESULTS")
    print("=" * 50)
    print(f"Average Loss: {avg_test_loss:.4f}\n")

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