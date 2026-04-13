import json
import os
import cv2
import numpy
import pandas
import mediapipe
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def extract_keypoints(results):
    lh = numpy.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else numpy.zeros(63)
    rh = numpy.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else numpy.zeros(63)
    return numpy.concatenate((lh, rh))

def cut_down_data(video_path, model, frame_per_video=50):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_data = []

    # get data numpy for each video
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok: break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model.process(img_rgb)
        video_data.append(extract_keypoints(results))

    cap.release()
    raw_data = numpy.array(video_data)
    T = raw_data.shape[0] # (133, 126), num_frames = 133 , 126 = 3 * 42 (3 is x, y, z, 42 is points about hand)

    reshaped = raw_data.reshape(T, 42, 3)
    diff = numpy.diff(reshaped, axis=0) # (132, 42, 3) frame(t) - frame(t-1) -> 133 frame -> 132 frame diff
    motion = numpy.sum(numpy.linalg.norm(diff, axis=2), axis=1) # speed's video
    motion = numpy.insert(motion, 0, 0) # add 1 frame (132,) + 1 -> (133,)

    robust_max = numpy.percentile(motion, 80) # index = q/100 * (n-1), q=80
    motion = numpy.clip(motion, 0, robust_max)

    window_size = 5
    smoothed_motion = numpy.convolve(motion, numpy.ones(window_size) / window_size, mode='same')

    threshold = numpy.max(smoothed_motion)*0.1
    active_frames = numpy.where(smoothed_motion > threshold)[0]

    if len(active_frames) == 0:
        return []

    gap_threshold = int(fps*0.3)
    split_points = numpy.where(numpy.diff(active_frames) > gap_threshold)[0] + 1
    segments_indices = numpy.split(active_frames, split_points)

    extract_segments = []

    for seg in segments_indices:
        start_idx = max(0, seg[0] - 2)
        end_idx = min(T-1, seg[-1] + 2)

        core_data = raw_data[start_idx:end_idx + 1]
        L = core_data.shape[0]

        indices = numpy.linspace(0, L-1, frame_per_video, dtype=int)
        normalized_segments = core_data[indices]
        extract_segments.append(normalized_segments)

    return extract_segments

def augment_data(base_sample):
    augmented = base_sample.copy()
    aug_type = numpy.random.choice(["noise", "shift", "scale"])

    if aug_type == "noise":
        noise = numpy.random.normal(0, 0.003, augmented.shape)
        augmented += noise # x' = x + epsilon, with epsilon ~ N(0, sigma2), gaussian
    elif aug_type == "shift":
        shift_val = numpy.random.uniform(-0.05, 0.05, (1, 126))
        shift_val[0, 2::3] = 0 # x' = x + s, s ~ U(-0.05, 0.05)
        augmented += shift_val
    elif aug_type == "scale":
        scale = numpy.random.uniform(0.9, 1.1)
        augmented *= scale # x' = x * alpha, alpha ~ U(0.9, 1.1)

    return augmented


def prepare_dataset(args):
    label_path = os.path.join(args.dataset, 'labels/label.csv')
    df = pandas.read_csv(label_path)

    # create directory numpy dataset and label
    os.makedirs(os.path.join(args.data, 'numpy'), exist_ok=True)
    os.makedirs(os.path.join(args.data, 'label'), exist_ok=True)
    print(f'Create directory {args.data}')

    unique_labels = df['LABEL'].unique()
    id_to_label = {int(i): text for i, text in enumerate(unique_labels)}
    label_to_id = {text: int(i) for i, text in enumerate(unique_labels)}

    # create labels.json
    with open(os.path.join(args.data, 'label', 'labels.json'), 'w', encoding='utf-8') as f:
        json.dump(id_to_label, f, ensure_ascii=False, indent=4)

    print(f'Prepare dataset ...')
    mp_holistic = mediapipe.solutions.holistic

    with mp_holistic.Holistic(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic_model:
        for label in tqdm(unique_labels, desc='Prepare dataset'):
            label_id = label_to_id[label]
            video_file = df[df['LABEL'] == label]['VIDEO'].tolist()

            label_matrices = []
            for name in video_file:
                path = os.path.join(args.dataset, 'videos', name)
                if not os.path.exists(path):
                    continue

                segments = cut_down_data(path, holistic_model, frame_per_video=args.frames)
                if not segments: continue
                for seg in segments: label_matrices.append(seg)

            original_count = len(label_matrices)
            if original_count > 0:
                current_count = original_count
                for i in range(current_count, args.num_data_label + 1):
                    base_sample = label_matrices[i%original_count]
                    fake_sample = augment_data(base_sample)
                    label_matrices.append(fake_sample)

                final_array = numpy.array(label_matrices)
                save_path = os.path.join(args.data, 'numpy', f'{label_id}.npy')
                numpy.save(save_path, final_array)