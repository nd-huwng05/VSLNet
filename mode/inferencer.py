import os
import sys
import runpy
import shutil
import tempfile
import mediapipe as mp
import pandas as pd
import torch
import torch.nn.functional as F
import uvicorn
import subprocess
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pose_format import Pose
from torchvision.transforms import Compose
from dataset.data_preprocessing import PoseJoinSelect, TemporalInterpolatePose, PoseNormalize
from models.vsl_net import VSLContrastiveNet

VIDEO_TO_POSE_SCRIPT = shutil.which('video_to_pose')


def run_frontend():
    try:
        subprocess.run("npm run dev", shell=True, cwd="./demo")
    except Exception as e:
        print(f"Lỗi khởi động Frontend: {e}")


def inference(args):
    # frontend_thread = threading.Thread(target=run_frontend, daemon=True)
    # frontend_thread.start()
    # time.sleep(2)
    # webbrowser.open("http://localhost:5173")

    app = FastAPI(title="VSLNet Real-time API")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    device = torch.device("cuda" if torch.cuda.is_available() and args.GPU else "cpu")

    model = VSLContrastiveNet(
        vocab_size=args.VOCAB_SIZE,
        embedding_size=args.EMBEDDING_SIZE
    ).to(device)

    checkpoint_path = os.path.join(args.CHECKPOINT, "best_v1.pth")
    assert os.path.exists(checkpoint_path)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print("[+] Load weights successfully!")
    model.eval()

    mp_holistic = mp.solutions.holistic
    holistic_model = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        refine_face_landmarks=True
    )

    assert os.path.exists(os.path.join(args.DATA_PATH, 'gloss.csv'))
    gloss_df = pd.read_csv(os.path.join(args.DATA_PATH, 'gloss.csv'))
    vocab_dict = dict(zip(gloss_df.index, gloss_df.iloc[:, 1]))
    print("[+] Load vocabulary successfully!")

    with torch.no_grad():
        labels_idx = torch.arange(len(vocab_dict)).to(device)
        text_embeddings = model.text_encoder(labels_idx)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)

    inference_transforms = Compose([
        PoseJoinSelect(),
        TemporalInterpolatePose(frames=args.FRAMES),
        PoseNormalize()
    ])

    @app.post("/predict")
    async def predict(req: Request):
        try:
            form_data = await req.form()
            file = form_data.get("file", None)

            if not file:
                return JSONResponse(status_code=400, content={"status": "error", "message": "No receive file video!"})

            with tempfile.TemporaryDirectory() as tmpdir:
                # 1. Lưu file .webm raw do trình duyệt gửi lên
                raw_video_path = os.path.join(tmpdir, "raw_video.webm")
                video_tmp_path = os.path.join(tmpdir, "input_video.mp4")
                pose_tmp_path = os.path.join(tmpdir, "output.pose")

                with open(raw_video_path, "wb") as f:
                    f.write(await file.read())

                try:
                    subprocess.run(
                        ['ffmpeg', '-y', '-i', '-filter:v', 'fps=30', '-c:v', 'libx264', '-preset', 'ultrafast',
                         video_tmp_path],
                        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                    )
                except Exception:
                    video_tmp_path = raw_video_path

                if not VIDEO_TO_POSE_SCRIPT:
                    return JSONResponse(status_code=500,
                                        content={"status": "error", "message": "video_to_pose script not found"})

                old_argv = sys.argv
                sys.argv = ['video_to_pose', '-i', video_tmp_path, '-o', pose_tmp_path, '--format', 'mediapipe']

                try:
                    runpy.run_path(VIDEO_TO_POSE_SCRIPT, run_name="__main__")
                except SystemExit as e:
                    if e.code != 0 and e.code is not None:
                        return JSONResponse(status_code=500,
                                            content={"status": "error", "message": "Error extract file"})
                finally:
                    sys.argv = old_argv

                if not os.path.exists(pose_tmp_path):
                    return {
                        "status": "success",
                        "action": "...",
                        "confidence": 0.0,
                        "message": "Video quá ngắn, bỏ qua."
                    }

                with open(pose_tmp_path, "rb") as f:
                    pose_data_bytes = f.read()
                    pose_obj = Pose.read(pose_data_bytes)

            pose_data = inference_transforms(pose_obj)
            pose_data = pose_data.reshape(pose_data.size(0), -1)
            pose_data = pose_data.unsqueeze(0).to(device)

            with torch.no_grad():
                video_embedding = model.video_encoder(pose_data)
                video_embedding = F.normalize(video_embedding, p=2, dim=-1)

                logit_scale = model.logit_scale.exp()
                logits = logit_scale * (video_embedding @ text_embeddings.T)
                probs = F.softmax(logits, dim=-1)

                conf, pred_idx = torch.max(probs, dim=1)
                predicted_word = vocab_dict.get(pred_idx.item(), "Unknown")

                return {
                    "status": "success",
                    "action": predicted_word,
                    "confidence": float(conf.item()),
                    "message": "Predict successfully!"
                }

        except Exception as e:
            return JSONResponse(status_code=500, content={"status": "error", "message": f"Error system: {str(e)}"})

    uvicorn.run(app, host="0.0.0.0", port=8000)