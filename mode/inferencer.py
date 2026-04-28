import os
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

    checkpoint_path = os.path.join(args.CHECKPOINT, "best.pth")
    assert os.path.exists(checkpoint_path)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print("[+] Load weights successfully!")
    model.eval()

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
                print("Nothing to predict")
                return JSONResponse(
                    status_code=400,
                    content={"status": "error", "message": "Nothing to predict!"}
                )

            data = await file.read()
            try:
                pose_obj = Pose.read(data)
            except Exception as e:
                print(f"Can't read file: {e}")
                raise e

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

                pred_index_val = pred_idx.item()
                predicted_word = vocab_dict.get(pred_index_val, str(pred_index_val))

                response = {
                    "status": "success",
                    "action": predicted_word,
                    "confidence": float(conf.item()),
                    "message": "Predict successfully!"
                }
                print(response)
                return response

        except Exception as e:
            import traceback
            traceback.print_exc()
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": f"Can't parse pose file: {e}"}
            )

    uvicorn.run(app, host="0.0.0.0", port=8000)