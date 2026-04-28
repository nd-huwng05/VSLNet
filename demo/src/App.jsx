import {useRef, useEffect, useState} from 'react';
import {Rnd} from 'react-rnd';
import axios from 'axios';

function App() {
    const [mode, setMode] = useState('camera');
    const [streamActive, setStreamActive] = useState(false);
    const [transcript, setTranscript] = useState("");
    const [isVideoEnded, setIsVideoEnded] = useState(false);
    const lastWord = useRef("");
    const modeRef = useRef('camera');
    const [cropBox, setCropBox] = useState({x: 50, y: 50, width: 350, height: 250});

    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const holistic = useRef(null);
    const [poseReady, setPoseReady] = useState(false);

    const poseBuffer = useRef([]);
    const prevPose = useRef(null);
    const isRecording = useRef(false);
    const silenceCounter = useRef(0);
    const isProcessingRef = useRef(false);

    const MOTION_THRESHOLD = 0.000015;
    const SILENCE_FRAMES = 15;
    const MIN_FRAMES = 0;

    const sendToServer = async (frames) => {
        if (!frames || frames.length < MIN_FRAMES) return;

        const PHeader = window.PoseHeader || window.dtpose?.PoseHeader;
        const PBody = window.PoseBody || window.dtpose?.PoseBody;
        const DPose = window.DTPose || window.dtpose?.DTPose;

        if (!PHeader || !PBody || !DPose) {
            return;
        }

        try {
            const handPoints = [
                "WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
                "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
                "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
                "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
                "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"
            ];

            const header = new PHeader({
                version: 0.1,
                dimensions: {width: 640, height: 480, depth: 1000},
                components: [
                    {
                        name: "POSE_LANDMARKS",
                        points: [
                            "NOSE", "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER",
                            "RIGHT_EYE_INNER", "RIGHT_EYE", "RIGHT_EYE_OUTER",
                            "LEFT_EAR", "RIGHT_EAR", "MOUTH_LEFT", "MOUTH_RIGHT",
                            "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
                            "LEFT_WRIST", "RIGHT_WRIST", "LEFT_PINKY", "RIGHT_PINKY",
                            "LEFT_INDEX", "RIGHT_INDEX", "LEFT_THUMB", "RIGHT_THUMB",
                            "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE",
                            "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL",
                            "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"
                        ],
                        limbs: [], colors: [], format: "XYZc"
                    },
                    {
                        name: "LEFT_HAND_LANDMARKS",
                        points: handPoints,
                        limbs: [], colors: [], format: "XYZc"
                    },
                    {
                        name: "RIGHT_HAND_LANDMARKS",
                        points: handPoints,
                        limbs: [], colors: [], format: "XYZc"
                    }
                ]
            });

            const fps = 30;
            const numFrames = frames.length;
            const totalPoints = 75;
            const dataArray = new Float32Array(numFrames * totalPoints * 3);
            const confArray = new Float32Array(numFrames * totalPoints);

            let dataIdx = 0;
            let confIdx = 0;

            frames.forEach(frame => {
                for (let i = 0; i < 33; i++) {
                    const point = frame.pose[i] || {x: 0, y: 0, z: 0, visibility: 0};
                    dataArray[dataIdx++] = point.x;
                    dataArray[dataIdx++] = point.y;
                    dataArray[dataIdx++] = point.z || 0;
                    confArray[confIdx++] = point.visibility || 1.0;
                }
                for (let i = 0; i < 21; i++) {
                    const point = (frame.left && frame.left[i]) ? frame.left[i] : {x: 0, y: 0, z: 0};
                    dataArray[dataIdx++] = point.x;
                    dataArray[dataIdx++] = point.y;
                    dataArray[dataIdx++] = point.z || 0;
                    confArray[confIdx++] = (frame.left && frame.left[i]) ? 1.0 : 0.0;
                }
                for (let i = 0; i < 21; i++) {
                    const point = (frame.right && frame.right[i]) ? frame.right[i] : {x: 0, y: 0, z: 0};
                    dataArray[dataIdx++] = point.x;
                    dataArray[dataIdx++] = point.y;
                    dataArray[dataIdx++] = point.z || 0;
                    confArray[confIdx++] = (frame.right && frame.right[i]) ? 1.0 : 0.0;
                }
            });

            const body = new PBody(fps, dataArray, confArray);
            const poseFile = new DPose(header, body);
            const binaryBuffer = await poseFile.write();
            const blob = new Blob([binaryBuffer], {type: 'application/octet-stream'});
            const formData = new FormData();
            formData.append("file", blob, "video_capture.pose");

            axios.post('http://localhost:8000/predict', formData, {
                headers: {'Content-Type': 'multipart/form-data'}
            })
            .then(res => {
                const {action, confidence} = res.data;
                if (confidence > 0.8 && action !== lastWord.current) {
                    setTranscript(prev => prev + (prev ? " " : "") + action);
                    lastWord.current = action;
                }
            })
            .catch(err => {});
        } catch (error) {}
    };

    useEffect(() => {
        const initHolistic = () => {
            if (window.Holistic) {
                holistic.current = new window.Holistic({
                    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`,
                });
                holistic.current.setOptions({
                    modelComplexity: 1,
                    smoothLandmarks: true,
                    minDetectionConfidence: 0.5,
                    minTrackingConfidence: 0.5
                });
                holistic.current.onResults((results) => {
                    if (results.poseLandmarks) {
                        const frameData = {
                            pose: results.poseLandmarks,
                            left: results.leftHandLandmarks,
                            right: results.rightHandLandmarks
                        };

                        if (modeRef.current === 'upload') {
                            poseBuffer.current.push(frameData);
                        } else {
                            let motion = 0;
                            if (prevPose.current) {
                                [15, 16, 19, 20].forEach(p => {
                                    if (results.poseLandmarks[p] && prevPose.current[p]) {
                                        motion += Math.abs(results.poseLandmarks[p].x - prevPose.current[p].x) +
                                                  Math.abs(results.poseLandmarks[p].y - prevPose.current[p].y);
                                    }
                                });
                            }
                            prevPose.current = results.poseLandmarks;
                            if (motion > MOTION_THRESHOLD) {
                                if (!isRecording.current) {
                                    isRecording.current = true;
                                    poseBuffer.current = [];
                                }
                                silenceCounter.current = 0;
                            } else if (isRecording.current) {
                                silenceCounter.current += 1;
                                if (silenceCounter.current > SILENCE_FRAMES) {
                                    isRecording.current = false;
                                    sendToServer([...poseBuffer.current]);
                                    poseBuffer.current = [];
                                }
                            }
                            if (isRecording.current || silenceCounter.current > 0) {
                                poseBuffer.current.push(frameData);
                            }
                        }
                    }
                });
                setPoseReady(true);
            } else {
                setTimeout(initHolistic, 500);
            }
        };
        initHolistic();
    }, []);

    useEffect(() => {
        let animationId;
        const processFrame = async () => {
            if (isProcessingRef.current) {
                animationId = requestAnimationFrame(processFrame);
                return;
            }
            if (videoRef.current && videoRef.current.readyState >= 2 && holistic.current) {
                isProcessingRef.current = true;
                try {
                    const video = videoRef.current;
                    if (mode === 'screen' && streamActive && canvasRef.current) {
                        const rect = video.getBoundingClientRect();
                        const scaleX = video.videoWidth / rect.width;
                        const scaleY = video.videoHeight / rect.height;
                        const ctx = canvasRef.current.getContext('2d');
                        canvasRef.current.width = cropBox.width * scaleX;
                        canvasRef.current.height = cropBox.height * scaleY;
                        ctx.drawImage(video, cropBox.x * scaleX, cropBox.y * scaleY, canvasRef.current.width, canvasRef.current.height, 0, 0, canvasRef.current.width, canvasRef.current.height);
                        await holistic.current.send({image: canvasRef.current});
                    } else if (streamActive) {
                        await holistic.current.send({image: video});
                    }
                } catch (err) {}
                finally { isProcessingRef.current = false; }
            }
            animationId = requestAnimationFrame(processFrame);
        };
        processFrame();
        return () => cancelAnimationFrame(animationId);
    }, [cropBox, mode, streamActive]);

    const stopActiveStream = () => {
        if (videoRef.current) {
            if (videoRef.current.srcObject) {
                videoRef.current.srcObject.getTracks().forEach(t => t.stop());
                videoRef.current.srcObject = null;
            }
            if (videoRef.current.src && videoRef.current.src.startsWith('blob:')) {
                URL.revokeObjectURL(videoRef.current.src);
                videoRef.current.removeAttribute('src');
                videoRef.current.load();
            }
        }
    };

    const handleTabChange = (newMode) => {
        setMode(newMode);
        modeRef.current = newMode;
        stopActiveStream();
        setStreamActive(false);
        setIsVideoEnded(false);
        poseBuffer.current = [];
        isRecording.current = false;
        silenceCounter.current = 0;
    };

    const handleVideoUpload = (e) => {
        stopActiveStream();
        const file = e.target.files[0];
        if (file && videoRef.current) {
            videoRef.current.src = URL.createObjectURL(file);
            videoRef.current.load();
            videoRef.current.play().then(() => {
                setStreamActive(true);
                setIsVideoEnded(false);
                poseBuffer.current = [];
                isRecording.current = false;
                silenceCounter.current = 0;
            }).catch(err => {});
        }
    };

    return (
        <div className="fixed inset-0 overflow-hidden bg-zinc-950 text-zinc-300 flex flex-col font-sans antialiased">
            <header className="h-14 bg-zinc-950/80 backdrop-blur-md border-b border-white/5 px-6 flex items-center justify-between shrink-0 z-20">
                <div className="flex items-center gap-3">
                    <div className="w-6 h-6 rounded bg-gradient-to-tr from-indigo-500 to-cyan-400 flex items-center justify-center">
                        <div className="w-2 h-2 bg-white rounded-full"></div>
                    </div>
                    <h1 className="text-base font-semibold text-zinc-100 tracking-tight">VSLNet Workspace</h1>
                </div>
                <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-zinc-900 border border-white/5 text-xs font-medium">
                    <span className={`relative flex h-2 w-2 ${!poseReady ? 'animate-pulse' : ''}`}>
                        <span className={`relative inline-flex rounded-full h-2 w-2 ${!poseReady ? 'bg-amber-500' : 'bg-emerald-500'}`}></span>
                    </span>
                    {!poseReady ? 'Initializing Engine...' : 'Ready'}
                </div>
            </header>

            <main className="flex-1 flex flex-row w-full overflow-hidden">
                <aside className="w-[280px] bg-zinc-900/40 border-r border-white/5 flex flex-col shrink-0 z-10 backdrop-blur-sm">
                    <div className="px-5 py-4 text-[11px] font-semibold text-zinc-500 uppercase tracking-widest mt-2">Chế độ phân tích</div>
                    <div className="px-3 flex flex-col gap-1">
                        <button onClick={() => handleTabChange('camera')} className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all ${mode === 'camera' ? 'bg-zinc-800 text-zinc-100' : 'text-zinc-400 hover:bg-zinc-800/50'}`}>Camera Trực Tiếp</button>
                        <button onClick={() => handleTabChange('screen')} className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all ${mode === 'screen' ? 'bg-zinc-800 text-zinc-100' : 'text-zinc-400 hover:bg-zinc-800/50'}`}>Quét Màn Hình</button>
                        <button onClick={() => handleTabChange('upload')} className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all ${mode === 'upload' ? 'bg-zinc-800 text-zinc-100' : 'text-zinc-400 hover:bg-zinc-800/50'}`}>Tải Video Lên</button>
                    </div>
                    <div className="px-4 mt-auto mb-6">
                        {mode === 'camera' && <button onClick={async () => { stopActiveStream(); const s = await navigator.mediaDevices.getUserMedia({video: true}); videoRef.current.srcObject = s; setStreamActive(true); }} className="w-full py-2.5 bg-zinc-100 text-zinc-900 rounded-lg text-sm font-semibold">Khởi động Camera</button>}
                        {mode === 'screen' && <button onClick={async () => { stopActiveStream(); const s = await navigator.mediaDevices.getDisplayMedia({video: true}); videoRef.current.srcObject = s; setStreamActive(true); }} className="w-full py-2.5 bg-zinc-100 text-zinc-900 rounded-lg text-sm font-semibold">Chọn Nguồn Quét</button>}
                        {mode === 'upload' && <input type="file" accept="video/*" onChange={handleVideoUpload} className="w-full text-sm text-zinc-400 file:bg-zinc-800 file:text-zinc-200 file:rounded-lg file:border-0 file:px-4 file:py-2" />}
                    </div>
                </aside>

                <section className="flex-1 bg-black relative flex items-center justify-center overflow-hidden">
                    <video ref={videoRef} autoPlay loop={mode !== 'upload'} muted playsInline onEnded={() => { if (mode === 'upload') { sendToServer([...poseBuffer.current]); setIsVideoEnded(true); } }} className={`absolute inset-0 w-full h-full object-contain pointer-events-none ${streamActive ? 'opacity-100' : 'opacity-0'}`} />
                    {streamActive && mode === 'upload' && isVideoEnded && (
                        <div className="absolute inset-0 z-40 flex items-center justify-center bg-black/40 backdrop-blur-sm transition-all text-center">
                            <button onClick={() => { setIsVideoEnded(false); poseBuffer.current = []; isRecording.current = false; silenceCounter.current = 0; if (videoRef.current) { videoRef.current.currentTime = 0; videoRef.current.play(); } }} className="px-6 py-3 bg-zinc-100 text-zinc-900 rounded-full text-sm font-semibold shadow-xl hover:scale-105 transition-all">Phát lại video</button>
                        </div>
                    )}
                    {streamActive && mode === 'screen' && (
                        <Rnd position={{x: cropBox.x, y: cropBox.y}} size={{width: cropBox.width, height: cropBox.height}} onDragStop={(e, d) => setCropBox({...cropBox, x: d.x, y: d.y})} onResizeStop={(e, dir, ref, delta, pos) => setCropBox({ x: pos.x, y: pos.y, width: parseInt(ref.style.width), height: parseInt(ref.style.height) })} bounds="parent" className="border-[1.5px] border-rose-500/80 z-50 cursor-move">
                            <div className="absolute -top-6 left-0 bg-rose-500/90 text-white text-[10px] font-bold px-2 py-1 rounded-t-sm">SCAN AREA</div>
                        </Rnd>
                    )}
                    {!streamActive && <div className="text-zinc-600 text-sm font-medium">Signal Interrupted</div>}
                    <canvas ref={canvasRef} className="hidden" />
                </section>

                <aside className="w-[320px] bg-zinc-900/40 border-l border-white/5 flex flex-col shrink-0 z-10 backdrop-blur-sm">
                    <div className="px-5 py-4 border-b border-white/5 flex items-center justify-between">
                        <span className="text-[11px] font-semibold text-zinc-500 uppercase tracking-widest mt-2">Live Transcript</span>
                        {streamActive && <span className="h-2 w-2 rounded-full bg-rose-500 animate-pulse"></span>}
                    </div>
                    <div className="flex-1 p-6 overflow-y-auto font-medium text-zinc-200 tracking-tight text-xl">
                        {!transcript ? <span className="text-zinc-600 text-base font-normal">Listening for gestures...</span> : transcript}
                    </div>
                    <div className="px-4 mb-6">
                        <button onClick={() => { setTranscript(""); lastWord.current = ""; }} className="w-full py-2.5 bg-zinc-800/50 text-zinc-400 rounded-lg text-sm font-medium hover:text-zinc-200 transition-colors">Xóa lịch sử</button>
                    </div>
                </aside>
            </main>
        </div>
    );
}

export default App;