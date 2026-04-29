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
    const [poseReady, setPoseReady] = useState(false);

    const mediaRecorderRef = useRef(null);
    const isRecording = useRef(false);
    const silenceCounter = useRef(0);
    const prevFrameData = useRef(null);

    const chunksRef = useRef([]);

    const MOTION_THRESHOLD = 12;
    const SILENCE_FRAMES = 50;

    const sendVideoToServer = (videoBlob) => {
        if (!videoBlob || videoBlob.size === 0) return;

        const formData = new FormData();
        formData.append("file", videoBlob, "video_stream.webm");

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
    };

    const startRecording = () => {
        if (!streamActive || mode === 'upload') return;

        let streamToRecord = null;
        if (mode === 'screen' && canvasRef.current) {
            streamToRecord = canvasRef.current.captureStream(60);
        } else if (videoRef.current && videoRef.current.srcObject) {
            streamToRecord = videoRef.current.srcObject;
        }

        if (streamToRecord) {
            try {
                const recorder = new MediaRecorder(streamToRecord, { mimeType: 'video/webm' });
                chunksRef.current = [];

                recorder.ondataavailable = (e) => {
                    if (e.data && e.data.size > 0) {
                        chunksRef.current.push(e.data);
                    }
                };

                recorder.onstop = () => {
                    const videoBlob = new Blob(chunksRef.current, { type: 'video/webm' });
                    if (videoBlob.size > 15000) {
                        sendVideoToServer(videoBlob);
                    }
                    chunksRef.current = [];
                };

                recorder.start(500);
                mediaRecorderRef.current = recorder;
                isRecording.current = true;
            } catch (err) {}
        }
    };

    const stopRecordingAndSend = () => {
        if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
            mediaRecorderRef.current.stop();
        }
        isRecording.current = false;
        silenceCounter.current = 0;
    };

    useEffect(() => {
        setPoseReady(true);
    }, []);

    useEffect(() => {
        let animationId;

        const diffCanvas = document.createElement('canvas');
        diffCanvas.width = 64;
        diffCanvas.height = 48;
        const diffCtx = diffCanvas.getContext('2d', { willReadFrequently: true });

        const processFrame = () => {
            if (videoRef.current && videoRef.current.readyState >= 2) {
                const video = videoRef.current;

                if (mode === 'screen' && streamActive && canvasRef.current) {
                    const rect = video.getBoundingClientRect();
                    const scaleX = video.videoWidth / rect.width;
                    const scaleY = video.videoHeight / rect.height;
                    const ctx = canvasRef.current.getContext('2d');

                    const newW = cropBox.width * scaleX;
                    const newH = cropBox.height * scaleY;

                    if (canvasRef.current.width !== newW) canvasRef.current.width = newW;
                    if (canvasRef.current.height !== newH) canvasRef.current.height = newH;

                    ctx.drawImage(video, cropBox.x * scaleX, cropBox.y * scaleY, newW, newH, 0, 0, newW, newH);
                }

                if (streamActive && mode !== 'upload') {
                    diffCtx.drawImage(mode === 'screen' ? canvasRef.current : video, 0, 0, 64, 48);
                    const currentData = diffCtx.getImageData(0, 0, 64, 48).data;

                    if (prevFrameData.current) {
                        let diffSum = 0;
                        for (let i = 0; i < currentData.length; i += 4) {
                            diffSum += Math.abs(currentData[i] - prevFrameData.current[i]) +
                                       Math.abs(currentData[i+1] - prevFrameData.current[i+1]) +
                                       Math.abs(currentData[i+2] - prevFrameData.current[i+2]);
                        }

                        const avgDiff = diffSum / (64 * 48 * 3);

                        if (avgDiff > MOTION_THRESHOLD) {
                            if (!isRecording.current) {
                                startRecording();
                            }
                            silenceCounter.current = 0;
                        } else {
                            if (isRecording.current) {
                                silenceCounter.current += 1;
                                if (silenceCounter.current > SILENCE_FRAMES) {
                                    stopRecordingAndSend();
                                }
                            }
                        }
                    }
                    prevFrameData.current = new Uint8ClampedArray(currentData);
                }
            }
            animationId = requestAnimationFrame(processFrame);
        };

        processFrame();
        return () => cancelAnimationFrame(animationId);
    }, [cropBox, mode, streamActive]);

    const stopActiveStream = () => {
        stopRecordingAndSend();
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
        prevFrameData.current = null;
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
            }).catch(err => {});

            sendVideoToServer(file);
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
                        {mode === 'camera' && <button onClick={async () => { stopActiveStream(); const s = await navigator.mediaDevices.getUserMedia({video: { frameRate: { ideal: 60 } }}); videoRef.current.srcObject = s; setStreamActive(true); }} className="w-full py-2.5 bg-zinc-100 text-zinc-900 rounded-lg text-sm font-semibold">Khởi động Camera</button>}
                        {mode === 'screen' && <button onClick={async () => { stopActiveStream(); const s = await navigator.mediaDevices.getDisplayMedia({video: { frameRate: { ideal: 60 } }}); videoRef.current.srcObject = s; setStreamActive(true); }} className="w-full py-2.5 bg-zinc-100 text-zinc-900 rounded-lg text-sm font-semibold">Chọn Nguồn Quét</button>}
                        {mode === 'upload' && <input type="file" accept="video/*" onChange={handleVideoUpload} className="w-full text-sm text-zinc-400 file:bg-zinc-800 file:text-zinc-200 file:rounded-lg file:border-0 file:px-4 file:py-2" />}
                    </div>
                </aside>

                <section className="flex-1 bg-black relative flex items-center justify-center overflow-hidden">
                    <video ref={videoRef} autoPlay loop={mode !== 'upload'} muted playsInline onEnded={() => { if (mode === 'upload') { setIsVideoEnded(true); } }} className={`absolute inset-0 w-full h-full object-contain pointer-events-none ${streamActive ? 'opacity-100' : 'opacity-0'}`} />
                    {streamActive && mode === 'upload' && isVideoEnded && (
                        <div className="absolute inset-0 z-40 flex items-center justify-center bg-black/40 backdrop-blur-sm transition-all text-center">
                            <button onClick={() => { setIsVideoEnded(false); if (videoRef.current) { videoRef.current.currentTime = 0; videoRef.current.play(); } }} className="px-6 py-3 bg-zinc-100 text-zinc-900 rounded-full text-sm font-semibold shadow-xl hover:scale-105 transition-all">Phát lại video</button>
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