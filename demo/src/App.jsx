import { useRef, useEffect, useState } from 'react';
import { Rnd } from 'react-rnd';
import axios from 'axios';

const Pose = window.Pose;

function App() {
    const [mode, setMode] = useState('camera');
    const [streamActive, setStreamActive] = useState(false);
    const [transcript, setTranscript] = useState("");
    const lastWord = useRef("");
    const [cropBox, setCropBox] = useState({ x: 50, y: 50, width: 350, height: 250 });

    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const poseBuffer = useRef([]);
    const pose = useRef(null);
    const [poseReady, setPoseReady] = useState(false);

    useEffect(() => {
        const initPose = () => {
            if (window.Pose) {
                pose.current = new window.Pose({
                    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`,
                });
                pose.current.setOptions({ modelComplexity: 1, smoothLandmarks: true, minDetectionConfidence: 0.5 });
                pose.current.onResults((results) => {
                    if (results.poseLandmarks) {
                        poseBuffer.current.push(results.poseLandmarks);
                        if (poseBuffer.current.length === 64) {
                            axios.post('http://localhost:8000/predict', { frames: poseBuffer.current })
                                .then(res => {
                                    const { action, confidence } = res.data;
                                    if (confidence > 0.8 && action !== lastWord.current) {
                                        setTranscript(prev => prev + (prev ? " " : "") + action);
                                        lastWord.current = action;
                                    }
                                })
                                .catch(err => console.error(err));
                            poseBuffer.current = poseBuffer.current.slice(10);
                        }
                    }
                });
                setPoseReady(true);
            } else {
                setTimeout(initPose, 500);
            }
        };
        initPose();
    }, []);

    useEffect(() => {
        let animationId;
        const processFrame = async () => {
            if (videoRef.current && videoRef.current.readyState >= 2 && canvasRef.current && pose.current) {
                const ctx = canvasRef.current.getContext('2d');
                const video = videoRef.current;

                if (mode === 'screen' && streamActive) {
                    const rect = video.getBoundingClientRect();
                    const scaleX = video.videoWidth / rect.width;
                    const scaleY = video.videoHeight / rect.height;
                    const sx = cropBox.x * scaleX;
                    const sy = cropBox.y * scaleY;
                    const sWidth = cropBox.width * scaleX;
                    const sHeight = cropBox.height * scaleY;

                    canvasRef.current.width = sWidth;
                    canvasRef.current.height = sHeight;
                    ctx.drawImage(video, sx, sy, sWidth, sHeight, 0, 0, sWidth, sHeight);
                    await pose.current.send({ image: canvasRef.current });
                } else if (streamActive) {
                    await pose.current.send({ image: video });
                }
            }
            animationId = requestAnimationFrame(processFrame);
        };
        processFrame();
        return () => cancelAnimationFrame(animationId);
    }, [cropBox, mode, streamActive]);

    const stopActiveStream = () => {
        if (videoRef.current && videoRef.current.srcObject) {
            const tracks = videoRef.current.srcObject.getTracks();
            tracks.forEach(track => track.stop());
            videoRef.current.srcObject = null;
        }
    };

    const handleTabChange = (newMode) => {
        setMode(newMode);
        stopActiveStream();
        setStreamActive(false);
        poseBuffer.current = [];
        if (videoRef.current) videoRef.current.src = "";
    };

    const startCameraCapture = async () => {
        stopActiveStream();
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            if (videoRef.current) {
                videoRef.current.srcObject = stream;
            }
            setStreamActive(true);
            poseBuffer.current = [];
        } catch (err) { console.error(err); }
    };

    const startScreenCapture = async () => {
        stopActiveStream();
        try {
            const stream = await navigator.mediaDevices.getDisplayMedia({ video: true });
            if (videoRef.current) {
                videoRef.current.srcObject = stream;
            }
            setStreamActive(true);
            poseBuffer.current = [];
        } catch (err) { console.error(err); }
    };

    const handleVideoUpload = (e) => {
        stopActiveStream();
        const file = e.target.files[0];
        if (file) {
            if (videoRef.current) {
                videoRef.current.src = URL.createObjectURL(file);
            }
            setStreamActive(true);
            poseBuffer.current = [];
        }
    };

    return (
        <div className="fixed inset-0 overflow-hidden bg-zinc-950 text-zinc-300 flex flex-col font-sans m-0 p-0 antialiased selection:bg-indigo-500/30">

            <header className="h-14 bg-zinc-950/80 backdrop-blur-md border-b border-white/5 px-6 flex items-center justify-between shrink-0 z-20">
                <div className="flex items-center gap-3">
                    <div className="w-6 h-6 rounded bg-gradient-to-tr from-indigo-500 to-cyan-400 flex items-center justify-center">
                        <div className="w-2 h-2 bg-white rounded-full"></div>
                    </div>
                    <h1 className="text-base font-semibold text-zinc-100 tracking-tight">VSLNet Workspace</h1>
                </div>
                {!poseReady ? (
                    <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-zinc-900 border border-white/5 text-xs font-medium text-zinc-400">
                        <span className="relative flex h-2 w-2">
                            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-amber-400 opacity-75"></span>
                            <span className="relative inline-flex rounded-full h-2 w-2 bg-amber-500"></span>
                        </span>
                        Initializing Engine...
                    </div>
                ) : (
                    <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-zinc-900 border border-white/5 text-xs font-medium text-zinc-300">
                        <span className="relative flex h-2 w-2">
                            <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
                        </span>
                        Ready
                    </div>
                )}
            </header>

            <main className="flex-1 flex flex-row w-full h-full overflow-hidden">
                <aside className="w-[280px] bg-zinc-900/40 border-r border-white/5 flex flex-col shrink-0 z-10 backdrop-blur-sm">
                    <div className="px-5 py-4 text-[11px] font-semibold text-zinc-500 uppercase tracking-widest mt-2">
                        Chế độ phân tích
                    </div>
                    <div className="px-3 flex flex-col gap-1">
                        <button onClick={() => handleTabChange('camera')} className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all duration-200 ${mode === 'camera' ? 'bg-zinc-800 text-zinc-100 shadow-sm' : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/50'}`}>
                            <svg className="w-4 h-4 opacity-70" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2"><path strokeLinecap="round" strokeLinejoin="round" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" /></svg>
                            Camera Trực Tiếp
                        </button>
                        <button onClick={() => handleTabChange('screen')} className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all duration-200 ${mode === 'screen' ? 'bg-zinc-800 text-zinc-100 shadow-sm' : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/50'}`}>
                            <svg className="w-4 h-4 opacity-70" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2"><path strokeLinecap="round" strokeLinejoin="round" d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" /></svg>
                            Quét Màn Hình
                        </button>
                        <button onClick={() => handleTabChange('upload')} className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all duration-200 ${mode === 'upload' ? 'bg-zinc-800 text-zinc-100 shadow-sm' : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/50'}`}>
                            <svg className="w-4 h-4 opacity-70" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2"><path strokeLinecap="round" strokeLinejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" /></svg>
                            Tải Video Lên
                        </button>
                    </div>

                    <div className="px-4 mt-auto mb-6">
                        {mode === 'camera' && <button onClick={startCameraCapture} className="w-full py-2.5 bg-zinc-100 hover:bg-white text-zinc-900 rounded-lg text-sm font-semibold transition-colors shadow-sm">Khởi động Camera</button>}
                        {mode === 'screen' && <button onClick={startScreenCapture} className="w-full py-2.5 bg-zinc-100 hover:bg-white text-zinc-900 rounded-lg text-sm font-semibold transition-colors shadow-sm">Chọn Nguồn Quét</button>}
                        {mode === 'upload' && <input type="file" accept="video/*" onChange={handleVideoUpload} className="w-full text-sm text-zinc-400 file:mr-3 file:py-2 file:px-4 file:rounded-lg file:border-0 file:bg-zinc-800 file:text-zinc-200 hover:file:bg-zinc-700 cursor-pointer transition-colors" />}
                    </div>
                </aside>

                <section className="flex-1 bg-black relative flex items-center justify-center overflow-hidden">
                    <video
                        ref={videoRef}
                        autoPlay
                        loop
                        muted
                        playsInline
                        className={`absolute inset-0 w-full h-full object-contain pointer-events-none transition-opacity duration-500 ${streamActive ? 'opacity-100' : 'opacity-0'}`}
                    />

                    {streamActive && mode === 'screen' && (
                        <Rnd
                            position={{x: cropBox.x, y: cropBox.y}}
                            size={{width: cropBox.width, height: cropBox.height}}
                            onDragStop={(e, d) => setCropBox({...cropBox, x: d.x, y: d.y})}
                            onResizeStop={(e, dir, ref, delta, pos) => {
                                setCropBox({ x: pos.x, y: pos.y, width: parseInt(ref.style.width), height: parseInt(ref.style.height) });
                            }}
                            bounds="parent"
                            className="border-[1.5px] border-rose-500/80 z-50 cursor-move group hover:border-rose-400 transition-colors"
                        >
                            <div className="absolute -top-6 left-0 bg-rose-500/90 text-white text-[10px] font-bold px-2 py-1 rounded-t-sm tracking-wider backdrop-blur-sm opacity-0 group-hover:opacity-100 transition-opacity">
                                SCAN AREA
                            </div>
                            <div className="absolute inset-0 bg-rose-500/5 group-hover:bg-rose-500/10 transition-colors pointer-events-none"></div>
                        </Rnd>
                    )}

                    {!streamActive && (
                        <div className="flex flex-col items-center justify-center gap-3 text-zinc-600">
                            <svg className="w-12 h-12 stroke-1 opacity-50" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" /></svg>
                            <p className="text-sm font-medium tracking-wide">Signal Interrupted</p>
                        </div>
                    )}

                    <canvas ref={canvasRef} className="hidden" />
                </section>

                <aside className="w-[320px] bg-zinc-900/40 border-l border-white/5 flex flex-col shrink-0 z-10 backdrop-blur-sm">
                    <div className="px-5 py-4 border-b border-white/5 flex items-center justify-between">
                        <span className="text-[11px] font-semibold text-zinc-500 uppercase tracking-widest mt-2">
                            Live Transcript
                        </span>
                        {streamActive && (
                            <span className="flex h-2 w-2 relative mt-2">
                                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-rose-400 opacity-75"></span>
                                <span className="relative inline-flex rounded-full h-2 w-2 bg-rose-500"></span>
                            </span>
                        )}
                    </div>

                    <div className="flex-1 p-6 overflow-y-auto">
                        <p className="text-xl leading-relaxed font-medium text-zinc-200 tracking-tight">
                            {!transcript ? (
                                <span className="text-zinc-600 font-normal text-base">Listening for gestures...</span>
                            ) : (
                                transcript
                            )}
                        </p>
                    </div>

                    <div className="px-4 mb-6">
                        <button onClick={() => { setTranscript(""); lastWord.current = ""; }} className="w-full py-2.5 bg-zinc-800/50 hover:bg-zinc-800 text-zinc-400 hover:text-zinc-200 rounded-lg text-sm font-medium transition-colors border border-transparent hover:border-white/5">
                            Xóa lịch sử
                        </button>
                    </div>
                </aside>

            </main>
        </div>
    );
}

export default App;