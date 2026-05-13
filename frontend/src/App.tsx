import { useState, useRef, useEffect } from 'react'
import type { DragEvent, ChangeEvent } from 'react'
import './index.css'

const API_BASE = import.meta.env.VITE_API_BASEURL || 'http://localhost:8000'

function App() {
    const [file, setFile] = useState<File | null>(null)
    const [status, setStatus] = useState<'idle' | 'uploading' | 'queued' | 'processing' | 'completed' | 'error'>('idle')
    const [jobId, setJobId] = useState<string | null>(null)
    const [result, setResult] = useState<any>(null)
    const [errorMsg, setErrorMsg] = useState('')
    const [startTime, setStartTime] = useState<number | null>(null)
    const [latency, setLatency] = useState<string>("0.000")

    const fileInputRef = useRef<HTMLInputElement>(null)

    const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
        e.preventDefault()
    }

    const handleDrop = (e: DragEvent<HTMLDivElement>) => {
        e.preventDefault()
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            setFile(e.dataTransfer.files[0])
        }
    }

    const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files.length > 0) {
            setFile(e.target.files[0])
        }
    }

    const handleBoxClick = () => {
        fileInputRef.current?.click()
    }

    const handleInitialize = async () => {
        if (!file) {
            setErrorMsg("FILE_MISSING: Provide valid media signal")
            return
        }
        setStatus('uploading')
        setErrorMsg('')
        setResult(null)
        setStartTime(Date.now())
        setLatency("0.000")

        const formData = new FormData()
        formData.append('video', file)

        try {
            const res = await fetch(`${API_BASE}/detect`, {
                method: 'POST',
                body: formData,
            })
            if (!res.ok) throw new Error('Failed to initialize detection')
            const data = await res.json()
            setJobId(data.job_id)
            setStatus('queued')
        } catch (err: any) {
            setStatus('error')
            setErrorMsg(err.message || 'Unknown error occurred contacting API')
        }
    }

    useEffect(() => {
        let interval: ReturnType<typeof setInterval>

        if (jobId && (status === 'queued' || status === 'processing')) {
            interval = setInterval(async () => {
                try {
                    if (startTime) {
                        setLatency(((Date.now() - startTime) / 1000).toFixed(3))
                    }
                    const res = await fetch(`${API_BASE}/job/${jobId}`)
                    const data = await res.json()

                    if (data.status === 'completed') {
                        setStatus('completed')
                        setResult(data.results)
                        if (startTime) {
                            setLatency(((Date.now() - startTime) / 1000).toFixed(3))
                        }
                    } else if (data.status === 'failed') {
                        setStatus('error')
                        setErrorMsg(data.error || 'Job processing failed')
                    } else {
                        setStatus(data.status)
                    }
                } catch (err) {
                    console.error("Polling error", err)
                }
            }, 1000)
        }

        return () => {
            if (interval) clearInterval(interval)
        }
    }, [jobId, status, startTime])

    // Derive visual values
    const displayProb = result?.final_synthetic_probability ?? 0.5;
    const isFake = result?.label === 'fake' || displayProb > 0.5;
    const confidenceScore = isFake ? displayProb : (1 - displayProb);
    const confidence = (confidenceScore * 100).toFixed(1);
    const heatmapUrl = (result?.heatmap_url && isFake) ? `${API_BASE}${result.heatmap_url}` : null;
    const isProcessing = status === 'uploading' || status === 'queued' || status === 'processing';

    let scanModeText = 'ACTIVE_SCAN'
    if (status === 'uploading') scanModeText = 'UPLOADING_MEDIA...'
    if (status === 'queued') scanModeText = 'AWAITING_WORKER'
    if (status === 'processing') scanModeText = 'ANALYZING_FRAMES'
    if (status === 'completed') scanModeText = 'ANALYSIS_COMPLETE'
    if (status === 'error') scanModeText = 'SYSTEM_FAILURE'

    return (
        <div className="min-h-screen p-8 bg-[#f0f0f0]">
            <header className="flex flex-col md:flex-row justify-between items-start md:items-end mb-16 gap-8">
                <div className="brutalist-border bg-black text-white p-6 rotate-[-1deg] transition-transform hover:rotate-0">
                    <h1 className="text-6xl font-black tracking-tighter leading-none">ORIVIS<span className="text-[#ff6b00]">_</span></h1>
                </div>

            </header>

            <main className="grid grid-cols-1 md:grid-cols-12 gap-12">
                {/* Input Section */}
                <div className="md:col-span-4 space-y-10">
                    <section className="brutalist-border bg-white p-8 relative overflow-hidden">
                        <div className="absolute top-0 left-0 bg-black text-white px-4 py-1 text-sm brutalist-font font-bold uppercase transition-colors">MODE: {scanModeText}</div>
                        <h2 className="text-3xl font-black mt-8 mb-6">INPUT_SIGNAL</h2>

                        <input
                            type="file"
                            ref={fileInputRef}
                            onChange={handleFileChange}
                            className="hidden"
                            accept="video/*,audio/*"
                        />

                        <div
                            onClick={handleBoxClick}
                            onDragOver={handleDragOver}
                            onDrop={handleDrop}
                            className={`brutalist-border-small border-dashed h-64 flex flex-col items-center justify-center bg-zinc-100 hover:bg-white cursor-pointer group transition-colors ${file ? 'border-orange-500 bg-orange-50/50' : ''}`}
                        >
                            {file ? (
                                <div className="flex flex-col items-center">
                                    <svg className="w-16 h-16 mb-4 text-[#ff6b00] group-hover:scale-110 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" /></svg>
                                    <p className="font-black text-center px-4 text-[#ff6b00] break-all">FILE_READY:<br />{file.name}</p>
                                </div>
                            ) : (
                                <div className="flex flex-col items-center">
                                    <svg className="w-16 h-16 mb-4 group-hover:scale-110 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" strokeLinecap="round" strokeLinejoin="round" strokeWidth="3" /></svg>
                                    <p className="font-black text-center px-4">DRAG MEDIA TO START<br />FORGERY DETECTION</p>
                                </div>
                            )}
                        </div>

                        {errorMsg && <p className="text-red-600 brutalist-font font-bold mt-4 animate-pulse">{errorMsg}</p>}

                        <button
                            onClick={handleInitialize}
                            disabled={isProcessing}
                            className={`w-full mt-8 bg-black text-white py-5 text-2xl font-black brutalist-font hover:bg-[#ff6b00] hover:text-black transition-all ${isProcessing ? 'opacity-50 cursor-not-allowed' : ''}`}
                        >
                            {isProcessing ? 'INITIALIZING...' : 'INITIALIZE'}
                        </button>
                    </section>

                    <section className="brutalist-border bg-[#ff6b00] p-6">
                        <h3 className="text-xl font-black mb-4">SYSTEM_INFO</h3>
                        <div className="space-y-4 brutalist-font font-bold text-sm">
                            <div className="flex justify-between border-b-2 border-black pb-2">
                                <span>LATENCY</span>
                                <span>{latency}S</span>
                            </div>
                            <div className="flex justify-between border-b-2 border-black pb-2">
                                <span>CONFIDENCE</span>
                                <span>{confidence}%</span>
                            </div>
                            <div className="flex justify-between border-b-2 border-black pb-2">
                                <span>V-WEIGHT</span>
                                <span>{result?.video_details?.weight ? (result.video_details.weight * 100).toFixed(0) + '%' : '60%'}</span>
                            </div>
                            <div className="flex justify-between border-b-2 border-black pb-2">
                                <span>A-WEIGHT</span>
                                <span>{result?.audio_details?.weight ? (result.audio_details.weight * 100).toFixed(0) + '%' : '40%'}</span>
                            </div>
                        </div>
                    </section>
                </div>

                {/* Display Section */}
                <div className="md:col-span-8 space-y-12">
                    <section className="brutalist-border p-0 bg-white overflow-hidden relative min-h-[500px]">
                        <div className="bg-black text-white p-4 flex justify-between items-center brutalist-font font-bold">
                            <span>{status === 'completed' ? 'ANALYSIS_OUTCOME' : 'LIVE_X_ANALYSIS_VIEW'}</span>
                        </div>
                        <div className="p-12">
                            <div className="brutalist-border bg-zinc-900 h-96 relative group cursor-crosshair overflow-hidden flex items-center justify-center">

                                {heatmapUrl ? (
                                    <img src={heatmapUrl} alt="Heatmap" className="w-full h-full object-cover opacity-80 mix-blend-screen" />
                                ) : (
                                    <>
                                        {/* Default processing or idle UI */}
                                        <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,_var(--tw-gradient-stops))] from-orange-500/40 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
                                        <div className={`absolute top-1/2 left-1/4 w-48 h-48 rounded-full blur-3xl transition-all duration-700 ease-in-out ${isProcessing ? 'bg-[#ff6b00]/80 opacity-60 animate-pulse' : 'bg-red-600/50 opacity-0 group-hover:opacity-100'}`}></div>

                                        <div className="absolute inset-0 flex flex-col items-center justify-center">
                                            <span className={`text-white brutalist-font font-black text-4xl opacity-20 transition-opacity delay-100 duration-300 ${isProcessing ? 'opacity-100 animate-pulse' : 'group-hover:opacity-100'}`}>
                                                {isProcessing ? 'ANALYZING...' : (status === 'completed' ? 'AUTHENTIC_SIGNAL' : 'AWAITING_MEDIA')}
                                            </span>
                                        </div>
                                    </>
                                )}
                            </div>
                        </div>

                        {(status === 'completed' || isProcessing || status === 'error') && (
                            <div className="absolute bottom-8 right-8 rotate-[3deg] transition-transform hover:rotate-0">
                                <div className={`brutalist-border p-6 max-w-xs shadow-xl ${isFake || status === 'error' ? 'bg-red-500 text-white' : 'bg-[#ff6b00] text-black'} `}>
                                    <h4 className="text-2xl font-black mb-2">
                                        {isProcessing ? 'AWAITING_RESULT...' : (status === 'error' ? 'ERROR' : (isFake ? 'FORGERY_DETECTED' : 'ORIGINAL_MEDIA'))}
                                    </h4>
                                    {status === 'completed' && (
                                        <p className="font-bold text-sm uppercase">
                                            {isFake ? 'HIGH_CONFIDENCE:' : 'CONFIDENCE:'} {confidence}%<br />
                                            {heatmapUrl ? 'V-SALIENCY HOTSPOTS DETECTED' : 'ANALYSIS COMPLETE'}
                                        </p>
                                    )}
                                </div>
                            </div>
                        )}
                    </section>

                    <section className="grid grid-cols-1 md:grid-cols-2 gap-8">
                        <div className="brutalist-border-small bg-white p-6 offset-card group">
                            <h5 className="brutalist-font font-black text-xl mb-4 group-hover:text-[#ff6b00] transition-colors">VIDEO_SCORE</h5>
                            <div className="h-4 bg-zinc-100 brutalist-border-small p-0 overflow-hidden">
                                <div className="h-full bg-black transition-all duration-1000" style={{ width: result?.video_details ? `${result.video_details.probability * 100}%` : (isProcessing ? '45%' : '0%') }}></div>
                            </div>
                        </div>
                        <div className="brutalist-border-small bg-white p-6 offset-card group">
                            <h5 className="brutalist-font font-black text-xl mb-4 group-hover:text-[#ff6b00] transition-colors">AUDIO_SCORE</h5>
                            <div className="h-4 bg-zinc-100 brutalist-border-small p-0 overflow-hidden">
                                <div className="h-full bg-[#ff6b00] transition-all duration-1000" style={{ width: result?.audio_details ? `${result.audio_details.probability * 100}%` : (isProcessing ? '45%' : '0%') }}></div>
                            </div>
                        </div>
                    </section>
                </div>
            </main>


        </div>
    )
}

export default App
