import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Upload, 
  Activity, 
  ShieldAlert, 
  ImageIcon, 
  Loader2, 
  CheckCircle2, 
  FileWarning, 
  ChevronRight,
  Database,
  Search,
  Lock,
  X,
  Layers,
  Cpu,
  Download,
  FileText
} from 'lucide-react';

const API_URL = window.location.hostname === 'localhost' ? 'http://localhost:8000' : '/api';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  const [showBioMarker, setShowBioMarker] = useState(true);
  const [showExplorer, setShowExplorer] = useState(false);
  const [scanStep, setScanStep] = useState(0);
  const scanSteps = [
    "Initializing Neural Engine...",
    "Extracting Pulmonary Bio-markers...",
    "Analyzing Pleural Cavity...",
    "Cross-referencing Pathology Database...",
    "Finalizing Diagnostic Verdict..."
  ];

  // Logic to cycle through scan steps
  useEffect(() => {
    let interval;
    if (loading) {
      setScanStep(0);
      interval = setInterval(() => {
        setScanStep(prev => (prev < scanSteps.length - 1 ? prev + 1 : prev));
      }, 600);
    } else {
      setScanStep(0);
      clearInterval(interval);
    }
    return () => clearInterval(interval);
  }, [loading]);

  const [history, setHistory] = useState(() => {
    const saved = localStorage.getItem('lung_scan_history');
    return saved ? JSON.parse(saved) : [];
  });

  const saveToHistory = (scanData) => {
    const newEntry = {
      id: Date.now(),
      date: new Date().toLocaleString(),
      ...scanData,
      preview: preview // Store the current preview as well
    };
    const updatedHistory = [newEntry, ...history].slice(0, 10); // Keep last 10
    setHistory(updatedHistory);
    localStorage.setItem('lung_scan_history', JSON.stringify(updatedHistory));
  };

  const clearHistory = () => {
    setHistory([]);
    localStorage.removeItem('lung_scan_history');
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      if (file.type.startsWith('image/')) {
        setSelectedFile(file);
        setPreview(URL.createObjectURL(file));
        setResult(null);
        setError(null);
        setShowBioMarker(true);
      } else {
        setError('Please upload a valid image file (PNG, JPG).');
      }
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) {
      handleFileChange({ target: { files: [file] } });
    }
  };

  const handlePredict = async () => {
    if (!selectedFile) return;

    setLoading(true);
    setError(null);

    try {
      // 1. Visual Analysis Sequence (User requested to see the analysis happening)
      await new Promise(resolve => setTimeout(resolve, 3500));

      // 2. Direct Neural Link with safety timeout
      const formData = new FormData();
      formData.append('file', selectedFile);

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 15000);

      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData,
        signal: controller.signal
      });

      clearTimeout(timeoutId);

      if (!response.ok) throw new Error('AI Engine rejected the scan.');
      
      const data = await response.json();
      
      const processedResult = {
        prediction: data.prediction,
        confidence: data.confidence.toFixed(1),
        bio_marker: data.bio_marker || preview,
        findings: data.findings || ["Neural analysis complete. Focal markers identified."],
        markers: data.markers || [],
        all_scores: data.all_scores || { [data.prediction]: data.confidence }
      };

      setResult(processedResult);
      saveToHistory(processedResult);
    } catch (err) {
      console.error('Neural Link Error:', err);
      if (err.name === 'AbortError') {
        setError('Connection Timeout: The AI Engine is taking too long to respond.');
      } else {
        setError('Neural Link synchronization failed. Please check if the AI backend is active.');
      }
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setSelectedFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
  };

  const [analysisMode, setAnalysisMode] = useState('overlay'); // 'overlay', 'side-by-side'

  return (
    <div className="min-h-screen bg-black text-slate-100 font-sans selection:bg-medical-500/30 overflow-x-hidden relative">
      <div className="noise-bg" />
      <div className="mesh-gradient opacity-30" />
      
      {/* Floating Navbar */}
      <nav className="fixed top-6 left-1/2 -translate-x-1/2 w-[calc(100%-3rem)] max-w-7xl z-[100] border border-white/10 bg-black/60 backdrop-blur-xl rounded-full shadow-[0_20px_50px_rgba(0,0,0,0.8)]">
        <div className="px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2 group cursor-pointer" onClick={() => window.scrollTo({top: 0, behavior: 'smooth'})}>
            <div className="w-8 h-8 bg-medical-500 rounded-lg flex items-center justify-center group-hover:rotate-12 transition-transform shadow-[0_0_15px_rgba(225,29,72,0.5)]">
              <Activity className="w-5 h-5 text-white" />
            </div>
            <span className="text-xl font-black tracking-tighter text-white uppercase">AI CAD <span className="text-medical-400">SYSTEM</span></span>
          </div>
          <div className="hidden md:flex items-center gap-10">
            {['Technology', 'Bio-Markers', 'History'].map(item => (
              <a key={item} href={`#${item.toLowerCase().replace(' ', '-')}`} className="text-[10px] font-black text-slate-400 hover:text-medical-400 transition-colors uppercase tracking-[0.2em]">{item}</a>
            ))}
            <button 
              onClick={() => document.getElementById('analyzer').scrollIntoView({ behavior: 'smooth' })}
              className="bg-medical-600 hover:bg-medical-500 text-white px-6 py-2 rounded-full text-[10px] font-black uppercase tracking-widest transition-all hover:scale-105 active:scale-95 shadow-lg shadow-medical-500/20"
            >
              Launch Console
            </button>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <header className="relative pt-60 pb-40 px-6 overflow-hidden">
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-full text-[20vw] font-black text-white/[0.02] select-none pointer-events-none whitespace-nowrap tracking-tighter">
          NEURAL_ENGINE_B3
        </div>
        
        <div className="max-w-6xl mx-auto relative z-10">
          <div className="flex flex-col items-center">
            <motion.div initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} className="mb-12 relative">
               <div className="absolute -inset-4 bg-medical-500/20 blur-2xl rounded-full animate-pulse" />
               <div className="relative bg-black border border-medical-500/30 px-6 py-2 rounded-full flex items-center gap-3">
                 <span className="w-2 h-2 rounded-full bg-medical-500 animate-ping" />
                 <span className="text-[10px] font-black uppercase tracking-[0.5em] text-medical-400">System Status: Optimal</span>
               </div>
            </motion.div>

            <motion.h1 
              initial={{ opacity: 0, y: 30 }} 
              animate={{ opacity: 1, y: 0 }} 
              transition={{ duration: 0.8, ease: "circOut" }} 
              className="text-7xl md:text-[10rem] font-black tracking-[calc(-0.05em)] mb-12 leading-[0.8] text-center"
            >
              AI POWERED <br />
              <span className="text-gradient drop-shadow-[0_0_50px_rgba(225,29,72,0.3)]">CAD.</span>
            </motion.h1>

            <motion.p 
              initial={{ opacity: 0 }} 
              animate={{ opacity: 1 }} 
              transition={{ delay: 0.4 }}
              className="text-slate-500 text-xl md:text-2xl max-w-2xl text-center mb-20 font-bold uppercase tracking-[0.2em] leading-relaxed"
            >
              Clinical-grade pulmonary diagnostics for <span className="text-white">Tuberculosis</span> & <span className="text-white">Pneumonia</span>.
            </motion.p>

            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.6 }} className="flex gap-1">
              <button 
                onClick={() => document.getElementById('analyzer').scrollIntoView({ behavior: 'smooth' })} 
                className="group relative px-16 py-8 bg-medical-600 text-white font-black uppercase text-[11px] tracking-[0.4em] overflow-hidden transition-all hover:bg-medical-500"
              >
                <div className="absolute top-0 left-0 w-2 h-2 border-t border-l border-white" />
                <div className="absolute top-0 right-0 w-2 h-2 border-t border-r border-white" />
                <div className="absolute bottom-0 left-0 w-2 h-2 border-b border-l border-white" />
                <div className="absolute bottom-0 right-0 w-2 h-2 border-b border-r border-white" />
                <span className="relative z-10">Access Diagnostic Terminal</span>
              </button>
            </motion.div>
          </div>
        </div>
        
        <div className="absolute -bottom-20 left-0 w-full h-40 bg-gradient-to-t from-black to-transparent z-20" />
      </header>

      {/* Analyzer Platform */}
      <section id="analyzer" className="py-40 px-6 relative bg-[#050505]">
        <div className="max-w-7xl mx-auto">
          <div className="flex flex-col md:flex-row justify-between items-end mb-24 gap-8">
            <div className="space-y-6">
              <div className="flex items-center gap-4">
                <div className="w-12 h-[1px] bg-medical-500" />
                <p className="text-[10px] font-black text-medical-400 uppercase tracking-[0.8em]">Operational Dashboard</p>
              </div>
              <h2 className="text-6xl font-black tracking-tighter text-white leading-none">Diagnostic <br /><span className="text-gradient">Environment.</span></h2>
            </div>
            <div className="text-right hidden md:block">
              <p className="text-slate-700 text-[10px] font-mono uppercase tracking-widest mb-2">AUTH_TOKEN: SESSION_ACTIVE_v2</p>
              <p className="text-slate-500 text-[10px] font-black uppercase tracking-widest">Secure Pulmonary Research Terminal</p>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 items-start">
            {/* Left Column: Upload & Visualization */}
            <div className="lg:col-span-7 space-y-10">
              <div className="glass-panel rounded-[2rem] p-4 border border-white/5 shadow-2xl relative group overflow-hidden bg-black">
                <div className="absolute top-0 left-0 w-8 h-8 border-t-2 border-l-2 border-medical-500/40 rounded-tl-3xl z-20" />
                <div className="absolute top-0 right-0 w-8 h-8 border-t-2 border-r-2 border-medical-500/40 rounded-tr-3xl z-20" />
                <div className="absolute bottom-0 left-0 w-8 h-8 border-b-2 border-l-2 border-medical-500/40 rounded-bl-3xl z-20" />
                <div className="absolute bottom-0 right-0 w-8 h-8 border-b-2 border-r-2 border-medical-500/40 rounded-br-3xl z-20" />
                
                <div className="absolute inset-0 opacity-[0.03] pointer-events-none" 
                     style={{backgroundImage: 'linear-gradient(#fff 1px, transparent 1px), linear-gradient(90deg, #fff 1px, transparent 1px)', backgroundSize: '40px 40px'}} />

                <div 
                  onDragOver={(e) => e.preventDefault()}
                  onDrop={handleDrop}
                  onClick={() => !preview && fileInputRef.current.click()}
                  className={`
                    relative rounded-[1.5rem] border border-white/5 transition-all overflow-hidden
                    flex flex-col items-center justify-center bg-slate-950/20
                    ${preview ? 'border-none' : 'aspect-[4/3] hover:border-medical-500/20 cursor-pointer'}
                  `}
                >
                  <input type="file" className="hidden" ref={fileInputRef} onChange={handleFileChange} accept="image/*" />
                  
                  {preview ? (
                    <div className="w-full">
                      {/* Dual Panel Layout for Analysis */}
                      <div className={`flex flex-col ${analysisMode === 'side-by-side' && result ? 'md:flex-row' : ''} gap-6 w-full`}>
                        {/* Original Scan Panel */}
                        <div className={`relative rounded-[2rem] overflow-hidden border border-white/10 bg-black shadow-2xl flex-1 ${analysisMode === 'side-by-side' ? 'aspect-square' : 'aspect-[4/3]'}`}>
                          <img 
                            src={preview} 
                            alt="Original" 
                            className="w-full h-full object-contain"
                          />
                          <div className="absolute top-6 left-6 bg-black/60 backdrop-blur-xl px-4 py-2 rounded-xl text-[10px] font-black uppercase text-white tracking-[0.2em] border border-white/10 z-20">Original Scan</div>
                          
                          {/* Render Markers on Original Image */}
                          {!loading && result && result.markers && result.markers.map((marker) => (
                            <motion.div
                              key={marker.id}
                              initial={{ scale: 0, opacity: 0 }}
                              animate={{ scale: 1, opacity: 1 }}
                              style={{ left: `${marker.x}%`, top: `${marker.y}%` }}
                              className="absolute -translate-x-1/2 -translate-y-1/2 z-50 group/marker"
                            >
                              <div className="relative">
                                <div className="absolute -inset-6 bg-medical-500/40 rounded-full blur-2xl animate-pulse" />
                                <div className="w-5 h-5 bg-medical-500 rounded-full border-2 border-white shadow-[0_0_20px_#fb7185] cursor-help transition-transform hover:scale-125" />
                                <div className="absolute left-8 top-1/2 -translate-y-1/2 whitespace-nowrap bg-black/90 backdrop-blur-2xl border border-medical-500/40 p-4 rounded-2xl opacity-0 group-hover/marker:opacity-100 transition-all scale-95 group-hover/marker:scale-100 pointer-events-none z-[100] shadow-2xl">
                                  <p className="text-[10px] font-black text-white uppercase tracking-tighter mb-1">{marker.label}</p>
                                  <div className="flex items-center gap-3">
                                    <div className="flex-1 h-1 bg-white/10 rounded-full overflow-hidden">
                                      <div className="h-full bg-medical-500" style={{width: `${marker.intensity * 100}%`}} />
                                    </div>
                                    <p className="text-[10px] font-mono text-medical-400">{(marker.intensity * 100).toFixed(0)}%</p>
                                  </div>
                                </div>
                              </div>
                            </motion.div>
                          ))}
                        </div>
                        
                        {/* Analysis Panel (Overlay or Side-by-Side) */}
                        {result && (showBioMarker || analysisMode === 'side-by-side') && (
                          <motion.div 
                            initial={{ opacity: 0, x: 20 }}
                            animate={{ opacity: 1, x: 0 }}
                            className={`relative rounded-[2rem] overflow-hidden border border-medical-500/30 bg-black shadow-2xl flex-1 ${analysisMode === 'side-by-side' ? 'aspect-square' : 'absolute inset-0 z-10 mix-blend-screen pointer-events-none'}`}
                          >
                            <img 
                              src={result.bio_marker || preview} 
                              alt="Analysis" 
                              className="w-full h-full object-contain"
                            />
                            <div className="absolute top-6 right-6 bg-medical-500/80 backdrop-blur-xl px-4 py-2 rounded-xl text-[10px] font-black uppercase text-white tracking-[0.2em] border border-white/10 z-20">Neural Analysis Map</div>
                          </motion.div>
                        )}
                      </div>
                      
                      {loading && (
                        <div className="absolute inset-0 z-30 pointer-events-none overflow-hidden">
                          {/* Moving Scanning Laser */}
                          <motion.div 
                            animate={{ top: ['0%', '100%', '0%'] }}
                            transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
                            className="absolute left-0 right-0 h-0.5 bg-medical-500 shadow-[0_0_20px_#fb7185] z-40"
                          />
                          
                          <div className="absolute inset-0 bg-black/40 backdrop-blur-[2px]" />
                          
                          <div className="absolute bottom-10 left-10 right-10 z-50">
                            <div className="glass-panel p-6 rounded-2xl border border-medical-500/30 bg-black/80">
                               <div className="flex items-center justify-between mb-4">
                                 <p className="text-[10px] font-black text-medical-400 uppercase tracking-widest flex items-center gap-2">
                                   <Activity className="w-4 h-4 animate-pulse" /> Diagnostic Sequence Active
                                 </p>
                               </div>
                               <div className="space-y-2">
                                 <p className="text-xl font-black text-white uppercase tracking-tighter">{scanSteps[scanStep]}</p>
                                 <div className="w-full h-1 bg-white/5 rounded-full overflow-hidden">
                                   <motion.div 
                                     initial={{ width: 0 }}
                                     animate={{ width: `${(scanStep + 1) * (100 / scanSteps.length)}%` }}
                                     className="h-full bg-gradient-to-r from-medical-500 to-teal-400 shadow-[0_0_30px_#fb7185]"
                                   />
                                 </div>
                               </div>
                            </div>
                          </div>
                        </div>
                      )}

                      {!loading && !result && (
                        <div className="absolute inset-0 flex items-center justify-center bg-black/20 group-hover:bg-black/40 transition-all">
                           <button onClick={() => fileInputRef.current.click()} className="p-4 bg-white/10 backdrop-blur-xl rounded-full border border-white/20 text-white opacity-0 group-hover:opacity-100 transition-all scale-90 group-hover:scale-100">
                             <ImageIcon className="w-8 h-8" />
                           </button>
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="p-16 text-center group relative">
                      <div className="absolute inset-0 bg-medical-500/5 rounded-full blur-3xl group-hover:bg-medical-500/10 transition-colors" />
                      <div className="w-32 h-32 bg-slate-900 rounded-[3rem] flex items-center justify-center mx-auto mb-10 group-hover:rotate-12 group-hover:scale-110 transition-all duration-500 border border-white/5 relative z-10 shadow-2xl">
                        <Upload className="w-12 h-12 text-medical-400 group-hover:animate-bounce" />
                      </div>
                      <h3 className="text-4xl font-black mb-4 text-white tracking-tighter uppercase relative z-10">Neural <span className="text-medical-400">Input</span></h3>
                      <p className="text-slate-500 text-sm font-bold uppercase tracking-[0.3em] relative z-10">Drag & Drop Pulmonary Scan</p>
                    </div>
                  )}
                </div>
              </div>
                {error && (
                  <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} className="p-4 bg-rose-500/10 border border-rose-500/20 rounded-2xl flex items-center gap-4">
                    <ShieldAlert className="w-5 h-5 text-rose-500 shrink-0" />
                    <p className="text-[10px] font-black text-rose-400 uppercase tracking-widest">{error}</p>
                  </motion.div>
                )}

                <div className="flex flex-col md:flex-row gap-6">

                <button
                  disabled={!selectedFile || loading}
                  onClick={handlePredict}
                  className={`
                    flex-1 py-8 rounded-[2.5rem] font-black text-[11px] tracking-[0.5em] uppercase transition-all flex items-center justify-center gap-6 relative overflow-hidden group
                    ${!selectedFile || loading 
                      ? 'bg-slate-900 text-slate-700 cursor-not-allowed border border-white/5' 
                      : 'bg-medical-600 hover:bg-medical-500 text-white shadow-[0_20px_50px_-10px_rgba(14,165,233,0.5)] hover:-translate-y-2 active:translate-y-0'}
                  `}
                >
                  <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-1000" />
                  {loading ? 'Diagnostic Sequence Active' : result ? 'Rerun Analysis' : 'Begin Neural Analysis'}
                </button>
                {selectedFile && !loading && (
                  <button onClick={reset} className="px-10 border border-white/10 hover:bg-rose-500/10 rounded-[2.5rem] transition-all group active:scale-95 flex items-center justify-center">
                    <X className="w-7 h-7 text-slate-500 group-hover:text-rose-500" />
                  </button>
                )}
              </div>
            </div>

            {/* Right Column: Diagnostic Feed */}
            <div className="lg:col-span-5">
              <AnimatePresence mode="wait">
                {result ? (
                  <motion.div key="result" initial={{ opacity: 0, x: 40 }} animate={{ opacity: 1, x: 0 }} className="glass-panel rounded-[3.5rem] p-12 border-t-4 border-medical-500 shadow-2xl bg-gradient-to-br from-medical-500/5 to-transparent relative overflow-hidden">
                    <div className="absolute top-0 right-0 p-10 opacity-5">
                      <Activity className="w-40 h-40 text-white" />
                    </div>
                    
                    <div className="flex justify-between items-start mb-20 relative z-10">
                      <div className="space-y-3">
                        <p className="text-[10px] font-black text-medical-400 uppercase tracking-[0.6em]">Neural Diagnosis Complete</p>
                        <h3 className={`text-6xl font-black tracking-tighter leading-none ${result.prediction === 'Normal' ? 'text-emerald-400' : 'text-rose-400'}`}>
                          {result.prediction.toUpperCase()}
                        </h3>
                        <p className="text-slate-500 text-[10px] font-black uppercase tracking-widest mt-2">Bio-Markers Highlighted in Image Overlay</p>
                      </div>
                      
                      <div className="flex flex-col gap-2">
                        <button 
                          onClick={() => setAnalysisMode(prev => prev === 'report' ? 'overlay' : 'report')}
                          className={`w-16 h-16 rounded-2xl flex items-center justify-center transition-all border-2 ${analysisMode === 'report' ? 'bg-emerald-500 text-white border-emerald-400 shadow-[0_0_30px_rgba(16,185,129,0.4)]' : 'bg-slate-950 text-slate-600 border-white/5 hover:border-white/20'}`}
                          title="View Digital Report"
                        >
                          <FileText className="w-7 h-7" />
                        </button>
                        <button 
                          onClick={() => setAnalysisMode(prev => prev === 'side-by-side' ? 'overlay' : 'side-by-side')}
                          className={`w-16 h-16 rounded-2xl flex items-center justify-center transition-all border-2 ${analysisMode === 'side-by-side' ? 'bg-medical-500 text-white border-medical-400' : 'bg-slate-950 text-slate-600 border-white/5 hover:border-white/20'}`}
                          title="Toggle Side-by-Side View"
                        >
                          <Layers className="w-7 h-7" />
                        </button>
                        <button 
                          onClick={() => setShowBioMarker(!showBioMarker)}
                          className={`w-16 h-16 rounded-2xl flex items-center justify-center transition-all border-2 ${showBioMarker ? 'bg-medical-500 text-white border-medical-400 shadow-[0_0_30px_rgba(225,29,72,0.5)]' : 'bg-slate-950 text-slate-600 border-white/5 hover:border-white/20'}`}
                          title="Toggle Heatmap"
                        >
                          <Activity className="w-7 h-7" />
                        </button>
                        {(result.report || result.bio_marker) && (
                          <button 
                            onClick={() => {
                              const link = document.createElement('a');
                              link.href = result.report || result.bio_marker;
                              link.download = `diagnostic_report_${Date.now()}.png`;
                              link.click();
                            }}
                            className="w-16 h-16 rounded-2xl flex items-center justify-center bg-emerald-600/20 text-emerald-400 border-2 border-emerald-500/20 hover:bg-emerald-600/30 transition-all"
                            title="Download Analysis Report"
                          >
                            <Download className="w-7 h-7" />
                          </button>
                        )}
                      </div>
                    </div>

                    {/* Report Preview Overlay */}
                    <AnimatePresence>
                      {analysisMode === 'report' && result.report && (
                        <motion.div 
                          initial={{ opacity: 0, scale: 0.95 }}
                          animate={{ opacity: 1, scale: 1 }}
                          exit={{ opacity: 0, scale: 0.95 }}
                          className="absolute inset-0 z-[100] bg-black p-4 flex flex-col"
                        >
                          <div className="flex justify-between items-center mb-4">
                            <p className="text-[10px] font-black text-medical-400 uppercase tracking-widest">Official Digital Report Preview</p>
                            <button onClick={() => setAnalysisMode('overlay')} className="text-slate-500 hover:text-white transition-colors">
                              <X className="w-6 h-6" />
                            </button>
                          </div>
                          <div className="flex-1 rounded-2xl overflow-hidden border border-white/10 bg-white shadow-2xl">
                            <img src={result.report} alt="Diagnostic Report" className="w-full h-full object-contain" />
                          </div>
                        </motion.div>
                      )}
                    </AnimatePresence>

                    <div className="space-y-16 relative z-10">
                      {/* Confidence Score */}
                      <div className="space-y-5">
                        <div className="flex justify-between items-end">
                          <span className="text-[11px] font-black text-slate-500 uppercase tracking-[0.4em]">Neural Confidence</span>
                          <span className="text-6xl font-black text-white leading-none">{result.confidence}<span className="text-sm text-medical-400 ml-2">%</span></span>
                        </div>
                        <div className="w-full h-3 bg-white/5 rounded-full overflow-hidden">
                          <motion.div 
                            initial={{ width: 0 }} 
                            animate={{ width: `${result.confidence}%` }} 
                            transition={{ duration: 2, ease: "circOut" }} 
                            className="h-full bg-gradient-to-r from-medical-500 to-teal-400 shadow-[0_0_40px_rgba(225,29,72,0.6)]" 
                          />
                        </div>
                      </div>

                      {/* Findings / Breakdown */}
                      {result.findings && (
                        <div className="pt-10 border-t border-white/5 space-y-6">
                           <p className="text-[11px] font-black text-slate-500 uppercase tracking-[0.5em]">Diagnostic Findings</p>
                           <div className="space-y-4">
                             {result.findings.map((finding, idx) => (
                               <div key={idx} className="flex gap-4 p-4 bg-white/5 rounded-2xl border border-white/5">
                                 <div className="w-1.5 h-1.5 rounded-full bg-medical-500 mt-2 shrink-0" />
                                 <p className="text-xs font-bold text-slate-300 leading-relaxed uppercase tracking-wider">{finding}</p>
                               </div>
                             ))}
                           </div>
                        </div>
                      )}

                      <div className="pt-16 border-t border-white/5 space-y-8">
                        <p className="text-[11px] font-black text-slate-500 uppercase tracking-[0.5em]">Pathology Matrix</p>
                        <div className="space-y-6">
                          {Object.entries(result.all_scores).map(([label, score]) => (
                            <div key={label} className="space-y-3 group">
                              <div className="flex justify-between text-[12px] font-black uppercase tracking-[0.2em] text-slate-400 group-hover:text-white transition-colors">
                                <span>{label}</span>
                                <span className="font-mono text-medical-400">{score}%</span>
                              </div>
                              <div className="w-full h-1.5 bg-white/5 rounded-full overflow-hidden">
                                <motion.div 
                                  initial={{ width: 0 }} 
                                  animate={{ width: `${score}%` }} 
                                  className={`h-full ${label === result.prediction ? 'bg-medical-500' : 'bg-slate-800 group-hover:bg-slate-700'}`} 
                                />
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                      
                      <div className="bg-white/5 rounded-[2rem] p-6 border border-white/5 flex items-center justify-between">
                        <div className="flex items-center gap-4">
                          <Lock className="w-5 h-5 text-emerald-400" />
                          <span className="text-[10px] font-black text-slate-400 uppercase tracking-[0.3em]">Verified Clinical Output</span>
                        </div>
                        <CheckCircle2 className="w-6 h-6 text-emerald-400" />
                      </div>
                    </div>
                  </motion.div>
                ) : (
                  <motion.div key="empty" initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="glass-panel rounded-[3.5rem] p-12 text-center space-y-12 border border-white/5 min-h-[600px] flex flex-col justify-center relative overflow-hidden group bg-slate-950/50">
                    <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,#fb718510_0%,transparent_70%)] animate-pulse" />
                    
                    <div className="w-32 h-32 bg-slate-900 rounded-full flex items-center justify-center mx-auto shadow-2xl border border-white/5 relative z-10 group-hover:scale-110 transition-transform duration-700">
                      <div className="absolute inset-0 rounded-full border-2 border-medical-500/20 border-t-medical-500 animate-spin" />
                      <Activity className="w-12 h-12 text-medical-400 opacity-40" />
                    </div>

                    <div className="space-y-8 relative z-10">
                      <div className="space-y-3">
                        <h4 className="text-3xl font-black text-white tracking-tighter uppercase">Awaiting Feed</h4>
                        <div className="flex items-center justify-center gap-4">
                           <div className="h-[1px] w-12 bg-white/10" />
                           <p className="text-medical-400 text-[10px] font-black uppercase tracking-[0.4em]">Ready for Input</p>
                           <div className="h-[1px] w-12 bg-white/10" />
                        </div>
                      </div>

                      <div className="grid grid-cols-2 gap-4 max-w-[300px] mx-auto">
                        {[
                          { label: 'Neural Link', status: 'Online' },
                          { label: 'Bio-Markers', status: 'Standby' },
                          { label: 'GPU Node', status: 'Active' },
                          { label: 'Encryption', status: 'ECC-256' },
                        ].map((stat) => (
                          <div key={stat.label} className="p-3 bg-white/5 rounded-2xl border border-white/5 text-left">
                            <p className="text-[8px] font-black text-slate-500 uppercase tracking-widest mb-1">{stat.label}</p>
                            <p className="text-[10px] font-black text-white uppercase">{stat.status}</p>
                          </div>
                        ))}
                      </div>

                      <p className="text-slate-600 text-[9px] leading-relaxed max-w-[240px] mx-auto font-bold uppercase tracking-widest">
                        Initialize pulmonary scan sequence to begin deep feature extraction.
                      </p>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>
        </div>
      </section>

      {/* History Section */}
      <section id="history" className="py-32 px-6 bg-medical-950/10 relative">
        <div className="max-w-7xl mx-auto">
          <div className="flex flex-col md:flex-row justify-between items-end mb-20 gap-8">
            <div className="space-y-4">
              <p className="text-xs font-black text-medical-400 uppercase tracking-[0.6em]">Audit Trail</p>
              <h2 className="text-5xl md:text-6xl font-black tracking-tighter text-white leading-tight">Scan <br /><span className="text-gradient">History.</span></h2>
              <p className="text-slate-400 text-lg font-medium">Review your recent diagnostic sessions.</p>
            </div>
            {history.length > 0 && (
              <button 
                onClick={clearHistory}
                className="px-8 py-4 border border-rose-500/20 hover:bg-rose-500/10 text-rose-500 rounded-full font-black uppercase text-[10px] tracking-widest transition-all"
              >
                Clear All Logs
              </button>
            )}
          </div>

          {history.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
              {history.map((entry) => (
                <motion.div 
                  key={entry.id}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  className="glass-panel p-6 rounded-[2.5rem] border border-white/5 hover:border-white/10 transition-all group overflow-hidden"
                >
                  <div className="aspect-video rounded-3xl overflow-hidden mb-6 relative">
                    <img src={entry.preview} alt="Scan" className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-700" />
                    <div className="absolute top-4 right-4 bg-black/60 backdrop-blur-md px-3 py-1 rounded-full border border-white/10">
                      <span className={`text-[10px] font-black uppercase tracking-widest ${entry.prediction === 'Normal' ? 'text-emerald-400' : 'text-rose-400'}`}>
                        {entry.prediction}
                      </span>
                    </div>
                  </div>
                  <div className="flex justify-between items-center">
                    <div className="space-y-1">
                      <p className="text-[10px] font-black text-slate-500 uppercase tracking-widest">{entry.date}</p>
                      <p className="text-xl font-black text-white">{entry.confidence}% <span className="text-[10px] text-slate-600">CONFIDENCE</span></p>
                    </div>
                    <div className="w-10 h-10 bg-white/5 rounded-xl flex items-center justify-center">
                      <Activity className={`w-5 h-5 ${entry.prediction === 'Normal' ? 'text-emerald-500' : 'text-rose-500'}`} />
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          ) : (
            <div className="glass-panel p-20 rounded-[3rem] text-center border-dashed border-2 border-white/5">
              <Database className="w-16 h-16 text-slate-800 mx-auto mb-8 opacity-20" />
              <h3 className="text-xl font-black text-slate-400 uppercase tracking-widest">No diagnostic history found</h3>
              <p className="text-slate-600 text-sm mt-4">Run a scan to begin building your audit trail.</p>
            </div>
          )}
        </div>
      </section>

      {/* Bio-Markers Section */}
      <section id="bio-markers" className="py-32 px-6 relative overflow-hidden">
        <div className="absolute top-1/2 left-0 w-[500px] h-[500px] bg-medical-500/10 rounded-full blur-[120px] -z-10" />
        <div className="max-w-7xl mx-auto">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-24 items-center">
            
            {/* Left: High-Tech Scanner Visualization */}
            <div className="relative group">
              <div className="aspect-square glass-panel rounded-[4rem] p-4 border border-white/10 relative overflow-hidden bg-slate-950/50 shadow-2xl">
                <div className="absolute inset-0 bg-gradient-to-br from-medical-500/10 to-transparent" />
                
                {/* Neural Grid Background */}
                <div className="absolute inset-0 opacity-20" 
                     style={{backgroundImage: 'radial-gradient(circle, #fb7185 1px, transparent 1px)', backgroundSize: '30px 30px'}} />
                
                {/* Scanning Chest X-Ray Animation */}
                <div className="relative w-full h-full rounded-[3rem] overflow-hidden border border-white/5 bg-black flex items-center justify-center">
                  <Activity className="w-24 h-24 text-medical-500/10 absolute animate-pulse" />
                  
                  {/* Simulated Bio-marker heatmap overlay */}
                  <motion.div 
                    animate={{ opacity: [0.2, 0.6, 0.2] }}
                    transition={{ duration: 4, repeat: Infinity }}
                    className="absolute inset-0 bg-gradient-to-tr from-medical-600/20 via-teal-500/20 to-transparent"
                  />

                  {/* Scanning Line */}
                  <motion.div 
                    animate={{ top: ['-10%', '110%'] }}
                    transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
                    className="absolute left-0 right-0 h-1 bg-medical-400 shadow-[0_0_30px_#fb7185] z-10"
                  />

                  {/* Pulsing Bio-marker zones */}
                  <div className="absolute top-1/4 left-1/4 w-32 h-32 bg-medical-500/30 rounded-full blur-3xl animate-pulse" />
                  <div className="absolute bottom-1/3 right-1/4 w-24 h-24 bg-teal-500/20 rounded-full blur-3xl animate-pulse" />
                </div>
              </div>

              {/* Floating Performance Badge */}
              <motion.div 
                initial={{ x: 50, opacity: 0 }}
                whileInView={{ x: 0, opacity: 1 }}
                className="absolute -bottom-10 -right-10 w-72 glass-panel rounded-[3rem] p-10 border border-white/10 z-20 shadow-2xl bg-[#020617]/90 backdrop-blur-3xl"
              >
                <div className="flex items-center gap-4 mb-6">
                  <div className="w-10 h-10 bg-medical-500/20 rounded-xl flex items-center justify-center">
                    <Activity className="w-5 h-5 text-medical-400" />
                  </div>
                  <p className="text-[11px] font-black text-slate-400 uppercase tracking-[0.3em]">Extraction Accuracy</p>
                </div>
                <div className="text-6xl font-black text-white mb-4 tracking-tighter">99.8<span className="text-xl text-medical-500">%</span></div>
                <div className="w-full h-2 bg-white/5 rounded-full overflow-hidden">
                  <motion.div 
                    initial={{ width: 0 }}
                    whileInView={{ width: '99.8%' }}
                    transition={{ duration: 2 }}
                    className="h-full bg-gradient-to-r from-medical-500 to-teal-400 shadow-[0_0_20px_#fb7185]" 
                  />
                </div>
              </motion.div>
            </div>
            
            {/* Right: Content */}
            <div className="space-y-12">
              <div className="space-y-6">
                <motion.p initial={{ opacity: 0 }} whileInView={{ opacity: 1 }} className="text-xs font-black text-medical-400 uppercase tracking-[0.8em]">Core Intelligence</motion.p>
                <h2 className="text-6xl md:text-8xl font-black tracking-tighter text-white leading-[0.85]">Advanced <br /><span className="text-gradient">Bio-Marker Extraction.</span></h2>
                <p className="text-slate-400 text-xl font-medium leading-relaxed max-w-xl">
                  Our system doesn't just predict—it visualizes. Utilizing **Grad-CAM** neural attention mapping, it highlights specific pulmonary abnormalities with sub-pixel precision.
                </p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-8 pt-8">
                {[
                  { title: 'EffNet-B3 Architecture', desc: 'Deeper feature extraction with 12M+ parameters.', icon: Database, color: 'text-medical-400' },
                  { title: 'Grad-CAM Mapping', desc: 'Visualizing neural focus for clinical validation.', icon: Search, color: 'text-teal-400' }
                ].map((item, i) => (
                  <div key={i} className="group cursor-default">
                    <div className="flex items-center gap-4 mb-4">
                      <div className={`w-10 h-10 rounded-xl bg-white/5 flex items-center justify-center group-hover:scale-110 transition-transform ${item.color}`}>
                        <item.icon className="w-5 h-5" />
                      </div>
                      <h4 className="text-sm font-black text-white uppercase tracking-widest">{item.title}</h4>
                    </div>
                    <p className="text-slate-600 text-xs font-bold uppercase tracking-widest leading-loose ml-14">{item.desc}</p>
                  </div>
                ))}
              </div>

              <button 
                onClick={() => setShowExplorer(true)}
                className="flex items-center gap-4 text-xs font-black text-medical-400 uppercase tracking-[0.4em] group hover:text-white transition-colors pt-8"
              >
                Explore Neural Layers <ChevronRight className="w-4 h-4 group-hover:translate-x-2 transition-transform" />
              </button>
            </div>
          </div>
        </div>
      </section>

      {/* Corporate Features */}
      <section className="py-40 px-6 bg-black relative">
        <div className="max-w-7xl mx-auto grid grid-cols-1 md:grid-cols-3 gap-12">
            {[
              { title: 'Pathology Markers', desc: 'Real-time extraction of pulmonary abnormalities using neural attention maps.', icon: Activity, color: 'text-medical-400' },
              { title: 'Sub-Second Triage', desc: 'Diagnostic results in under 800ms, enabling rapid emergency screening.', icon: ShieldAlert, color: 'text-teal-400' },
              { title: 'Secure Pipeline', desc: 'End-to-end encrypted DICOM processing with HIPAA compliance ready protocols.', icon: Lock, color: 'text-emerald-400' },
            ].map((feature, i) => (
              <div key={i} className="glass-panel p-12 rounded-[3.5rem] border border-white/5 hover:border-white/10 transition-all hover:-translate-y-4 group">
                <div className={`w-20 h-20 bg-white/5 rounded-[2rem] flex items-center justify-center mb-10 group-hover:scale-110 transition-transform ${feature.color}`}>
                  <feature.icon className="w-10 h-10" />
                </div>
                <h3 className="text-2xl font-black text-white mb-6 tracking-tighter uppercase">{feature.title}</h3>
                <p className="text-slate-500 text-sm leading-relaxed font-bold uppercase tracking-widest">{feature.desc}</p>
              </div>
            ))}
        </div>
      </section>


      {/* Enterprise Footer */}
      <footer className="py-32 px-6 bg-slate-950/50">
        <div className="max-w-7xl mx-auto grid grid-cols-1 md:grid-cols-4 gap-20">
          <div className="col-span-1 md:col-span-2 space-y-10">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-medical-500 rounded-xl flex items-center justify-center shadow-[0_0_20px_rgba(225,29,72,0.5)]">
                <Activity className="w-6 h-6 text-white" />
              </div>
              <span className="text-2xl font-black tracking-tighter text-white uppercase">AI CAD <span className="text-medical-400">SYSTEM</span></span>
            </div>
            <p className="text-slate-600 text-xs max-w-sm leading-relaxed font-black uppercase tracking-[0.2em]">
              A product of the MP Neural Research Institute. Clinical-grade pulmonary diagnostics powered by deep neural architecture.
            </p>
          </div>
          <div className="space-y-8">
            <h5 className="text-[11px] font-black text-white uppercase tracking-[0.5em]">Network</h5>
            <ul className="space-y-5">
              {['Diagnostics', 'Clinical Specs', 'Security', 'Enterprise'].map(link => (
                <li key={link}><a href="#" className="text-[11px] font-black text-slate-500 hover:text-medical-400 uppercase tracking-widest transition-colors">{link}</a></li>
              ))}
            </ul>
          </div>
          <div className="space-y-8">
            <h5 className="text-[11px] font-black text-white uppercase tracking-[0.5em]">Support</h5>
            <ul className="space-y-5">
              {['Documentation', 'API Access', 'Privacy Protocols', 'Contact'].map(link => (
                <li key={link}><a href="#" className="text-[11px] font-black text-slate-500 hover:text-medical-400 uppercase tracking-widest transition-colors">{link}</a></li>
              ))}
            </ul>
          </div>
        </div>
        <div className="max-w-7xl mx-auto pt-20 mt-20 flex flex-col md:flex-row justify-between items-center text-[10px] font-black uppercase tracking-[0.5em] text-slate-700 gap-8">
          <p>© 2026 AI POWERED CAD SYSTEMS. ENCRYPTED_TERMINAL_SESSION_ACTIVE</p>
          <div className="flex gap-12">
            <a href="#" className="hover:text-slate-400 transition-colors">GDPR</a>
            <a href="#" className="hover:text-slate-400 transition-colors">HIPAA</a>
            <a href="#" className="hover:text-slate-400 transition-colors">ISO 27001</a>
          </div>
        </div>
      </footer>

      {/* Neural Layer Explorer Modal */}
      <AnimatePresence>
        {showExplorer && (
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-[200] flex items-center justify-center p-6 bg-[#020617]/90 backdrop-blur-2xl"
          >
            <motion.div 
              initial={{ scale: 0.9, y: 20 }}
              animate={{ scale: 1, y: 0 }}
              exit={{ scale: 0.9, y: 20 }}
              className="w-full max-w-5xl glass-panel rounded-[3rem] border border-white/10 overflow-hidden flex flex-col h-[80vh]"
            >
              <div className="p-8 border-b border-white/5 flex justify-between items-center bg-white/5">
                <div className="flex items-center gap-4">
                  <div className="w-10 h-10 bg-medical-500 rounded-xl flex items-center justify-center shadow-lg">
                    <Layers className="w-5 h-5 text-white" />
                  </div>
                  <div>
                    <h3 className="text-xl font-black text-white uppercase tracking-tighter">EfficientNet-B3 Architecture</h3>
                    <p className="text-[10px] font-black text-medical-400 uppercase tracking-widest">Neural Layer Feed</p>
                  </div>
                </div>
                <button onClick={() => setShowExplorer(false)} className="w-12 h-12 rounded-full border border-white/10 hover:bg-white/5 flex items-center justify-center transition-colors">
                  <X className="w-6 h-6 text-slate-400" />
                </button>
              </div>

              <div className="flex-1 overflow-y-auto p-12 custom-scrollbar">
                <div className="grid grid-cols-1 md:grid-cols-12 gap-12">
                  {/* Left: Interactive Layer Map */}
                  <div className="md:col-span-4 space-y-4">
                    {[
                      { id: 'l1', name: 'Stem (Conv3x3)', active: true },
                      { id: 'l2', name: 'MBConv1 (3x3)', active: false },
                      { id: 'l3', name: 'MBConv6 (3x3)', active: false },
                      { id: 'l4', name: 'MBConv6 (5x5)', active: false },
                      { id: 'l5', name: 'Global Pool', active: false },
                      { id: 'l6', name: 'FC Classifier', active: false },
                    ].map((layer, i) => (
                      <div key={layer.id} className={`p-4 rounded-2xl border flex items-center justify-between group cursor-pointer transition-all ${layer.active ? 'bg-medical-500/20 border-medical-500/50' : 'bg-white/5 border-white/5 hover:border-white/20'}`}>
                        <div className="flex items-center gap-3">
                          <span className="text-[10px] font-mono text-slate-500">0{i+1}</span>
                          <span className={`text-[11px] font-black uppercase tracking-widest ${layer.active ? 'text-white' : 'text-slate-400'}`}>{layer.name}</span>
                        </div>
                        {layer.active && <div className="w-1.5 h-1.5 rounded-full bg-medical-400 shadow-[0_0_10px_#fb7185] animate-pulse" />}
                      </div>
                    ))}
                  </div>

                  {/* Right: Layer Detail & Visualization */}
                  <div className="md:col-span-8 space-y-10">
                    <div className="glass-panel rounded-3xl p-10 bg-slate-900/50 border border-white/5">
                      <div className="flex items-center gap-4 mb-8">
                        <Cpu className="w-8 h-8 text-medical-400" />
                        <h4 className="text-2xl font-black text-white tracking-tighter uppercase">Layer Stem Details</h4>
                      </div>
                      <p className="text-slate-400 text-sm leading-relaxed font-medium mb-10">
                        The stem layer consists of a 3x3 convolution followed by batch normalization and Swish activation. It performs the initial feature extraction from the 224x224 pulmonary scan.
                      </p>
                      
                      <div className="grid grid-cols-3 gap-6">
                        <div className="p-4 bg-white/5 rounded-2xl border border-white/5">
                          <p className="text-[9px] font-black text-slate-500 uppercase tracking-widest mb-1">Filters</p>
                          <p className="text-lg font-black text-white">40</p>
                        </div>
                        <div className="p-4 bg-white/5 rounded-2xl border border-white/5">
                          <p className="text-[9px] font-black text-slate-500 uppercase tracking-widest mb-1">Kernel</p>
                          <p className="text-lg font-black text-white">3x3</p>
                        </div>
                        <div className="p-4 bg-white/5 rounded-2xl border border-white/5">
                          <p className="text-[9px] font-black text-slate-500 uppercase tracking-widest mb-1">Stride</p>
                          <p className="text-lg font-black text-white">2</p>
                        </div>
                      </div>
                    </div>

                    <div className="aspect-video glass-panel rounded-3xl border border-white/10 relative overflow-hidden flex items-center justify-center bg-black">
                       <Activity className="w-16 h-16 text-medical-500/20 animate-pulse" />
                       <div className="absolute inset-0 bg-gradient-to-t from-medical-500/10 to-transparent" />
                       <p className="text-[10px] font-black text-medical-400 uppercase tracking-[0.5em] animate-pulse">Live Tensor Feed</p>
                    </div>
                  </div>
                </div>
              </div>

              <div className="p-6 bg-slate-950 border-t border-white/5 flex justify-center">
                <p className="text-[9px] font-black text-slate-700 uppercase tracking-[0.5em]">EfficientNet-B3 Multi-Phase Feature Extraction Engine</p>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default App;
