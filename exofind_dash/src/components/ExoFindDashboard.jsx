import React, { useState, useMemo } from 'react';
import { 
  Search, 
  Filter, 
  Settings2, 
  User,
  CheckCircle2,
  Mail,
  ChevronDown,
  ArrowUpDown,
  Telescope,
  Upload,
  LogOut,
  XCircle,
  Download
} from 'lucide-react';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  ZAxis,
  ComposedChart,
  AreaChart,
  Area,
  ReferenceLine
} from 'recharts';
import FitsUploader from './FitsUploader';

export default function ExoFindDashboard() {
  const [showUploader, setShowUploader] = useState(false);
  const [backendResults, setBackendResults] = useState(null);
  const [imageTimestamp, setImageTimestamp] = useState(Date.now());

  // Auto-refresh the image every 2 seconds so the dashboard acts as a live viewer
  // when the user runs main.py externally in the terminal.
  React.useEffect(() => {
    const interval = setInterval(() => {
      setImageTimestamp(Date.now());
    }, 2000);
    return () => clearInterval(interval);
  }, []);

  // Process light curve from backend
  const lightCurveData = useMemo(() => {
    if (!backendResults || !backendResults.light_curve) return [];
    
    const { time, flux } = backendResults.light_curve;
    
    // Normalize flux relative to median
    const sortedFlux = [...flux].sort((a, b) => a - b);
    const median = sortedFlux[Math.floor(sortedFlux.length / 2)];
    
    return time.map((t, i) => ({
      time: t,
      flux: flux[i] / median
    }));
  }, [backendResults]);

  // Process periodogram from backend
  const periodogramData = useMemo(() => {
    if (!backendResults || !backendResults.periodogram) return [];
    
    const { periods, sde } = backendResults.periodogram;
    return periods.map((p, i) => ({
      period: p,
      sde: sde[i]
    }));
  }, [backendResults]);

  // Process folded curve from backend
  const foldedCurveData = useMemo(() => {
    if (!backendResults || !backendResults.folded_curve) return [];
    
    const { phase, flux } = backendResults.folded_curve;
    // Downsample if it's too large to render smoothly (React Recharts struggles with >10k points)
    const step = Math.max(1, Math.floor(phase.length / 3000));
    
    const data = [];
    for (let i = 0; i < phase.length; i += step) {
      data.push({ phase: phase[i], flux: flux[i] });
    }
    return data;
  }, [backendResults]);

  // Chart domains
  const lcDomain = useMemo(() => {
    if (lightCurveData.length === 0) return { x: [0, 1], y: [0.99, 1.01] };
    const fluxVals = lightCurveData.map(d => d.flux).filter(v => isFinite(v));
    const minF = Math.min(...fluxVals);
    const maxF = Math.max(...fluxVals);
    const pad = (maxF - minF) * 0.1;
    return {
      x: ['dataMin', 'dataMax'],
      y: [minF - pad, maxF + pad]
    };
  }, [lightCurveData]);

  const foldedDomain = useMemo(() => {
    if (foldedCurveData.length === 0) return { x: [0, 1], y: [0.99, 1.01] };
    const fluxVals = foldedCurveData.map(d => d.flux).filter(v => isFinite(v));
    const minF = Math.min(...fluxVals);
    const maxF = Math.max(...fluxVals);
    const pad = (maxF - minF) * 0.1;
    return {
      x: [0, 1],
      y: [minF - pad, maxF + pad]
    };
  }, [foldedCurveData]);

  // Derive stats
  const stats = useMemo(() => {
    if (!backendResults) {
      return {
        target: 'No Data Loaded',
        period: '—',
        depth: '—',
        sde: '—',
        duration: '—',
        baseline: '—',
        cadences: '—',
        status: 'Waiting',
        confidence: null,
        checks: { oe: null, se: null }
      };
    }

    const b = backendResults;
    const c = b.best_candidate;

    let oeDiff = 0;
    let seRatio = 0;

    if (c) {
      oeDiff = c.odd_even_diff_sigma;
      seRatio = c.secondary_ratio;
    }

    // Determine confidence heuristically
    let conf = 0;
    let status = 'False Positive';
    if (c && c.status === 'PASSED') {
      conf = Math.min(99, Math.round((c.sde / 15.0) * 100)); // scale SDE to 100%
      if (conf < 40) conf = 40; // minimum passing
      status = 'Candidate';
    }

    return {
      target: b.target_name,
      period: c ? `${c.period.toFixed(5)} d` : 'None',
      depth: c ? `${(c.depth * 100).toFixed(4)}% (${Math.round(c.depth * 1e6)} ppm)` : '—',
      sde: c ? c.sde.toFixed(2) : '—',
      duration: c ? `${(c.duration_days * 24).toFixed(2)} hrs` : '—',
      baseline: `${b.baseline.toFixed(2)} days`,
      cadences: b.cadences.toLocaleString(),
      status: b.detection ? status : 'No Detection',
      confidence: b.detection ? conf : null,
      checks: {
        oe: c ? oeDiff < 3.0 : null,
        se: c ? seRatio < 0.1 : null
      }
    };
  }, [backendResults]);

  const handleDataLoaded = (data) => {
    setBackendResults(data);
    setShowUploader(false);
  };

  const downloadJSON = () => {
    if (!backendResults) return;
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(backendResults, null, 2));
    const dlAnchorElem = document.createElement('a');
    dlAnchorElem.setAttribute("href", dataStr);
    dlAnchorElem.setAttribute("download", `${backendResults.target_name || 'results'}.json`);
    dlAnchorElem.click();
  };

  const downloadCSV = () => {
    if (!backendResults || !backendResults.candidates) return;
    const candidates = backendResults.candidates;
    if (candidates.length === 0) {
      alert("No candidates to export");
      return;
    }
    
    const headers = Object.keys(candidates[0]);
    const csvRows = [headers.join(',')];
    
    for (const row of candidates) {
      const values = headers.map(header => {
        const val = row[header];
        return typeof val === 'object' ? `"${JSON.stringify(val).replace(/"/g, '""')}"` : `"${val}"`;
      });
      csvRows.push(values.join(','));
    }
    
    const dataStr = "data:text/csv;charset=utf-8," + encodeURIComponent(csvRows.join('\n'));
    const dlAnchorElem = document.createElement('a');
    dlAnchorElem.setAttribute("href", dataStr);
    dlAnchorElem.setAttribute("download", `${backendResults.target_name || 'candidates'}.csv`);
    dlAnchorElem.click();
  };

  return (
    <div className="flex flex-col h-screen w-full bg-[#111113] text-zinc-300 font-sans overflow-hidden">
      
      {/* FITS Upload Modal */}
      {showUploader && (
        <FitsUploader 
          onDataLoaded={handleDataLoaded} 
          onClose={() => setShowUploader(false)} 
        />
      )}
      
      {/* TOP HEADER */}
      <header className="h-14 border-b border-zinc-800 flex items-center justify-between px-4 shrink-0 bg-[#161618]">
        <div className="flex items-center gap-3">
          <Telescope className="text-cyan-500" size={20} />
          <span className="font-semibold text-white tracking-wide">ExoFind <span className="text-zinc-600 font-light px-2">|</span> <span className="text-zinc-400 font-normal">Exoplanet Candidate Dashboard</span></span>
        </div>
        
        <div className="flex items-center gap-4">
          {backendResults && (
            <div className="flex gap-2">
              <button onClick={downloadJSON} className="flex items-center gap-1.5 px-3 py-1.5 bg-zinc-800 hover:bg-zinc-700 text-xs font-medium text-zinc-300 rounded-md transition-colors border border-zinc-700">
                <Download size={14} />
                JSON
              </button>
              <button onClick={downloadCSV} className="flex items-center gap-1.5 px-3 py-1.5 bg-zinc-800 hover:bg-zinc-700 text-xs font-medium text-zinc-300 rounded-md transition-colors border border-zinc-700">
                <Download size={14} />
                CSV
              </button>
            </div>
          )}
          {/* User Menu placeholder (Auth removed) */}
          <div className="relative flex items-center">
            <User size={16} className="text-zinc-500" />
          </div>
        </div>
      </header>

      {/* MAIN BODY */}
      <div className="flex flex-1 overflow-hidden p-4 gap-4 bg-[#111113]">
        
        {/* LEFT SIDEBAR - TARGETS */}
        <aside className="w-64 flex flex-col gap-3 shrink-0">
          
          {/* Upload Button */}
          <button 
            onClick={() => setShowUploader(true)}
            className="w-full border border-dashed border-cyan-500/30 bg-cyan-500/5 p-3 rounded-md flex items-center gap-3 cursor-pointer hover:bg-cyan-500/10 hover:border-cyan-500/50 transition-all group"
          >
            <div className="p-1.5 bg-cyan-500/10 rounded group-hover:bg-cyan-500/20 transition-colors">
              <Upload size={14} className="text-cyan-400" />
            </div>
            <div className="text-left">
              <div className="text-xs font-medium text-cyan-400">Run Pipeline</div>
              <div className="text-[9px] text-zinc-600 mt-0.5">Upload TPF to analyze</div>
            </div>
          </button>
          
          <div className="flex flex-col gap-2 overflow-y-auto pr-1">
            {/* Selected Target */}
            {backendResults && (
              <div className="border border-cyan-500/30 bg-cyan-950/20 p-3 rounded-md flex gap-3 cursor-pointer relative overflow-hidden group">
                <div className="absolute left-0 top-0 bottom-0 w-1 bg-cyan-500"></div>
                <div className="mt-0.5 opacity-70"><Mail size={14} className="text-cyan-400" /></div>
                <div>
                  <div className="text-sm font-medium text-white tracking-wide">{stats.target}</div>
                  <div className="text-[10px] text-cyan-500/80 font-mono mt-0.5 tracking-wider">[LOADED]</div>
                </div>
              </div>
            )}

            {/* Candidates List */}
            {backendResults?.candidates.length > 0 && (
              <div className="mt-2">
                <h3 className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest mb-2 px-1">Raw Peaks Detected</h3>
                {backendResults.candidates.slice(0, 5).map((cand, i) => (
                  <div key={i} className={`border border-zinc-800/80 p-3 mb-2 rounded-md flex flex-col gap-1 ${cand.status === 'PASSED' ? 'bg-green-500/5 border-green-500/20' : 'bg-[#161618]'}`}>
                    <div className="flex justify-between items-center">
                      <span className="text-xs font-mono font-bold text-zinc-300">{cand.period.toFixed(4)} d</span>
                      <span className="text-[10px] font-mono text-zinc-500">SDE: {cand.sde.toFixed(1)}</span>
                    </div>
                    <div className="text-[9px] text-zinc-500 uppercase">{cand.status}</div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </aside>

        {/* RIGHT COLUMN - CHARTS & STATS */}
        <main className="flex-1 flex flex-col gap-4 min-w-0">
          
          {/* TOP CHARTS ROW */}
          <div className="flex-[2] flex gap-4 min-h-0">
            
            {/* Generated Plot Image */}
            <div className="flex-1 border border-zinc-800 rounded-md bg-[#161618] flex flex-col relative overflow-hidden min-h-0">
              <div className="flex justify-between items-center p-3 border-b border-zinc-800/50 shrink-0">
                <h2 className="text-[10px] font-bold text-zinc-400 uppercase tracking-widest">
                  ExoFind Pipeline Results (SDE, SR, Folded Phase)
                </h2>
                {backendResults?.best_candidate && (
                  <div className="text-[10px] font-mono text-cyan-400">
                    Best Candidate P = {backendResults.best_candidate.period.toFixed(4)} d
                  </div>
                )}
              </div>
              
              <div className="flex-1 w-full p-2 bg-[#ffffff]/5 overflow-y-auto flex justify-center items-start min-h-0">
                <img 
                  src={`http://localhost:8000/api/results/image?t=${imageTimestamp}`}
                  alt="BLS Pipeline Results" 
                  className="w-full h-auto object-contain rounded drop-shadow-lg" 
                  onError={(e) => {
                    // Hide broken image icon if it doesn't exist yet
                    e.target.style.display = 'none';
                  }}
                  onLoad={(e) => {
                    e.target.style.display = 'block';
                  }}
                />
              </div>
            </div>
            
          </div>

          {/* BOTTOM PANELS ROW */}
          <div className="flex-1 grid grid-cols-4 gap-4">
            
            {/* Panel 1: Periodogram */}
            <div className="border border-zinc-800 rounded-md bg-[#161618] p-3 flex flex-col">
              <h2 className="text-[10px] font-bold text-zinc-400 uppercase tracking-widest mb-2">Periodogram (SDE)</h2>
              <div className="flex-1 w-full relative">
                {periodogramData.length > 0 ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={periodogramData}>
                      <CartesianGrid strokeDasharray="2 2" stroke="#27272a" />
                      <XAxis 
                        dataKey="period" 
                        type="number"
                        domain={['dataMin', 'dataMax']}
                        tick={{fill: '#71717a', fontSize: 9}}
                        tickCount={4}
                      />
                      <YAxis hide domain={[0, 'dataMax']} />
                      <ReferenceLine y={6.0} stroke="#ef4444" strokeDasharray="3 3" />
                      <Area 
                        type="monotone" 
                        dataKey="sde" 
                        stroke="#06b6d4" 
                        fill="#06b6d4" 
                        fillOpacity={0.1}
                        strokeWidth={1.5}
                        isAnimationActive={false}
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="w-full h-full flex items-center justify-center text-zinc-600 text-xs">No data</div>
                )}
              </div>
            </div>

            {/* Panel 2: Candidate Statistics */}
            <div className="border border-zinc-800 rounded-md bg-[#161618] p-3 flex flex-col">
              <h2 className="text-[10px] font-bold text-zinc-400 uppercase tracking-widest mb-3">Best Candidate Stats</h2>
              <div className="flex-1 flex flex-col justify-between text-xs">
                {[
                  { label: 'Period:', value: stats.period },
                  { label: 'Depth:', value: stats.depth },
                  { label: 'SDE Score:', value: stats.sde },
                  { label: 'Duration:', value: stats.duration },
                  { label: 'Baseline:', value: stats.baseline },
                ].map((stat, i) => (
                  <div key={i} className="flex justify-between border-b border-zinc-800/50 pb-1">
                    <span className="text-zinc-500 font-medium">{stat.label}</span>
                    <span className={`font-mono ${stat.value === '—' || stat.value === 'Pending BLS' ? 'text-zinc-600' : 'text-zinc-200'}`}>
                      {stat.value}
                    </span>
                  </div>
                ))}
              </div>
            </div>

            {/* Panel 3: AI Confidence */}
            <div className="border border-zinc-800 rounded-md bg-[#161618] p-3 flex flex-col items-center justify-between">
              <h2 className="text-[10px] font-bold text-zinc-400 uppercase tracking-widest w-full text-left">Confidence Score</h2>
              
              <div className="relative w-32 h-16 mt-2 flex justify-center overflow-hidden">
                <div className="w-32 h-32 rounded-full border-[12px] border-zinc-800 absolute top-0"></div>
                {stats.confidence !== null ? (
                  <div className={`w-32 h-32 rounded-full border-[12px] border-transparent absolute top-0 rotate-45 ${stats.confidence > 75 ? 'border-t-green-500 border-l-green-500' : 'border-t-cyan-500 border-l-cyan-500'}`}></div>
                ) : (
                  <div className="w-32 h-32 rounded-full border-[12px] border-transparent border-t-amber-500/30 absolute top-0 rotate-45"></div>
                )}
                <div className="absolute bottom-0 text-3xl font-mono text-white font-light tracking-tight">
                  {stats.confidence !== null ? (
                    <>{stats.confidence}<span className="text-lg text-zinc-500">%</span></>
                  ) : (
                    <span className="text-lg text-amber-400/70">—</span>
                  )}
                </div>
              </div>
              
              <div className="text-[10px] font-bold tracking-widest text-zinc-500 uppercase mt-2">
                Status: <span className={stats.status === 'Candidate' ? 'text-green-500' : (stats.status === 'Waiting' ? 'text-zinc-500' : 'text-amber-400')}>{stats.status}</span>
              </div>

              {/* Mini bar chart */}
              <div className="w-full flex items-end justify-between h-8 mt-2 px-2 gap-1 border-t border-zinc-800/50 pt-2 opacity-50">
                {[30, 40, 25, 45, 60, 50, 80, stats.confidence || 10].map((h, i) => (
                  <div key={i} className={`w-full rounded-t-sm ${i === 7 ? (stats.confidence ? 'bg-cyan-500' : 'bg-amber-500/40') : 'bg-zinc-600'}`} style={{ height: `${h}%` }}></div>
                ))}
              </div>
            </div>

            {/* Panel 4: False-Positive Checks */}
            <div className="border border-zinc-800 rounded-md bg-[#161618] p-3 flex flex-col">
              <h2 className="text-[10px] font-bold text-zinc-400 uppercase tracking-widest mb-3">Vetting Checks</h2>
              <div className="flex flex-col gap-2.5 flex-1 justify-center">
                
                {/* Odd Even */}
                <div className="flex items-center justify-between group">
                  <div className="flex items-center gap-2">
                    <Mail size={12} className="text-zinc-600 group-hover:text-zinc-400 transition-colors" />
                    <span className="text-xs text-zinc-300 font-medium tracking-wide">Odd/Even Mismatch</span>
                  </div>
                  {stats.checks.oe === null ? (
                    <span className="text-[9px] text-zinc-600 font-mono">WAITING</span>
                  ) : stats.checks.oe ? (
                    <CheckCircle2 size={14} className="text-green-500" />
                  ) : (
                    <XCircle size={14} className="text-red-500" />
                  )}
                </div>

                {/* Secondary Eclipse */}
                <div className="flex items-center justify-between group">
                  <div className="flex items-center gap-2">
                    <Mail size={12} className="text-zinc-600 group-hover:text-zinc-400 transition-colors" />
                    <span className="text-xs text-zinc-300 font-medium tracking-wide">Secondary Eclipse</span>
                  </div>
                  {stats.checks.se === null ? (
                    <span className="text-[9px] text-zinc-600 font-mono">WAITING</span>
                  ) : stats.checks.se ? (
                    <CheckCircle2 size={14} className="text-green-500" />
                  ) : (
                    <XCircle size={14} className="text-red-500" />
                  )}
                </div>

                {/* SDE Threshold */}
                <div className="flex items-center justify-between group">
                  <div className="flex items-center gap-2">
                    <Mail size={12} className="text-zinc-600 group-hover:text-zinc-400 transition-colors" />
                    <span className="text-xs text-zinc-300 font-medium tracking-wide">SDE Threshold ({'>'} 6.0)</span>
                  </div>
                  {backendResults ? (
                    backendResults.best_candidate && backendResults.best_candidate.sde > 6.0 ? (
                      <CheckCircle2 size={14} className="text-green-500" />
                    ) : (
                      <XCircle size={14} className="text-red-500" />
                    )
                  ) : (
                    <span className="text-[9px] text-zinc-600 font-mono">WAITING</span>
                  )}
                </div>

              </div>
            </div>

          </div>
          
        </main>
      </div>
      
    </div>
  );
}
