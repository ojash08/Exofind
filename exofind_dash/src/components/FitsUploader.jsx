import React, { useState, useCallback, useRef } from 'react';
import { Upload, FileUp, X, AlertCircle, CheckCircle2 } from 'lucide-react';
import { uploadFitsFile, runPipeline, getResults } from '../services/api';

export default function FitsUploader({ onDataLoaded, onClose }) {
  const [dragOver, setDragOver] = useState(false);
  const [status, setStatus] = useState(null); // null | 'uploading' | 'processing' | 'success' | 'error'
  const [message, setMessage] = useState('');
  const [progressLog, setProgressLog] = useState([]);
  const fileInputRef = useRef(null);

  const handleFile = useCallback(async (file) => {
    if (!file) return;

    // Validate extension
    const lowerName = file.name.toLowerCase();
    if (!lowerName.endsWith('.fits') && !lowerName.endsWith('.fit') && !lowerName.endsWith('.fits.gz') && !lowerName.endsWith('.fts')) {
      setStatus('error');
      setMessage(`Invalid file type: "${file.name}". Only .fits, .fts, or .fits.gz files are accepted.`);
      return;
    }

    try {
      setStatus('uploading');
      setMessage(`Uploading ${file.name} to pipeline...`);
      setProgressLog([{ stage: 'Upload', status: 'running' }]);

      // 1. Upload File
      const { job_id, file_path } = await uploadFitsFile(file);
      
      setProgressLog(prev => prev.map(p => p.stage === 'Upload' ? { ...p, status: 'done' } : p));
      setStatus('processing');
      setMessage('Running ExoFind Pipeline (this may take 30-120s)...');

      // 2. Connect to SSE and run pipeline
      await runPipeline(job_id, file_path, (event) => {
        setProgressLog(prev => {
          // Check if stage already exists in log
          const existing = prev.find(p => p.stage === event.stage);
          if (existing) {
            return prev.map(p => p.stage === event.stage ? { ...p, ...event } : p);
          } else {
            return [...prev, event];
          }
        });
      });

      // 3. Fetch final results
      setMessage('Pipeline complete. Fetching results...');
      const results = await getResults(job_id);

      setStatus('success');
      setMessage(`Successfully processed ${file.name}. Found ${results.candidates.length} candidates.`);
      
      onDataLoaded(results);

    } catch (e) {
      setStatus('error');
      setMessage(e.message || 'An unknown error occurred');
      setProgressLog(prev => [...prev, { stage: 'Error', status: 'error', message: e.message }]);
    }
  }, [onDataLoaded]);

  const onDrop = useCallback((e) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    handleFile(file);
  }, [handleFile]);

  const onDragOver = useCallback((e) => {
    e.preventDefault();
    setDragOver(true);
  }, []);

  const onDragLeave = useCallback(() => {
    setDragOver(false);
  }, []);

  const onFileSelect = (e) => {
    const file = e.target.files[0];
    if (file) handleFile(file);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm">
      <div className="bg-[#161618] border border-zinc-800 rounded-lg w-full max-w-lg mx-4 flex flex-col max-h-[90vh]">
        
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-zinc-800 shrink-0">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-cyan-500/10 border border-cyan-500/20 rounded-md">
              <Upload size={16} className="text-cyan-400" />
            </div>
            <div>
              <h2 className="text-sm font-semibold text-white">Upload TPF / Light Curve</h2>
              <p className="text-[10px] text-zinc-500 uppercase tracking-widest mt-0.5">Automated Pipeline Run</p>
            </div>
          </div>
          <button 
            onClick={onClose}
            className="p-1.5 text-zinc-500 hover:text-zinc-300 hover:bg-zinc-800 rounded-md transition-colors"
          >
            <X size={16} />
          </button>
        </div>

        {/* Content (Scrollable) */}
        <div className="p-5 overflow-y-auto">
          
          {/* Drop Zone */}
          {(!status || status === 'error') && (
            <div
              onDrop={onDrop}
              onDragOver={onDragOver}
              onDragLeave={onDragLeave}
              onClick={() => fileInputRef.current?.click()}
              className={`
                relative border-2 border-dashed rounded-md p-10 text-center cursor-pointer transition-all duration-200 mb-5
                ${dragOver 
                  ? 'border-cyan-500 bg-cyan-500/5' 
                  : 'border-zinc-700 hover:border-zinc-600 bg-[#111113]'}
              `}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".fits,.fit,.fits.gz,.fts"
                onChange={onFileSelect}
                className="hidden"
              />
              
              <FileUp size={32} className={`mx-auto mb-4 ${dragOver ? 'text-cyan-400' : 'text-zinc-600'} transition-colors`} />
              
              <p className="text-sm text-zinc-300 mb-1">
                {dragOver ? 'Release to upload' : 'Drag & drop Target Pixel File (TPF)'}
              </p>
              <p className="text-xs text-zinc-600">
                or click to browse • Accepts .fits files
              </p>
            </div>
          )}

          {/* Status Panel */}
          {status && (
            <div className={`p-4 rounded-md border flex flex-col gap-3 ${
              status === 'error' ? 'border-red-500/30 bg-red-500/5' :
              status === 'success' ? 'border-green-500/30 bg-green-500/5' :
              'border-zinc-700 bg-zinc-900'
            }`}>
              
              <div className="flex items-start gap-3">
                {(status === 'uploading' || status === 'processing') && (
                  <div className="w-4 h-4 border-2 border-cyan-500 border-t-transparent rounded-full animate-spin shrink-0 mt-0.5" />
                )}
                {status === 'success' && <CheckCircle2 size={16} className="text-green-500 shrink-0 mt-0.5" />}
                {status === 'error' && <AlertCircle size={16} className="text-red-400 shrink-0 mt-0.5" />}
                
                <div className="flex-1 min-w-0">
                  <p className={`text-sm font-medium ${
                    status === 'error' ? 'text-red-400' :
                    status === 'success' ? 'text-green-400' :
                    'text-zinc-200'
                  }`}>
                    {message}
                  </p>
                </div>
              </div>

              {/* Progress Log */}
              {progressLog.length > 0 && (
                <div className="mt-2 flex flex-col gap-1.5 border-t border-zinc-800/50 pt-3">
                  {progressLog.map((log, i) => (
                    <div key={i} className="flex items-center justify-between text-xs font-mono">
                      <span className="text-zinc-400">{log.stage}</span>
                      {log.status === 'running' && <span className="text-cyan-500 animate-pulse">Running...</span>}
                      {log.status === 'done' && <span className="text-zinc-500">Done</span>}
                      {log.status === 'error' && <span className="text-red-400 truncate max-w-[200px]" title={log.message}>Error</span>}
                    </div>
                  ))}
                </div>
              )}

            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-5 py-3 border-t border-zinc-800 flex justify-between items-center bg-[#111113] shrink-0">
          <p className="text-[10px] text-zinc-600 uppercase tracking-widest">
            Pipeline: KZM 2002 BLS (Astropy backend)
          </p>
          {status === 'success' && (
            <button
              onClick={onClose}
              className="px-4 py-1.5 bg-cyan-500/10 border border-cyan-500/30 text-cyan-400 text-xs font-medium rounded-md hover:bg-cyan-500/20 transition-colors"
            >
              View Dashboard
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
