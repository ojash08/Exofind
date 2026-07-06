const API_BASE = 'http://localhost:8000/api';

/**
 * Upload a FITS file to the backend
 * @param {File} file 
 * @returns {Promise<{job_id: str, filename: str, file_path: str}>}
 */
export async function uploadFitsFile(file) {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await fetch(`${API_BASE}/upload`, {
    method: 'POST',
    body: formData
  });
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Upload failed');
  }
  
  return await response.json();
}

/**
 * Run the pipeline and listen for SSE progress events
 * @param {string} jobId 
 * @param {string} filePath 
 * @param {function} onProgress Callback for progress events
 * @returns {Promise<void>} Resolves when the stream is complete
 */
export function runPipeline(jobId, filePath, onProgress) {
  return new Promise((resolve, reject) => {
    // URL encode the file path to prevent issues
    const url = `${API_BASE}/run/${jobId}?file_path=${encodeURIComponent(filePath)}`;
    const eventSource = new EventSource(url);
    
    eventSource.addEventListener('progress', (e) => {
      const data = JSON.parse(e.data);
      onProgress(data);
    });
    
    eventSource.addEventListener('complete', (e) => {
      eventSource.close();
      resolve(JSON.parse(e.data));
    });
    
    eventSource.addEventListener('error', (e) => {
      eventSource.close();
      if (e.data) {
        reject(new Error(JSON.parse(e.data).message));
      } else {
        reject(new Error("Connection error or pipeline failed."));
      }
    });
  });
}

/**
 * Fetch the final results from the backend
 * @param {string} jobId 
 * @returns {Promise<Object>}
 */
export async function getResults(jobId) {
  const response = await fetch(`${API_BASE}/results/${jobId}`);
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to fetch results');
  }
  
  return await response.json();
}
