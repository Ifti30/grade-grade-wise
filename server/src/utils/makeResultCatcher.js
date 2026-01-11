// makeResultCatcher.js
export function makeResultCatcher({ onResult, onProgress, onError }) {
    let buffer = '';
    let result = null;
    let error = null;
  
    function tryParseJSON(line) {
      try {
        return JSON.parse(line);
      } catch {
        return null;
      }
    }
  
    function write(chunk) {
      buffer += chunk;
  
      let idx;
      while ((idx = buffer.indexOf('\n')) !== -1) {
        const line = buffer.slice(0, idx).trim();
        buffer = buffer.slice(idx + 1);
  
        if (line.startsWith('__PROGRESS__')) {
          const payload = tryParseJSON(line.slice(12));
          if (payload && onProgress) onProgress(payload);
        }
  
        else if (line.startsWith('__RESULT__')) {
          const payload = tryParseJSON(line.slice(10));
          if (payload) {
            result = payload;
            onResult?.(payload);
          } else {
            error = new Error('Invalid RESULT JSON');
            onError?.(error);
          }
        }
      }
  
      if (buffer.length > 1_000_000 && !buffer.includes('__RESULT__')) {
        buffer = buffer.slice(-100_000);
      }
    }
  
    function getResult() {
      return { result, error };
    }
  
    return { write, getResult };
  }
  