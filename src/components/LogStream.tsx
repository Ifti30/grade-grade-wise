import { useEffect, useRef, useState } from 'react';
import { Card } from '@/components/ui/card';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Terminal } from 'lucide-react';
import { api } from '@/lib/api';

interface LogStreamProps {
  url: string;
  token: string;
  onComplete?: (status: string) => void;
  runId?: string;
}

export function LogStream({ url, token, onComplete, runId }: LogStreamProps) {
  const [logs, setLogs] = useState<string>('');
  const [status, setStatus] = useState<'connecting' | 'streaming' | 'complete' | 'error'>('connecting');
  const [streamToken, setStreamToken] = useState(token);
  const [summaryReady, setSummaryReady] = useState(false);
  const [summaryProgress, setSummaryProgress] = useState(0);
  const scrollRef = useRef<HTMLPreElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const lastEventIdRef = useRef(0);
  const summaryLoggedRef = useRef<Set<string>>(new Set());

  const renderLine = (display: string, index: number) => {
    if (!display) {
      return <div key={`spacer-${index}`} className="h-2" />;
    }

    const progressMatch = display.match(/^\[PROGRESS\]\s+phase=([^\s]+)(.*)$/);
    if (progressMatch) {
      const phase = progressMatch[1];
      const rest = progressMatch[2]?.trim();
      const color =
        phase === 'training_start' ? 'text-amber-600' :
          phase === 'model_trained' ? 'text-emerald-600' :
            phase === 'training' ? 'text-sky-600' :
              phase === 'data_ready' ? 'text-violet-600' :
                'text-foreground';
      return (
        <div key={`progress-${index}`} className={`font-semibold ${color}`}>
          {phase.toUpperCase()}{rest ? ` — ${rest}` : ''}
        </div>
      );
    }

    if (display.startsWith('[ERROR]')) {
      return (
        <div key={`err-${index}`} className="text-destructive font-semibold">
          {display}
        </div>
      );
    }

    if (display.startsWith('[WARN]')) {
      return (
        <div key={`warn-${index}`} className="text-amber-600">
          {display}
        </div>
      );
    }

    if (display.startsWith('[INFO]')) {
      return (
        <div key={`info-${index}`} className="text-muted-foreground">
          {display}
        </div>
      );
    }

    if (display.startsWith('[LightGBM]')) {
      return (
        <div key={`lgbm-${index}`} className="text-emerald-600/80">
          {display}
        </div>
      );
    }

    return (
      <div key={`line-${index}`} className="text-foreground/90">
        {display}
      </div>
    );
  };

  const renderLogs = () => {
    const lines = logs.split('\n');
    const nodes: React.ReactNode[] = [];
    let epochBuffer: string[] = [];
    const epochChunkSize = 4;

    const flushEpochBuffer = () => {
      if (!epochBuffer.length) return;
      const chunks: string[][] = [];
      for (let i = 0; i < epochBuffer.length; i += epochChunkSize) {
        chunks.push(epochBuffer.slice(i, i + epochChunkSize));
      }
      chunks.forEach((chunk) => {
        nodes.push(
          <div key={`epoch-${nodes.length}`} className="flex flex-wrap text-foreground/90">
            {chunk.map((entry, entryIndex) => (
              <span
                key={`${entry}-${entryIndex}`}
                className={entryIndex === 0 ? '' : 'ml-2 pl-2 border-l-2 border-sky-500/70'}
              >
                {entry}
              </span>
            ))}
          </div>
        );
      });
      epochBuffer = [];
    };

    lines.forEach((line, index) => {
      const trimmedEnd = line.trimEnd();
      const trimmedStart = trimmedEnd.trimStart();
      if (!trimmedStart) {
        flushEpochBuffer();
        nodes.push(<div key={`spacer-${index}`} className="h-2" />);
        return;
      }

      if (trimmedStart.startsWith('epoch=') || trimmedStart.startsWith('valLoss=')) {
        epochBuffer.push(trimmedStart);
        return;
      }

      flushEpochBuffer();

      nodes.push(renderLine(trimmedStart, index));
    });

    flushEpochBuffer();

    return nodes;
  };

  useEffect(() => {
    setStreamToken(token);
  }, [token]);

  useEffect(() => {
    let cancelled = false;
    let eventSource: EventSource | null = null;
    let didRetry = false;
    lastEventIdRef.current = 0;
    setLogs('');

    const openStream = (authToken: string) => {
      const offset = lastEventIdRef.current;
      const query = `token=${encodeURIComponent(authToken)}${offset > 0 ? `&offset=${offset}` : ''}`;
      eventSource = new EventSource(`${url}?${query}`);

      eventSource.onopen = () => {
        setStatus('streaming');
      };

      eventSource.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (event.lastEventId) {
            const parsed = Number(event.lastEventId);
            if (Number.isFinite(parsed) && parsed >= 0) {
              lastEventIdRef.current = Math.floor(parsed);
            }
          }

          if (data.content) {
            setLogs((prev) => prev + data.content);
          }

          if (data.complete) {
            setStatus('complete');
            onComplete?.(data.status);
            eventSource?.close();
          }
        } catch (error) {
          console.error('Failed to parse log event:', error);
        }
      };

      eventSource.onerror = async () => {
        if (!didRetry) {
          didRetry = true;
          eventSource?.close();
          const refreshed = await api.refreshToken();
          if (!cancelled && refreshed) {
            const updatedToken = localStorage.getItem('token') || '';
            setStreamToken(updatedToken);
            openStream(updatedToken);
            return;
          }
        }
        setStatus('error');
        eventSource?.close();
      };
    };

    const init = async () => {
      setStatus('connecting');
      let nextToken = streamToken;
      if (!nextToken) {
        const refreshed = await api.refreshToken();
        nextToken = refreshed ? (localStorage.getItem('token') || '') : '';
      }
      if (!cancelled && nextToken) {
        openStream(nextToken);
      } else if (!cancelled) {
        setStatus('error');
      }
    };

    init();

    return () => {
      cancelled = true;
      eventSource?.close();
    };
  }, [url, onComplete, streamToken]);

  useEffect(() => {
    // Auto-scroll to bottom
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [logs]);

  useEffect(() => {
    if (status !== 'complete' || !runId) {
      setSummaryReady(false);
      setSummaryProgress(0);
      summaryLoggedRef.current = new Set();
      return;
    }

    let cancelled = false;
    const key = `summaryReady:${runId}`;
    const authToken = localStorage.getItem('token') || token;
    const url = `${api.getModelSummaryStreamUrl(runId)}&token=${encodeURIComponent(authToken)}`;
    const source = new EventSource(url);
    let totalChunks = 0;

    source.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        const logOnce = (id: string, message: string) => {
          if (summaryLoggedRef.current.has(id)) return;
          summaryLoggedRef.current.add(id);
          setLogs((prev) => `${prev}\n[SUMMARY] ${message}`);
        };
        if (data.type === 'start') {
          totalChunks = Number(data.totalChunks) || 0;
          setSummaryProgress(0);
        }
        if (data.type === 'complete') {
          setSummaryProgress(1);
          localStorage.setItem(key, 'true');
          setSummaryReady(true);
          source.close();
          return;
        }
        if (data.type === 'metrics') {
          logOnce('metrics', 'Accuracy / RMSE / MAE / R²');
        }
        if (data.type === 'plots') {
          logOnce('plots', 'Plots');
        }
        if (data.type === 'config') {
          logOnce('config', 'Training Config');
        }
        if (data.type === 'meta') {
          logOnce('meta', 'Summary Metadata');
        }
        if (data.type === 'report' && data.path) {
          const path = String(data.path);
          if (path.startsWith('dataset.final_cgpa_hist')) {
            logOnce('dataset.final_cgpa_hist', 'Final CGPA Histogram');
          } else if (path.startsWith('dataset.next_sem_cgpa_hist')) {
            logOnce('dataset.next_sem_cgpa_hist', 'Next-Sem CGPA Histogram');
          } else if (path === 'dataset.stats') {
            logOnce('dataset.stats', 'Dataset Overview');
          } else if (path.endsWith('.metrics.models')) {
            const task = path.split('.')[1] || 'task';
            logOnce(path, `Model Metrics (${task})`);
          } else if (path.includes('.metrics.predictions.')) {
            const parts = path.split('.');
            const task = parts[1] || 'task';
            const model = parts[4] || 'model';
            logOnce(path, `Predicted vs Actual (${task}, ${model})`);
          } else if (path.includes('.metrics.featureImportance.')) {
            const parts = path.split('.');
            const task = parts[1] || 'task';
            const model = parts[4] || 'model';
            logOnce(path, `Feature Importance (${task}, ${model})`);
          } else if (path.includes('.metrics.learningCurves.')) {
            const parts = path.split('.');
            const task = parts[1] || 'task';
            const model = parts[4] || 'model';
            logOnce(path, `Learning Curve (${task}, ${model})`);
          } else if (path.endsWith('.residualSamples')) {
            const task = path.split('.')[1] || 'task';
            logOnce(path, `Residuals (${task})`);
          } else if (path === 'classification.summary') {
            logOnce('classification.summary', 'Classification Summary');
          } else if (path === 'classification.confusion_matrix') {
            logOnce('classification.confusion_matrix', 'Confusion Matrix');
          }
        }
        if (totalChunks > 0 && Number.isFinite(data.index)) {
          const progress = Math.min(1, Math.max(0, data.index / totalChunks));
          setSummaryProgress(progress);
        }
      } catch (error) {
        console.error('Failed to parse summary stream:', error);
      }
    };

    source.onerror = () => {
      if (!cancelled) {
        source.close();
      }
    };

    return () => {
      cancelled = true;
      source.close();
    };
  }, [status, runId, token]);

  return (
    <Card className="bg-background/95 backdrop-blur-sm border-border/50 overflow-hidden">
      <div className="flex items-center gap-2 px-4 py-3 border-b border-border/50 bg-muted/30">
        <Terminal className="h-4 w-4 text-accent" />
        <span className="text-sm font-medium text-foreground">Training Logs</span>
        <div className="ml-auto flex items-center gap-2">
          <div className={`h-2 w-2 rounded-full ${status === 'streaming' ? 'bg-accent animate-pulse' :
            status === 'complete' ? 'bg-primary' :
              status === 'error' ? 'bg-destructive' :
                'bg-muted-foreground'
            }`} />
          <span className="text-xs text-muted-foreground capitalize">{status}</span>
        </div>
      </div>
      {status === 'error' && (
        <div className="border-b border-destructive/30 bg-destructive/10 px-4 py-2 text-xs text-destructive">
          Log stream disconnected. Verify the SSE endpoint and auth token.
        </div>
      )}
      <div className="border-b border-border/40 px-4 py-2 text-[11px] text-muted-foreground">
        SSE: {url}?token=***{runId ? ` • runId: ${runId}` : ''}
      </div>

      <div ref={containerRef} className="h-[400px] overflow-y-auto">
        <div ref={scrollRef} className="p-4 text-xs font-mono whitespace-pre-wrap break-words space-y-1">
          {logs
            ? renderLogs()
            : <div className="text-muted-foreground">Waiting for logs...</div>
          }
        </div>
      </div>
    </Card>
  );
}
