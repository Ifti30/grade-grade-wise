import { useEffect, useRef, useState } from 'react';
import { Card } from '@/components/ui/card';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Terminal } from 'lucide-react';

interface LogStreamProps {
  url: string;
  token: string;
  onComplete?: (status: string) => void;
  runId?: string;
}

export function LogStream({ url, token, onComplete, runId }: LogStreamProps) {
  const [logs, setLogs] = useState<string>('');
  const [status, setStatus] = useState<'connecting' | 'streaming' | 'complete' | 'error'>('connecting');
  const scrollRef = useRef<HTMLPreElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

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

    const compactEpochEntry = (entry: string) => (
      entry
        .replace('epoch=', 'e')
        .replace('valLoss=', 'v')
        .replace(/\s+/g, ' ')
        .trim()
    );

    const flushEpochs = () => {
      if (!epochBuffer.length) return;
      const key = `epoch-block-${nodes.length}`;
      const perRow = 6;
      for (let i = 0; i < epochBuffer.length; i += perRow) {
        const row = epochBuffer.slice(i, i + perRow);
        nodes.push(
          <div
            key={`${key}-${i}`}
            className="pl-4 text-[11px] text-sky-600/90 flex flex-wrap gap-x-4 gap-y-1"
          >
            {row.map((entry, idx) => (
              <span key={`${key}-${i}-${idx}`} className="whitespace-nowrap">
                {compactEpochEntry(entry)}
              </span>
            ))}
          </div>
        );
      }
      epochBuffer = [];
    };

    lines.forEach((line, index) => {
      const trimmedEnd = line.trimEnd();
      const trimmedStart = trimmedEnd.trimStart();
      if (!trimmedStart) {
        flushEpochs();
        nodes.push(<div key={`spacer-${index}`} className="h-2" />);
        return;
      }

      if (trimmedStart.startsWith('epoch=') || trimmedStart.startsWith('valLoss=')) {
        epochBuffer.push(trimmedStart);
        return;
      }

      flushEpochs();
      nodes.push(renderLine(trimmedStart, index));
    });

    flushEpochs();
    return nodes;
  };

  useEffect(() => {
    const eventSource = new EventSource(`${url}?token=${encodeURIComponent(token)}`);

    eventSource.onopen = () => {
      setStatus('streaming');
    };

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        if (data.content) {
          setLogs((prev) => prev + data.content);
        }

        if (data.complete) {
          setStatus('complete');
          onComplete?.(data.status);
          eventSource.close();
        }
      } catch (error) {
        console.error('Failed to parse log event:', error);
      }
    };

    eventSource.onerror = () => {
      setStatus('error');
      eventSource.close();
    };

    return () => {
      eventSource.close();
    };
  }, [url, token, onComplete]);

  useEffect(() => {
    // Auto-scroll to bottom
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [logs]);

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
