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

  const renderLine = (line: string, index: number) => {
    const trimmed = line.trimEnd();
    const renderLine = (trimmed: string, index: number) => {
      if (!trimmed) {
        return <div key={`spacer-${index}`} className="h-2" />;
      }

      const progressMatch = trimmed.match(/^\[PROGRESS\]\s+phase=([^\s]+)(.*)$/);
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

      if (trimmed.startsWith('[ERROR]')) {
        return (
          <div key={`err-${index}`} className="text-destructive font-semibold">
            {trimmed}
          </div>
        );
      }

      if (trimmed.startsWith('[WARN]')) {
        return (
          <div key={`warn-${index}`} className="text-amber-600">
            {trimmed}
          </div>
        );
      }

      if (trimmed.startsWith('[INFO]')) {
        return (
          <div key={`info-${index}`} className="text-muted-foreground">
            {trimmed}
          </div>
        );
      }

      if (trimmed.startsWith('[LightGBM]')) {
        return (
          <div key={`lgbm-${index}`} className="text-emerald-600/80">
            {trimmed}
          </div>
        );
      }

      return (
        <div key={`line-${index}`} className="text-foreground/90">
          {trimmed}
        </div>
      );
    };

    const renderLogs = () => {
      const lines = logs.split('\n');
      const nodes: React.ReactNode[] = [];
      let epochBuffer: string[] = [];

      const flushEpochs = () => {
        if (!epochBuffer.length) return;
        const key = `epoch-block-${nodes.length}`;
        nodes.push(
          <div key={key} className="pl-4 flex flex-wrap gap-x-3 gap-y-1 text-[11px] text-sky-600/90">
            {epochBuffer.map((entry, idx) => (
              <span key={`${key}-${idx}`} className="whitespace-nowrap">
                {entry}
              </span>
            ))}
          </div>
        );
        epochBuffer = [];
      };

      lines.forEach((line, index) => {
        const trimmed = line.trimEnd();
        if (!trimmed) {
          flushEpochs();
          nodes.push(<div key={`spacer-${index}`} className="h-2" />);
          return;
        }

        if (trimmed.startsWith('epoch=') || trimmed.startsWith('valLoss=')) {
          epochBuffer.push(trimmed);
          return;
        }

        flushEpochs();
        nodes.push(renderLine(trimmed, index));
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
}