import { Card } from '@/components/ui/card';

type ChartCardProps = {
  title: string;
  children: React.ReactNode;
  className?: string;
};

export function ChartCard({ title, children, className }: ChartCardProps) {
  return (
    <Card className={`p-4 bg-card/40 border-border/50 space-y-3${className ? ` ${className}` : ''}`}>
      <h5 className="text-sm font-semibold text-foreground">{title}</h5>
      {children}
    </Card>
  );
}
