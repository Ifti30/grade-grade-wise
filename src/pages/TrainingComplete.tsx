import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '@/lib/api';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { CheckCircle, BarChart3, TrendingUp, Target } from 'lucide-react';
import { toast } from 'sonner';
import { ChartContainer, ChartTooltip, ChartTooltipContent, ChartLegendContent } from '@/components/ui/chart';
import { LineChart, Line, CartesianGrid, XAxis, YAxis, BarChart, Bar, Cell, ScatterChart, Scatter, ReferenceLine } from 'recharts';
import { Accordion, AccordionItem, AccordionTrigger, AccordionContent } from '@/components/ui/accordion';

const MODEL_COLORS = {
  DecisionTree: '#0ea5e9',
  RandomForest: '#6366f1',
  LightGBM: '#22c55e',
  SVR: '#f59e0b',
  MLP: '#f43f5e'
};

const MODEL_SHORT_NAMES = {
  DecisionTree: 'DT',
  RandomForest: 'RF',
  LightGBM: 'LGBM',
  SVR: 'SVR',
  MLP: 'MLP'
};

const formatDecimal = (value: unknown, decimals: number) => {
  const num = typeof value === 'number' ? value : Number(value);
  return Number.isFinite(num) ? num.toFixed(decimals) : String(value ?? '');
};

const formatGpa = (value: unknown) => formatDecimal(value, 2);

const buildCurveData = (curve?: { train?: number[]; valid?: number[] }) => {
  if (!curve) return [];
  const train = curve.train || [];
  const valid = curve.valid || [];
  const length = Math.max(train.length, valid.length);
  return Array.from({ length }, (_, i) => ({
    epoch: i + 1,
    train: train[i],
    valid: valid[i]
  }));
};

const getScatterDomain = (points: { actual: number; predicted: number }[]) => {
  if (!points.length) return { min: 0, max: 1 };
  let min = Infinity, max = -Infinity;
  points.forEach(p => {
    min = Math.min(min, p.actual, p.predicted);
    max = Math.max(max, p.actual, p.predicted);
  });
  if (min === max) { min -= 1; max += 1; }
  return { min, max };
};

const buildComparisonSeries = (models: Record<string, any>, names: string[], key: 'mae' | 'rmse' | 'r2') =>
  names.map(name => ({
    name: MODEL_SHORT_NAMES[name] || name,
    color: MODEL_COLORS[name] || '#94a3b8',
    [key]: models?.[name]?.[key]
  })).filter(entry => entry[key] != null);

const ChartCard = ({ title, children }: { title: string; children: React.ReactNode }) => (
  <Card className="p-4 bg-card/40 border-border/50 space-y-3">
    <h5 className="text-sm font-semibold text-foreground">{title}</h5>
    {children}
  </Card>
);

const MetricBarChart = ({ data, dataKey, label }: { data: any[]; dataKey: string; label: string }) => (
  <ChartContainer config={{ [dataKey]: { label } }} className="h-56 w-full">
    <BarChart data={data}>
      <CartesianGrid strokeDasharray="3 3" />
      <XAxis dataKey="name" />
      <YAxis tickFormatter={value => formatDecimal(value, 3)} />
      <ChartTooltip content={<ChartTooltipContent formatter={(v) => formatDecimal(v, 3)} />} />
      <Bar dataKey={dataKey} radius={[6, 6, 0, 0]}>
        {data.map((entry) => <Cell key={entry.name} fill={entry.color || '#94a3b8'} />)}
      </Bar>
    </BarChart>
  </ChartContainer>
);

export default function TrainingComplete() {
  const [summary, setSummary] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => { loadSummary(); }, []);

  const loadSummary = async () => {
    try { const data = await api.getModelSummary(); setSummary(data); }
    catch { toast.error('Failed to load training summary'); }
    finally { setLoading(false); }
  };

  if (loading) return <div className="min-h-screen flex items-center justify-center"><div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary" /></div>;
  if (!summary?.hasModel) return (
    <div className="min-h-screen flex items-center justify-center p-6">
      <Card className="p-8 max-w-md text-center bg-card/50 backdrop-blur-sm">
        <p className="text-muted-foreground mb-4">No training results found</p>
        <Button onClick={() => navigate('/train-models')}>Start Training</Button>
      </Card>
    </div>
  );

  const summaryMetrics = summary?.metrics?.summary || summary?.metrics || {};
  const metrics = summary.metrics;
  const bestModel = metrics?.bestModel;
  const enabledModels = Array.isArray(metrics?.enabledModels) ? metrics.enabledModels : Object.keys(metrics.final?.models || {});
  const finalMetrics = metrics.final;
  const nextMetrics = metrics.next;
  const modelNames = Array.from(new Set(enabledModels));

  const finalMaeData = buildComparisonSeries(finalMetrics?.models, modelNames, 'mae');
  const finalRmseData = buildComparisonSeries(finalMetrics?.models, modelNames, 'rmse');
  const finalR2Data = buildComparisonSeries(finalMetrics?.models, modelNames, 'r2');

  const nextMaeData = buildComparisonSeries(nextMetrics?.models, modelNames, 'mae');
  const nextRmseData = buildComparisonSeries(nextMetrics?.models, modelNames, 'rmse');
  const nextR2Data = buildComparisonSeries(nextMetrics?.models, modelNames, 'r2');

  return (
    <div className="min-h-screen bg-background p-6">
      <div className="container mx-auto max-w-4xl space-y-8">
        {/* Success Header */}
        <div className="text-center space-y-4 animate-fade-in">
          <div className="inline-flex items-center justify-center p-4 rounded-full bg-primary/10 animate-glow-pulse">
            <CheckCircle className="h-16 w-16 text-primary" />
          </div>
          <h1 className="text-4xl font-bold text-foreground">Training Complete!</h1>
          <p className="text-lg text-muted-foreground">Your model has been trained successfully and is ready to make predictions</p>
        </div>

        {/* Metrics Cards */}
        {summaryMetrics && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 animate-slide-up">
            <Card className="p-6 bg-card/50">
              <div className="flex items-center gap-3 mb-2"><BarChart3 className="h-5 w-5 text-accent" /><h3 className="font-semibold text-foreground">Accuracy</h3></div>
              <p className="text-3xl font-bold text-foreground">{summaryMetrics.accuracy != null ? `${(summaryMetrics.accuracy * 100).toFixed(1)}%` : '—'}</p>
            </Card>
            <Card className="p-6 bg-card/50"><div className="flex items-center gap-3 mb-2"><Target className="h-5 w-5 text-accent" /><h3 className="font-semibold text-foreground">RMSE</h3></div>
              <p className="text-3xl font-bold text-foreground">{summaryMetrics.rmse != null ? summaryMetrics.rmse.toFixed(3) : '—'}</p>
            </Card>
            <Card className="p-6 bg-card/50"><div className="flex items-center gap-3 mb-2"><TrendingUp className="h-5 w-5 text-accent" /><h3 className="font-semibold text-foreground">R² Score</h3></div>
              <p className="text-3xl font-bold text-foreground">{summaryMetrics.r2 != null ? summaryMetrics.r2.toFixed(3) : '—'}</p>
            </Card>
          </div>
        )}

        {/* Model Insights */}
        {finalMetrics && modelNames.length > 0 && (
          <Card className="p-6 bg-card/50 space-y-4">
            <h3 className="text-xl font-semibold text-foreground">Model Insights</h3>
            <Accordion type="single" collapsible defaultValue={bestModel || modelNames[0]}>
              {modelNames.map(modelName => {
                const finalModelMetrics = finalMetrics.models?.[modelName];
                const finalPredictions = finalMetrics.predictions?.[modelName] || [];
                const finalImportance = (finalMetrics.featureImportance?.[modelName] || []).slice(0, 8);
                const finalCurve = buildCurveData(finalMetrics.learningCurves?.[modelName]);
                const finalDomain = getScatterDomain(finalPredictions);
                const modelColor = MODEL_COLORS[modelName] || '#94a3b8';

                const cards: JSX.Element[] = [];

                if (finalCurve.length) cards.push(
                  <ChartCard key="curve" title="Learning Curve">
                    <ChartContainer config={{ train: { label: 'Train', color: '#6366f1' }, valid: { label: 'Valid', color: '#f59e0b' } }} className="h-52 w-full">
                      <LineChart data={finalCurve}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="epoch" /><YAxis tickFormatter={formatDecimal} /><ChartTooltip content={<ChartTooltipContent formatter={formatDecimal} />} /><Line type="monotone" dataKey="train" stroke="#6366f1" dot={false} /><Line type="monotone" dataKey="valid" stroke="#f59e0b" dot={false} /></LineChart>
                    </ChartContainer>
                  </ChartCard>
                );

                if (finalImportance.length) cards.push(
                  <ChartCard key="importance" title="Feature Importance">
                    <ChartContainer config={{ importance: { label: 'Importance', color: modelColor } }} className="h-52 w-full">
                      <BarChart data={finalImportance} layout="vertical">
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis type="number" dataKey="importance" tickFormatter={formatDecimal} />
                        <YAxis type="category" dataKey="feature" width={90} />
                        <ChartTooltip content={<ChartTooltipContent formatter={formatDecimal} />} />
                        <Bar dataKey="importance" fill={modelColor} />
                      </BarChart>
                    </ChartContainer>
                  </ChartCard>
                );

                if (finalPredictions.length) cards.push(
                  <ChartCard key="scatter" title="Predicted vs Actual">
                    <ChartContainer config={{ predicted: { label: 'Predicted', color: modelColor } }} className="h-52 w-full">
                      <ScatterChart>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis type="number" dataKey="actual" domain={[finalDomain.min, finalDomain.max]} tickFormatter={formatGpa} />
                        <YAxis type="number" dataKey="predicted" domain={[finalDomain.min, finalDomain.max]} tickFormatter={formatGpa} />
                        <ChartTooltip content={<ChartTooltipContent formatter={formatGpa} />} />
                        <ReferenceLine segment={[{ x: finalDomain.min, y: finalDomain.min }, { x: finalDomain.max, y: finalDomain.max }]} stroke="#94a3b8" strokeDasharray="4 4" />
                        <Scatter data={finalPredictions} fill={modelColor} />
                      </ScatterChart>
                    </ChartContainer>
                  </ChartCard>
                );

                return (
                  <AccordionItem key={modelName} value={modelName}>
                    <AccordionTrigger>{modelName}</AccordionTrigger>
                    <AccordionContent>
                      <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">{cards}</div>
                    </AccordionContent>
                  </AccordionItem>
                );
              })}
            </Accordion>
          </Card>
        )}

        {/* Model Comparisons */}
        {(finalMetrics?.models) && (
          <Card className="p-6 bg-card/50 space-y-4">
            <h3 className="text-xl font-semibold text-foreground">Model Comparisons</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {finalMaeData.length > 0 && <ChartCard title="MAE (lower better)"><MetricBarChart data={finalMaeData} dataKey="mae" label="MAE" /></ChartCard>}
              {finalRmseData.length > 0 && <ChartCard title="RMSE (lower better)"><MetricBarChart data={finalRmseData} dataKey="rmse" label="RMSE" /></ChartCard>}
              {finalR2Data.length > 0 && <ChartCard title="R² (higher better)"><MetricBarChart data={finalR2Data} dataKey="r2" label="R²" /></ChartCard>}
            </div>
          </Card>
        )}

        {/* Actions */}
        <div className="flex gap-4 justify-center">
          <Button size="lg" onClick={() => navigate('/dashboard/summary')}>Go to Dashboard</Button>
          <Button size="lg" variant="outline" onClick={() => navigate('/dashboard/predict')}>Make Prediction</Button>
        </div>
      </div>
    </div>
  );
}
