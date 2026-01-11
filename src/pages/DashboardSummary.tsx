import { useEffect, useState } from 'react';
import { api } from '@/lib/api';
import { Layout } from '@/components/Layout';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from '@/components/ui/accordion';
import { ChartContainer, ChartLegend, ChartLegendContent, ChartTooltip, ChartTooltipContent } from '@/components/ui/chart';
import { BarChart3, Target, TrendingUp, Loader2, AlertCircle } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { toast } from 'sonner';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  ReferenceLine,
  Scatter,
  ScatterChart,
  XAxis,
  YAxis
} from 'recharts';

const CONFIG_SECTIONS = [
  {
    title: 'Data & Runtime',
    keys: ['RANDOM_SEED', 'TEST_SIZE', 'THREADS']
  },
  {
    title: 'Decision Tree',
    keys: ['DT_ENABLE', 'DT_MAX_DEPTH', 'DT_MIN_SAMPLES_LEAF']
  },
  {
    title: 'Random Forest',
    keys: ['RF_ENABLE', 'RF_TREES', 'RF_MAX_DEPTH', 'RF_MIN_SAMPLES_LEAF']
  },
  {
    title: 'LightGBM',
    keys: ['LGBM_ENABLE', 'LGBM_N_ESTIMATORS', 'LGBM_REG_ALPHA', 'LGBM_REG_LAMBDA']
  },
  {
    title: 'Neural Net (MLP)',
    keys: ['MLP_ENABLE', 'MLP_HIDDEN', 'MLP_EPOCHS', 'MLP_PATIENCE']
  },
  {
    title: 'Support Vector Regression (SVR)',
    keys: ['SVR_ENABLE', 'SVR_C', 'SVR_EPSILON']
  },
  {
    title: 'Risk Thresholds',
    keys: ['RISK_HIGH_MAX', 'RISK_MED_MAX']
  }
] as const;

const MODEL_COLORS: Record<string, string> = {
  DecisionTree: '#0ea5e9',
  RandomForest: '#6366f1',
  LightGBM: '#22c55e',
  SVR: '#f59e0b',
  MLP: '#f43f5e'
};

const MODEL_SHORT_NAMES: Record<string, string> = {
  DecisionTree: 'DT',
  RandomForest: 'RF',
  LightGBM: 'LGBM',
  SVR: 'SVR',
  MLP: 'MLP'
};

const getOrderedModels = (models: string[], bestModel?: string) => {
  const unique = Array.from(new Set(models));
  if (!bestModel) return unique;
  return [bestModel, ...unique.filter((name) => name !== bestModel)];
};

const formatDecimal = (value: unknown, decimals: number) => {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value.toFixed(decimals);
  }
  const parsed = typeof value === 'string' ? Number(value) : NaN;
  if (Number.isFinite(parsed)) {
    return parsed.toFixed(decimals);
  }
  return String(value ?? '');
};

const formatGpa = (value: unknown) => formatDecimal(value, 2);

const getScatterDomain = (points: { actual: number; predicted: number }[]) => {
  if (!points.length) {
    return { min: 0, max: 1 };
  }
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  points.forEach((point) => {
    min = Math.min(min, point.actual, point.predicted);
    max = Math.max(max, point.actual, point.predicted);
  });
  if (min === max) {
    min -= 1;
    max += 1;
  }
  return { min, max };
};

const ChartCard = ({ title, children }: { title: string; children: React.ReactNode }) => (
  <Card className="p-4 bg-card/40 border-border/50 space-y-3">
    <h5 className="text-sm font-semibold text-foreground">{title}</h5>
    {children}
  </Card>
);

const buildCurveData = (curve?: { train?: number[]; valid?: number[] }) => {
  if (!curve) return [];
  const train = curve.train || [];
  const valid = curve.valid || [];
  const length = Math.max(train.length, valid.length);
  return Array.from({ length }, (_, index) => ({
    epoch: index + 1,
    train: train[index],
    valid: valid[index]
  }));
};

const MetricBarChart = ({
  data,
  dataKey,
  label
}: {
  data: { name: string; color?: string; [key: string]: number | string | undefined }[];
  dataKey: string;
  label: string;
}) => (
  <ChartContainer
    config={{ [dataKey]: { label } }}
    className="h-56 w-full aspect-auto"
  >
    <BarChart data={data}>
      <CartesianGrid strokeDasharray="3 3" />
      <XAxis dataKey="name" />
      <YAxis tickFormatter={(value) => formatDecimal(value, 3)} />
      <ChartTooltip content={<ChartTooltipContent formatter={(value) => formatDecimal(value, 3)} />} />
      <Bar dataKey={dataKey} radius={[6, 6, 0, 0]}>
        {data.map((entry) => (
          <Cell key={entry.name} fill={entry.color || MODEL_COLORS[entry.name] || '#94a3b8'} />
        ))}
      </Bar>
    </BarChart>
  </ChartContainer>
);

const buildComparisonSeries = (
  models: Record<string, { [key: string]: number }> | undefined,
  names: string[],
  key: 'mae' | 'rmse' | 'r2'
) =>
  names
    .map((name) => ({
      name: MODEL_SHORT_NAMES[name] || name,
      color: MODEL_COLORS[name] || '#94a3b8',
      [key]: models?.[name]?.[key]
    }))
    .filter((entry) => entry[key] != null);

export default function DashboardSummary() {
  const [summary, setSummary] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [showBestExplanation, setShowBestExplanation] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    loadSummary();
  }, []);

  const loadSummary = async () => {
    try {
      const data = await api.getModelSummary();
      setSummary(data);
    } catch (error: any) {
      toast.error('Failed to load summary');
    } finally {
      setLoading(false);
      console.log(summary);
    }
  };

  if (loading) {
    return (
      <Layout>
        <div className="flex items-center justify-center py-20">
          <Loader2 className="h-8 w-8 animate-spin text-primary" />
        </div>
      </Layout>
    );
  }

  if (!summary?.hasModel) {
    return (
      <Layout>
        <Card className="p-12 text-center bg-card/50 backdrop-blur-sm">
          <AlertCircle className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
          <h2 className="text-2xl font-bold mb-2 text-foreground">No Model Trained</h2>
          <p className="text-muted-foreground mb-6">Train your first model to start making predictions</p>
          <Button onClick={() => navigate('/train-models')}>Train Model</Button>
        </Card>
      </Layout>
    );
  }

  const metrics = summary?.metrics || {};
  const summaryMetrics = metrics.summary || metrics;
  const bestModel = metrics.bestModel || summary.bestModel;
  const enabledModels = Array.isArray(metrics.enabledModels) ? metrics.enabledModels : [];
  const finalMetrics = metrics.final || null;
  const nextMetrics = metrics.next || null;
  const modelNames = getOrderedModels(
    enabledModels.length
      ? enabledModels
      : [
          ...Object.keys(finalMetrics?.models || {}),
          ...Object.keys(nextMetrics?.models || {})
        ],
    bestModel
  );
  const finalMaeData = buildComparisonSeries(finalMetrics?.models, modelNames, 'mae');
  const finalRmseData = buildComparisonSeries(finalMetrics?.models, modelNames, 'rmse');
  const finalR2Data = buildComparisonSeries(finalMetrics?.models, modelNames, 'r2');
  const nextMaeData = buildComparisonSeries(nextMetrics?.models, modelNames, 'mae');
  const nextRmseData = buildComparisonSeries(nextMetrics?.models, modelNames, 'rmse');
  const nextR2Data = buildComparisonSeries(nextMetrics?.models, modelNames, 'r2');
  const finalModelRanks = finalMetrics?.models
    ? Object.entries(finalMetrics.models)
        .map(([name, values]) => ({
          name,
          rmse: values.rmse,
          mae: values.mae,
          r2: values.r2
        }))
        .filter((entry) => typeof entry.rmse === 'number')
        .sort((a, b) => (a.rmse ?? 0) - (b.rmse ?? 0))
    : [];
  const bestEntry = finalModelRanks[0];
  const runnerUp = finalModelRanks[1];

  return (
    <Layout>
      <div className="space-y-6">
        <div>
          <h2 className="text-3xl font-bold text-foreground mb-2">Model Summary</h2>
          <p className="text-muted-foreground">Performance metrics and configuration</p>
          {bestModel && (
            <p className="text-sm text-muted-foreground mt-1">
              Summary metrics reflect the best model: <span className="font-semibold text-foreground">{bestModel}</span>
            </p>
          )}
        </div>

        {/* Metrics Grid */}
        {summaryMetrics && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <Card className="p-6 bg-card/50 backdrop-blur-sm border-border/50 hover:border-primary/50 transition-all">
              <div className="flex items-center gap-3 mb-3">
                <div className="p-2 rounded-lg bg-accent/10">
                  <BarChart3 className="h-5 w-5 text-accent" />
                </div>
                <h3 className="font-semibold text-foreground">Accuracy</h3>
              </div>
              <p className="text-4xl font-bold text-foreground mb-1">
                {summaryMetrics.accuracy != null
                  ? `${(summaryMetrics.accuracy * 100).toFixed(1)}%`
                  : '—'}
              </p>
              <p className="text-xs text-muted-foreground">Classification accuracy</p>
            </Card>

            <Card className="p-6 bg-card/50 backdrop-blur-sm border-border/50 hover:border-primary/50 transition-all">
              <div className="flex items-center gap-3 mb-3">
                <div className="p-2 rounded-lg bg-accent/10">
                  <Target className="h-5 w-5 text-accent" />
                </div>
                <h3 className="font-semibold text-foreground">RMSE</h3>
              </div>
              <p className="text-4xl font-bold text-foreground mb-1">
                {summaryMetrics.rmse != null ? summaryMetrics.rmse.toFixed(3) : '—'}
              </p>
              <p className="text-xs text-muted-foreground">Root mean squared error</p>
            </Card>

            <Card className="p-6 bg-card/50 backdrop-blur-sm border-border/50 hover:border-primary/50 transition-all">
              <div className="flex items-center gap-3 mb-3">
                <div className="p-2 rounded-lg bg-accent/10">
                  <TrendingUp className="h-5 w-5 text-accent" />
                </div>
                <h3 className="font-semibold text-foreground">R² Score</h3>
              </div>
              <p className="text-4xl font-bold text-foreground mb-1">
                {summaryMetrics.r2 != null ? summaryMetrics.r2.toFixed(3) : '—'}
              </p>
              <p className="text-xs text-muted-foreground">Coefficient of determination</p>
            </Card>
          </div>
        )}

        {/* Configuration */}
        {summary.config && (
          <Card className="p-6 bg-card/50 backdrop-blur-sm border-border/50">
            <h3 className="text-xl font-semibold text-foreground mb-4">Training Configuration</h3>
            <div className="space-y-6">
              {CONFIG_SECTIONS.map((section) => {
                const sectionKeys = section.keys.filter((key) => key in summary.config);
                if (sectionKeys.length === 0) return null;

                return (
                  <div
                    key={section.title}
                    className="rounded-lg border border-border/60 bg-background/40 p-4 space-y-3"
                  >
                    <h4 className="text-sm font-semibold text-foreground">{section.title}</h4>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      {sectionKeys.map((key) => {
                        const value = summary.config[key];
                        const displayValue = typeof value === 'boolean' ? (value ? 'Enabled' : 'Disabled') : String(value);
                        return (
                          <div key={key} className="p-3 rounded-lg bg-muted/30">
                            <p className="text-xs text-muted-foreground mb-1">{key.replace(/_/g, ' ')}</p>
                            <p className="font-semibold text-foreground">{displayValue}</p>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                );
              })}
              {(() => {
                const knownKeys = new Set(CONFIG_SECTIONS.flatMap((section) => section.keys));
                const remainingKeys = Object.keys(summary.config).filter(
                  (key) => key !== 'GRADE_POINTS' && !knownKeys.has(key)
                );
                if (remainingKeys.length === 0) return null;
                return (
                  <div className="rounded-lg border border-border/60 bg-background/40 p-4 space-y-3">
                    <h4 className="text-sm font-semibold text-foreground">Other</h4>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      {remainingKeys.map((key) => (
                        <div key={key} className="p-3 rounded-lg bg-muted/30">
                          <p className="text-xs text-muted-foreground mb-1">{key.replace(/_/g, ' ')}</p>
                          <p className="font-semibold text-foreground">{String(summary.config[key])}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                );
              })()}
            </div>
          </Card>
        )}

        {/* Model Insights */}
        {finalMetrics && modelNames.length > 0 && (
          <Card className="p-6 bg-card/50 backdrop-blur-sm border-border/50 space-y-4">
            <h3 className="text-xl font-semibold text-foreground">Model Insights</h3>
            <Accordion
              type="single"
              collapsible
              defaultValue={modelNames.includes(bestModel) ? bestModel : modelNames[0]}
              className="w-full"
            >
              {modelNames.map((modelName) => {
                const finalModelMetrics = finalMetrics?.models?.[modelName];
                const finalPredictions = finalMetrics?.predictions?.[modelName] || [];
                const finalImportance = (finalMetrics?.featureImportance?.[modelName] || []).slice(0, 8);
                const finalCurve = buildCurveData(finalMetrics?.learningCurves?.[modelName]);
                const finalDomain = getScatterDomain(finalPredictions);

                const nextModelMetrics = nextMetrics?.models?.[modelName];
                const nextPredictions = nextMetrics?.predictions?.[modelName] || [];
                const nextImportance = (nextMetrics?.featureImportance?.[modelName] || []).slice(0, 8);
                const nextCurve = buildCurveData(nextMetrics?.learningCurves?.[modelName]);
                const nextDomain = getScatterDomain(nextPredictions);

                const modelColor = MODEL_COLORS[modelName] || '#94a3b8';

                const finalCards = [];
                if (finalCurve.length > 0) {
                  finalCards.push(
                    <ChartCard title="Learning Curve" key="final-curve">
                      <ChartContainer
                        config={{
                          train: { label: 'Train', color: '#6366f1' },
                          valid: { label: 'Validation', color: '#f59e0b' }
                        }}
                        className="h-52 w-full aspect-auto"
                      >
                        <LineChart data={finalCurve}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="epoch" />
                          <YAxis tickFormatter={(value) => formatDecimal(value, 3)} />
                          <ChartTooltip content={<ChartTooltipContent formatter={(value) => formatDecimal(value, 3)} />} />
                          <ChartLegend content={<ChartLegendContent />} />
                          <Line type="monotone" dataKey="train" stroke="var(--color-train)" strokeWidth={2} dot={false} />
                          <Line type="monotone" dataKey="valid" stroke="var(--color-valid)" strokeWidth={2} dot={false} />
                        </LineChart>
                      </ChartContainer>
                    </ChartCard>
                  );
                }

                if (finalImportance.length > 0) {
                  finalCards.push(
                    <ChartCard title="Feature Importance" key="final-importance">
                      <ChartContainer
                        config={{ importance: { label: 'Importance', color: modelColor } }}
                        className="h-52 w-full aspect-auto"
                      >
                        <BarChart data={finalImportance} layout="vertical">
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis type="number" dataKey="importance" tickFormatter={(value) => formatDecimal(value, 3)} />
                          <YAxis type="category" dataKey="feature" width={90} />
                          <ChartTooltip content={<ChartTooltipContent formatter={(value) => formatDecimal(value, 3)} />} />
                          <Bar dataKey="importance" fill="var(--color-importance)" radius={[0, 6, 6, 0]} />
                        </BarChart>
                      </ChartContainer>
                    </ChartCard>
                  );
                }

                if (finalPredictions.length > 0) {
                  finalCards.push(
                    <ChartCard title="Predicted vs Actual" key="final-scatter">
                      <ChartContainer
                        config={{ predicted: { label: 'Predicted', color: modelColor } }}
                        className="h-52 w-full aspect-auto"
                      >
                        <ScatterChart>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis
                            type="number"
                            dataKey="actual"
                            domain={[finalDomain.min, finalDomain.max]}
                            tickFormatter={formatGpa}
                          />
                          <YAxis
                            type="number"
                            dataKey="predicted"
                            domain={[finalDomain.min, finalDomain.max]}
                            tickFormatter={formatGpa}
                          />
                          <ChartTooltip content={<ChartTooltipContent formatter={(value) => formatGpa(value)} />} />
                          <ReferenceLine
                            segment={[
                              { x: finalDomain.min, y: finalDomain.min },
                              { x: finalDomain.max, y: finalDomain.max }
                            ]}
                            stroke="#94a3b8"
                            strokeDasharray="4 4"
                          />
                          <Scatter data={finalPredictions} fill="var(--color-predicted)" />
                        </ScatterChart>
                      </ChartContainer>
                    </ChartCard>
                  );
                }

                const nextCards = [];
                if (nextCurve.length > 0) {
                  nextCards.push(
                    <ChartCard title="Learning Curve" key="next-curve">
                      <ChartContainer
                        config={{
                          train: { label: 'Train', color: '#6366f1' },
                          valid: { label: 'Validation', color: '#f59e0b' }
                        }}
                        className="h-52 w-full aspect-auto"
                      >
                        <LineChart data={nextCurve}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="epoch" />
                          <YAxis tickFormatter={(value) => formatDecimal(value, 3)} />
                          <ChartTooltip content={<ChartTooltipContent formatter={(value) => formatDecimal(value, 3)} />} />
                          <ChartLegend content={<ChartLegendContent />} />
                          <Line type="monotone" dataKey="train" stroke="var(--color-train)" strokeWidth={2} dot={false} />
                          <Line type="monotone" dataKey="valid" stroke="var(--color-valid)" strokeWidth={2} dot={false} />
                        </LineChart>
                      </ChartContainer>
                    </ChartCard>
                  );
                }

                if (nextImportance.length > 0) {
                  nextCards.push(
                    <ChartCard title="Feature Importance" key="next-importance">
                      <ChartContainer
                        config={{ importance: { label: 'Importance', color: modelColor } }}
                        className="h-52 w-full aspect-auto"
                      >
                        <BarChart data={nextImportance} layout="vertical">
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis type="number" dataKey="importance" tickFormatter={(value) => formatDecimal(value, 3)} />
                          <YAxis type="category" dataKey="feature" width={90} />
                          <ChartTooltip content={<ChartTooltipContent formatter={(value) => formatDecimal(value, 3)} />} />
                          <Bar dataKey="importance" fill="var(--color-importance)" radius={[0, 6, 6, 0]} />
                        </BarChart>
                      </ChartContainer>
                    </ChartCard>
                  );
                }

                if (nextPredictions.length > 0) {
                  nextCards.push(
                    <ChartCard title="Predicted vs Actual" key="next-scatter">
                      <ChartContainer
                        config={{ predicted: { label: 'Predicted', color: modelColor } }}
                        className="h-52 w-full aspect-auto"
                      >
                        <ScatterChart>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis
                            type="number"
                            dataKey="actual"
                            domain={[nextDomain.min, nextDomain.max]}
                            tickFormatter={formatGpa}
                          />
                          <YAxis
                            type="number"
                            dataKey="predicted"
                            domain={[nextDomain.min, nextDomain.max]}
                            tickFormatter={formatGpa}
                          />
                          <ChartTooltip content={<ChartTooltipContent formatter={(value) => formatGpa(value)} />} />
                          <ReferenceLine
                            segment={[
                              { x: nextDomain.min, y: nextDomain.min },
                              { x: nextDomain.max, y: nextDomain.max }
                            ]}
                            stroke="#94a3b8"
                            strokeDasharray="4 4"
                          />
                          <Scatter data={nextPredictions} fill="var(--color-predicted)" />
                        </ScatterChart>
                      </ChartContainer>
                    </ChartCard>
                  );
                }

                return (
                  <AccordionItem key={modelName} value={modelName} className="border-border/40">
                    <AccordionTrigger className="text-left">
                      <div className="flex w-full items-center gap-3">
                        <span className="font-semibold text-foreground">{modelName}</span>
                        {bestModel === modelName && (
                          <span className="ml-auto rounded-full bg-primary/10 px-3 py-1 text-xs font-semibold text-primary">
                            Best Model
                          </span>
                        )}
                      </div>
                    </AccordionTrigger>
                    <AccordionContent>
                      <div className="space-y-6">
                        <div className="space-y-3">
                          <div className="flex flex-wrap items-center justify-between gap-2">
                            <h4 className="text-sm font-semibold text-foreground">Final CGPA Insights</h4>
                            {finalModelMetrics && (
                              <div className="flex flex-wrap gap-3 text-xs text-muted-foreground">
                                <span>MAE {finalModelMetrics.mae.toFixed(3)}</span>
                                <span>RMSE {finalModelMetrics.rmse.toFixed(3)}</span>
                                <span>R² {finalModelMetrics.r2.toFixed(3)}</span>
                              </div>
                            )}
                          </div>
                          {finalCards.length > 0 && (
                            <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
                              {finalCards}
                            </div>
                          )}
                        </div>

                        <div className="space-y-3">
                          <div className="flex flex-wrap items-center justify-between gap-2">
                            <h4 className="text-sm font-semibold text-foreground">Next Semester Insights</h4>
                            {nextModelMetrics && (
                              <div className="flex flex-wrap gap-3 text-xs text-muted-foreground">
                                <span>MAE {nextModelMetrics.mae.toFixed(3)}</span>
                                <span>RMSE {nextModelMetrics.rmse.toFixed(3)}</span>
                                <span>R² {nextModelMetrics.r2.toFixed(3)}</span>
                              </div>
                            )}
                          </div>
                          {nextCards.length > 0 && (
                            <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
                              {nextCards}
                            </div>
                          )}
                        </div>
                      </div>
                    </AccordionContent>
                  </AccordionItem>
                );
              })}
            </Accordion>
          </Card>
        )}

        {/* Model Comparisons */}
        {(finalMetrics?.models || nextMetrics?.models) && modelNames.length > 0 && (
          <Card className="p-6 bg-card/50 backdrop-blur-sm border-border/50 space-y-6">
            <h3 className="text-xl font-semibold text-foreground">Model Comparisons</h3>
            <div className="space-y-6">
              {finalMetrics?.models && (finalMaeData.length || finalRmseData.length || finalR2Data.length) ? (
                <div className="space-y-4">
                  <h4 className="text-sm font-semibold text-foreground">Final CGPA Comparison</h4>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    {finalMaeData.length > 0 && (
                      <ChartCard title="MAE (lower is better)">
                        <MetricBarChart data={finalMaeData} dataKey="mae" label="MAE" />
                      </ChartCard>
                    )}
                    {finalRmseData.length > 0 && (
                      <ChartCard title="RMSE (lower is better)">
                        <MetricBarChart data={finalRmseData} dataKey="rmse" label="RMSE" />
                      </ChartCard>
                    )}
                    {finalR2Data.length > 0 && (
                      <ChartCard title="R² (higher is better)">
                        <MetricBarChart data={finalR2Data} dataKey="r2" label="R²" />
                      </ChartCard>
                    )}
                  </div>
                </div>
              ) : null}

              {nextMetrics?.models && (nextMaeData.length || nextRmseData.length || nextR2Data.length) ? (
                <div className="space-y-4">
                  <h4 className="text-sm font-semibold text-foreground">Next Semester Comparison</h4>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    {nextMaeData.length > 0 && (
                      <ChartCard title="MAE (lower is better)">
                        <MetricBarChart data={nextMaeData} dataKey="mae" label="MAE" />
                      </ChartCard>
                    )}
                    {nextRmseData.length > 0 && (
                      <ChartCard title="RMSE (lower is better)">
                        <MetricBarChart data={nextRmseData} dataKey="rmse" label="RMSE" />
                      </ChartCard>
                    )}
                    {nextR2Data.length > 0 && (
                      <ChartCard title="R² (higher is better)">
                        <MetricBarChart data={nextR2Data} dataKey="r2" label="R²" />
                      </ChartCard>
                    )}
                  </div>
                </div>
              ) : null}
            </div>
          </Card>
        )}

        {/* Metadata */}
        <Card className="p-6 bg-card/50 backdrop-blur-sm border-border/50">
          <h3 className="text-xl font-semibold text-foreground mb-4">Model Details</h3>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-muted-foreground">Trained at:</span>
              <span className="text-foreground font-medium">
                {new Date(summary.createdAt).toLocaleString()}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Artifacts:</span>
              <span className="text-foreground font-mono text-xs">{summary.artifactsDir}</span>
            </div>
            <div className="pt-2">
              <button
                type="button"
                onClick={() => setShowBestExplanation((prev) => !prev)}
                className="text-xs text-muted-foreground underline"
              >
                {showBestExplanation ? 'Hide' : 'Show'} why this is the best model
              </button>
              {showBestExplanation && (
                <p className="mt-2 text-xs text-muted-foreground">
                  Best model is selected by the lowest test RMSE on final CGPA.{' '}
                  {bestEntry
                    ? `${bestEntry.name} has RMSE ${formatDecimal(bestEntry.rmse, 3)}, MAE ${formatDecimal(bestEntry.mae, 3)}, R² ${formatDecimal(bestEntry.r2, 3)}${
                        runnerUp
                          ? `, beating ${runnerUp.name} with RMSE ${formatDecimal(runnerUp.rmse, 3)}.`
                          : '.'
                      }`
                    : 'This run does not have enough metrics to compare models.'}
                </p>
              )}
            </div>
          </div>
        </Card>
      </div>
    </Layout>
  );
}
