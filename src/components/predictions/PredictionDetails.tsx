import { ChartContainer, ChartLegend, ChartLegendContent, ChartTooltip, ChartTooltipContent } from '@/components/ui/chart';
import { Card } from '@/components/ui/card';
import { Bar, BarChart, CartesianGrid, Line, LineChart, XAxis, YAxis } from 'recharts';

type PredictionRecord = {
  id?: string;
  studentId?: string;
  createdAt?: string;
  results?: Record<string, any>;
  summary?: Record<string, any>;
  outFile?: string | null;
  outFileUrl?: string | null;
  inputPath?: string | null;
  plots?: Record<string, string>;
  bestModel?: string;
};

interface PredictionDetailsProps {
  prediction: PredictionRecord;
  profile?: {
    studentId: string;
    department?: string;
    sscGpa?: number;
    hscGpa?: number;
    gender?: string;
    cgpaTrend: { label: string; value: number }[];
  };
}

const MODEL_COLORS: Record<string, string> = {
  DecisionTree: '#0ea5e9',
  RandomForest: '#6366f1',
  LightGBM: '#22c55e',
  SVR: '#f59e0b',
  MLP: '#f43f5e',
  History: '#64748b'
};

const MODEL_SHORT_NAMES: Record<string, string> = {
  DecisionTree: 'DT',
  RandomForest: 'RF',
  LightGBM: 'LGBM',
  SVR: 'SVR',
  MLP: 'MLP',
  History: 'History'
};

const formatDecimal = (value: unknown, decimals: number) => {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value.toFixed(decimals);
  }
  const parsed = typeof value === 'string' ? Number(value) : NaN;
  if (Number.isFinite(parsed)) {
    return parsed.toFixed(decimals);
  }
  return '-';
};

const formatGpa = (value: unknown) => formatDecimal(value, 2);

const toNumber = (value: unknown) => {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === 'string' && value.trim()) {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }
  return null;
};

const averageValues = (values?: Record<string, any>) => {
  if (!values || typeof values !== 'object') return null;
  const numbers = Object.values(values)
    .map((value) => (typeof value === 'string' ? Number(value) : value))
    .filter((value) => typeof value === 'number' && Number.isFinite(value));
  if (!numbers.length) return null;
  return numbers.reduce((sum, value) => sum + value, 0) / numbers.length;
};

const SummaryCard = ({
  title,
  value,
  tone
}: {
  title: string;
  value: string;
  tone?: string;
}) => (
  <Card className={`p-4 border-border/50 ${tone || 'bg-muted/30'}`}>
    <p className="text-xs text-muted-foreground mb-1">{title}</p>
    <p className="text-lg font-semibold text-foreground">{value || '—'}</p>
  </Card>
);

const Bullet = ({ color }: { color: string }) => (
  <span
    className="inline-block w-3 h-3 rounded-full mr-3"
    style={{ backgroundColor: color }}
  ></span>
);

export function PredictionDetails({ prediction, profile }: PredictionDetailsProps) {
  const summary = prediction.summary || {};
  const predictions = prediction.results || {};

  const finalCgpa = predictions.final_cgpa || {};
  const nextSem = predictions.next_sem_cgpa || {};
  const ensemble = summary.ensemble || predictions.ensemble || {};
  const loadAdjusted = summary.loadAdjusted || summary.load_adjusted || null;
  const loadContext = loadAdjusted?.context || null;
  const riskLevel = summary.risk || 'Unknown';
  const current = summary.current || {};
  const bestModel = summary.bestModel || prediction.bestModel;
  const modelNames = Array.from(
    new Set([...Object.keys(finalCgpa), ...Object.keys(nextSem)])
  );

  const studentId = prediction.studentId || profile?.studentId;
  const creditHours = toNumber(summary.creditHours ?? summary.credit_hours);
  const courseLoad = toNumber(summary.courseLoad ?? summary.course_load);
  const baselineFinal = ensemble.final_cgpa_mean ?? averageValues(finalCgpa);
  const baselineNext = ensemble.next_sem_cgpa_mean ?? averageValues(nextSem);
  const adjustedFinal =
    loadAdjusted?.ensemble?.final_cgpa_mean ?? averageValues(loadAdjusted?.final_cgpa);
  const adjustedNext =
    loadAdjusted?.ensemble?.next_sem_cgpa_mean ?? averageValues(loadAdjusted?.next_sem_cgpa);
  const deltaFinal = loadAdjusted?.delta?.final_cgpa ?? null;
  const deltaNext = loadAdjusted?.delta?.next_sem_cgpa ?? null;
  const comparisonData = [
    {
      label: 'Next',
      baseline: baselineNext,
      adjusted: adjustedNext
    },
    {
      label: 'Final',
      baseline: baselineFinal,
      adjusted: adjustedFinal
    }
  ].filter((entry) => entry.baseline != null || entry.adjusted != null);
  const showCreditComparison = creditHours != null || adjustedFinal != null || adjustedNext != null;
  const semesterCount = profile?.cgpaTrend?.length || 0;
  const cgpaStart = semesterCount ? profile?.cgpaTrend?.[0] : null;
  const cgpaEnd = semesterCount ? profile?.cgpaTrend?.[semesterCount - 1] : null;

  const hasTrend = profile?.cgpaTrend && profile.cgpaTrend.length > 0;
  const trajectoryData = hasTrend
    ? (() => {
        const trend = profile?.cgpaTrend || [];
        const lastPoint = trend[trend.length - 1];
        const baseRows = trend.map((point, index) => {
          const row: Record<string, any> = {
            label: point.label,
            History: point.value
          };
          if (index === trend.length - 1) {
            modelNames.forEach((model) => {
              row[model] = point.value;
            });
          }
          return row;
        });

        const nextRow: Record<string, any> = { label: 'Next' };
        const finalRow: Record<string, any> = { label: 'Final' };
        modelNames.forEach((model) => {
          if (nextSem[model] != null) {
            nextRow[model] = nextSem[model];
          }
          if (finalCgpa[model] != null) {
            finalRow[model] = finalCgpa[model];
          }
        });

        if (lastPoint) {
          baseRows.push(nextRow, finalRow);
        }
        return baseRows;
      })()
    : [];

  const riskTone =
    riskLevel?.toLowerCase() === 'high'
      ? 'bg-destructive/10 border-destructive/30'
      : riskLevel?.toLowerCase() === 'medium'
      ? 'bg-yellow-500/10 border-yellow-500/30'
      : 'bg-green-500/10 border-green-500/30';

  const riskColorClass =
    riskLevel?.toLowerCase() === 'high'
      ? 'text-destructive'
      : riskLevel?.toLowerCase() === 'medium'
      ? 'text-yellow-500'
      : 'text-green-500';

  const bestModelName = bestModel ? String(bestModel).toLowerCase() : null;
  const bestModelInfo = bestModelName ? modelNames.find(m => m.toLowerCase() === bestModelName) : undefined;
  const otherModels = modelNames.filter(name => name.toLowerCase() !== bestModelName);

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {studentId && (
          <SummaryCard title="Student ID" value={String(studentId)} tone="bg-primary/10 border-primary/20" />
        )}
        <SummaryCard title="Risk Level" value={riskLevel} tone={riskTone} />
        {bestModel && (
          <SummaryCard
            title="Best Model"
            value={String(bestModel)}
            tone="bg-accent/10 border-accent/30"
          />
        )}
      </div>

      {profile && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <SummaryCard title="Department" value={profile.department || '—'} />
          <SummaryCard title="Gender" value={profile.gender || '—'} />
          <SummaryCard title="SSC GPA" value={formatGpa(profile.sscGpa)} />
          <SummaryCard title="HSC GPA" value={formatGpa(profile.hscGpa)} />
          <SummaryCard title="Semesters" value={semesterCount ? String(semesterCount) : '—'} />
          <SummaryCard title="Latest Sem GPA" value={formatGpa(current.last_sem_gpa)} />
          <SummaryCard title="Current CGPA" value={formatGpa(current.current_cgpa)} />
          <SummaryCard
            title="CGPA Trend"
            value={
              cgpaStart && cgpaEnd
                ? `${cgpaStart.label}: ${formatGpa(cgpaStart.value)} → ${cgpaEnd.label}: ${formatGpa(cgpaEnd.value)}`
                : '—'
            }
          />
        </div>
      )}

      {trajectoryData.length > 0 && (
        <Card className="p-6 bg-card/40 border-border/50 space-y-4">
          <h4 className="text-lg font-semibold text-foreground">Performance Trajectory</h4>
          <ChartContainer
            config={{
              History: { label: MODEL_SHORT_NAMES.History, color: MODEL_COLORS.History },
              ...Object.fromEntries(
                modelNames.map((model) => [
                  model,
                  { label: MODEL_SHORT_NAMES[model] || model, color: MODEL_COLORS[model] || '#94a3b8' }
                ])
              )
            }}
            className="h-72 w-full aspect-auto"
          >
            <LineChart data={trajectoryData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="label" />
              <YAxis domain={[0, 4]} tickFormatter={(value) => formatGpa(value)} />
              <ChartTooltip content={<ChartTooltipContent formatter={(value) => formatGpa(value)} />} />
              <ChartLegend content={<ChartLegendContent />} />
              <Line type="monotone" dataKey="History" stroke="var(--color-History)" strokeWidth={2} dot={false} />
              {modelNames.map((model) => (
                <Line
                  key={model}
                  type="monotone"
                  dataKey={model}
                  stroke={`var(--color-${model})`}
                  strokeWidth={2}
                  strokeDasharray="4 4"
                  dot={false}
                  connectNulls
                />
              ))}
            </LineChart>
          </ChartContainer>
        </Card>
      )}

      {modelNames.length > 0 && (
        <div className="space-y-6">
          {bestModelInfo && (
            <div className="flex justify-center">
              <Card
                key={bestModelInfo}
                className="p-6 bg-card/60 border-l-4 space-y-4 w-full max-w-md shadow-lg"
                style={{
                  borderLeftColor: MODEL_COLORS[bestModelInfo] || '#94a3b8',
                }}
              >
                <div className="flex items-start justify-between">
                  <div className="flex items-center">
                    <Bullet color={MODEL_COLORS[bestModelInfo] || '#94a3b8'} />
                    <p className="text-lg font-bold">{bestModelInfo}</p>
                  </div>
                  <span className="text-sm font-semibold text-primary px-3 py-1 rounded-full bg-primary/10">
                    Best Model
                  </span>
                </div>
                <div className={`text-xl font-bold capitalize ${riskColorClass}`}>{riskLevel}</div>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm text-muted-foreground">Next Semester</p>
                    <p className="text-2xl font-bold text-foreground">
                      {formatGpa(nextSem[bestModelInfo])}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Final CGPA</p>
                    <p className="text-2xl font-bold text-foreground">
                      {formatGpa(finalCgpa[bestModelInfo])}
                    </p>
                  </div>
                </div>
              </Card>
            </div>
          )}

          {otherModels.length > 0 && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 justify-center">
              {otherModels.map((model, index) => (
                <Card
                  key={model}
                  className={`p-4 bg-card/50 border-l-4 space-y-3 ${otherModels.length % 2 !== 0 && index === otherModels.length - 1 ? 'md:col-span-2 md:max-w-sm md:mx-auto' : ''}`}
                  style={{
                    borderLeftColor: MODEL_COLORS[model] || '#94a3b8',
                  }}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center">
                      <Bullet color={MODEL_COLORS[model] || '#94a3b8'} />
                      <p className="text-sm font-semibold">{model}</p>
                    </div>
                  </div>
                  <div className={`text-base font-semibold capitalize ${riskColorClass}`}>{riskLevel}</div>
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <p className="text-xs text-muted-foreground">Next Semester</p>
                      <p className="text-lg font-semibold text-foreground">{formatGpa(nextSem[model])}</p>
                    </div>
                    <div>
                      <p className="text-xs text-muted-foreground">Final CGPA</p>
                      <p className="text-lg font-semibold text-foreground">{formatGpa(finalCgpa[model])}</p>
                    </div>
                  </div>
                </Card>
              ))}
            </div>
          )}
        </div>
      )}

      {showCreditComparison && (
        <Card className="p-6 bg-card/40 border-border/50 space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div>
              <h4 className="text-lg font-semibold text-foreground">Credit Hour Impact</h4>
              <p className="text-xs text-muted-foreground">
                Baseline vs. load-adjusted predictions based on upcoming credit hours.
              </p>
            </div>
            <div className="text-xs text-muted-foreground text-right">
              {creditHours != null && <div>Credit Hours: {formatDecimal(creditHours, 0)}</div>}
              {courseLoad != null && <div>Course Load: {formatDecimal(courseLoad, 0)} courses</div>}
            </div>
          </div>
          {comparisonData.length > 0 && (adjustedFinal != null || adjustedNext != null) ? (
            <ChartContainer
              config={{
                baseline: { label: 'Baseline', color: '#94a3b8' },
                adjusted: { label: 'With Credit Hours', color: '#38bdf8' }
              }}
              className="h-64 w-full aspect-auto"
            >
              <BarChart data={comparisonData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="label" />
                <YAxis domain={[0, 4]} tickFormatter={(value) => formatGpa(value)} />
                <ChartTooltip content={<ChartTooltipContent formatter={(value) => formatGpa(value)} />} />
                <ChartLegend content={<ChartLegendContent />} />
                <Bar dataKey="baseline" fill="var(--color-baseline)" radius={[6, 6, 0, 0]} />
                <Bar dataKey="adjusted" fill="var(--color-adjusted)" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ChartContainer>
          ) : (
            <p className="text-xs text-muted-foreground">
              Load-adjusted comparison is unavailable for this prediction.
            </p>
          )}
          {loadContext && (
            <div className="text-xs text-muted-foreground space-y-1">
              <div>
                Base load: {loadContext.base_bucket ?? '—'}{' '}
                {loadContext.base_load != null ? `(${formatDecimal(loadContext.base_load, 2)} avg)` : ''}
                , Requested load: {loadContext.requested_bucket ?? '—'}
              </div>
              <div>
                Expected Next GPA: {formatGpa(loadContext.base_sem_expect)} →{' '}
                {formatGpa(loadContext.requested_sem_expect)}{' '}
                {deltaNext != null ? `(Δ ${formatDecimal(deltaNext, 3)})` : ''}
              </div>
              <div>
                Expected Final GPA: {formatGpa(loadContext.base_final_expect)} →{' '}
                {formatGpa(loadContext.requested_final_expect)}{' '}
                {deltaFinal != null ? `(Δ ${formatDecimal(deltaFinal, 3)})` : ''}
              </div>
            </div>
          )}
        </Card>
      )}

        {(ensemble.final_cgpa_mean || ensemble.next_sem_cgpa_mean) && (
            <div className="space-y-4">
                <p className="text-lg font-semibold">Ensemble Averages</p>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                     <div
                        className="rounded-xl border bg-card text-card-foreground shadow p-6 flex flex-col items-start gap-2">
                        <p className="text-sm font-medium text-muted-foreground">Next Sem GPA (Mean)</p>
                        <div className="flex items-baseline gap-2">
                            <p className="text-4xl font-bold tracking-tight">{formatGpa(ensemble.next_sem_cgpa_mean)}</p>
                        </div>
                    </div>
                    <div
                        className="rounded-xl border bg-card text-card-foreground shadow p-6 flex flex-col items-start gap-2">
                        <p className="text-sm font-medium text-muted-foreground">Final CGPA (Mean)</p>
                        <div className="flex items-baseline gap-2">
                            <p className="text-4xl font-bold tracking-tight">{formatGpa(ensemble.final_cgpa_mean)}</p>
                        </div>
                    </div>
                </div>
            </div>
        )}


      <div className="pt-4 border-t border-border/50 text-xs text-muted-foreground space-y-2">
        <p>Predicted at: {prediction.createdAt ? new Date(prediction.createdAt).toLocaleString() : 'Unknown'}</p>
        {prediction.id && (
          <p>
            Prediction ID: <span className="font-mono break-all text-foreground">{prediction.id}</span>
          </p>
        )}
      </div>
    </div>
  );
}

export default PredictionDetails;
