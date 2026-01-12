
import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '@/lib/api';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card } from '@/components/ui/card';
import { GradeScaleEditor } from '@/components/GradeScaleEditor';
import { LogStream } from '@/components/LogStream';
import { Loader2, Sparkles, BarChart3 } from 'lucide-react';
import { toast } from 'sonner';
import { CustomFileInput } from '@/components/ui/CustomFileInput';

const DEFAULT_CONFIG: any = {
  RANDOM_SEED: 42,
  TEST_SIZE: 0.2,
  THREADS: 5,
  DT_ENABLE: true,
  DT_MAX_DEPTH: 0,
  DT_MIN_SAMPLES_LEAF: 1,
  RF_ENABLE: true,
  RF_TREES: 300,
  RF_MAX_DEPTH: 0,
  RF_MIN_SAMPLES_LEAF: 1,
  LGBM_ENABLE: true,
  LGBM_N_ESTIMATORS: 2000,
  LGBM_REG_ALPHA: 0,
  LGBM_REG_LAMBDA: 0,
  MLP_ENABLE: true,
  MLP_HIDDEN: 64,
  MLP_EPOCHS: 300,
  MLP_PATIENCE: 40,
  SVR_ENABLE: true,
  SVR_C: 10,
  SVR_EPSILON: 0.1,
  RISK_HIGH_MAX: 3.30,
  RISK_MED_MAX: 3.50,
  GRADE_POINTS: {
    "A+": 4.0, "A": 3.75, "A-": 3.5, "B+": 3.25, "B": 3.0,
    "B-": 2.75, "C+": 2.5, "C": 2.25, "D": 2.0, "F": 0.0
  }
};

const HYPERPARAMETER_GROUPS = [
  {
    title: 'Data & Runtime',
    description: 'Split ratio, threading, and reproducibility controls.',
    fields: [
      { key: 'RANDOM_SEED', label: 'Random Seed', step: '1', type: 'number' },
      { key: 'TEST_SIZE', label: 'Test Size', step: '0.01', min: 0.1, max: 0.3, type: 'number' },
      { key: 'THREADS', label: 'Threads', step: '1', min: 1, max: 16, type: 'number' }
    ]
  },
  {
    title: 'Decision Tree',
    description: 'Single-tree baseline model.',
    enabledKey: 'DT_ENABLE',
    fields: [
      { key: 'DT_ENABLE', label: 'Enabled', type: 'select' },
      { key: 'DT_MAX_DEPTH', label: 'Max Depth (0 = unlimited)', step: '1', min: 0, max: 50, type: 'number' },
      { key: 'DT_MIN_SAMPLES_LEAF', label: 'Min Samples per Leaf', step: '1', min: 1, max: 50, type: 'number' }
    ]
  },
  {
    title: 'Random Forest',
    description: 'Ensemble of decision trees.',
    enabledKey: 'RF_ENABLE',
    fields: [
      { key: 'RF_ENABLE', label: 'Enabled', type: 'select' },
      { key: 'RF_TREES', label: 'Number of Trees', step: '1', min: 50, max: 1000, type: 'number' },
      { key: 'RF_MAX_DEPTH', label: 'Max Depth (0 = unlimited)', step: '1', min: 0, max: 50, type: 'number' },
      { key: 'RF_MIN_SAMPLES_LEAF', label: 'Min Samples per Leaf', step: '1', min: 1, max: 50, type: 'number' }
    ]
  },
  {
    title: 'LightGBM',
    description: 'Gradient boosting with regularization.',
    enabledKey: 'LGBM_ENABLE',
    fields: [
      { key: 'LGBM_ENABLE', label: 'Enabled', type: 'select' },
      { key: 'LGBM_N_ESTIMATORS', label: 'Estimators', step: '1', min: 200, max: 4000, type: 'number' },
      { key: 'LGBM_REG_ALPHA', label: 'Reg Alpha (L1)', step: '0.01', min: 0, max: 10, type: 'number' },
      { key: 'LGBM_REG_LAMBDA', label: 'Reg Lambda (L2)', step: '0.01', min: 0, max: 10, type: 'number' }
    ]
  },
  {
    title: 'Neural Net (MLP)',
    description: 'Neural network size and training behavior.',
    enabledKey: 'MLP_ENABLE',
    fields: [
      { key: 'MLP_ENABLE', label: 'Enabled', type: 'select' },
      { key: 'MLP_HIDDEN', label: 'Hidden Units', step: '1', min: 16, max: 256, type: 'number' },
      { key: 'MLP_EPOCHS', label: 'Epochs', step: '1', min: 50, max: 600, type: 'number' },
      { key: 'MLP_PATIENCE', label: 'Early Stop Patience', step: '1', min: 10, max: 100, type: 'number' }
    ]
  },
  {
    title: 'Support Vector Regression (SVR)',
    description: 'Kernel-based regression.',
    enabledKey: 'SVR_ENABLE',
    fields: [
      { key: 'SVR_ENABLE', label: 'Enabled', type: 'select' },
      { key: 'SVR_C', label: 'Regularization (C)', step: '0.1', min: 0.1, max: 100, type: 'number' },
      { key: 'SVR_EPSILON', label: 'Epsilon', step: '0.01', min: 0.001, max: 1, type: 'number' }
    ]
  },
  {
    title: 'Risk Thresholds',
    description: 'Defines GPA cutoffs for risk buckets.',
    fields: [
      { key: 'RISK_HIGH_MAX', label: 'High Risk Max GPA', step: '0.01', type: 'number' },
      { key: 'RISK_MED_MAX', label: 'Medium Risk Max GPA', step: '0.01', type: 'number' }
    ]
  }
] as const;

type DatasetProfile = {
  studentCount: number;
  genderCounts: Record<string, number>;
  departmentCounts: Record<string, number>;
  gradeCounts: Record<string, number>;
  sscStats: { min: number; max: number; avg: number } | null;
  hscStats: { min: number; max: number; avg: number } | null;
  birthYearStats: { min: number; max: number; avg: number } | null;
  semesterStats: { min: number; max: number; avg: number } | null;
  attendanceStats: { min: number; max: number; avg: number } | null;
};

const sumCounts = (counts: Record<string, number>) =>
  Object.values(counts).reduce((sum, value) => sum + value, 0);

const renderDistribution = (
  counts: Record<string, number>,
  total: number,
  limit = 6
) => {
  const entries = Object.entries(counts).sort((a, b) => b[1] - a[1]).slice(0, limit);
  if (entries.length === 0) {
    return <p className="text-xs text-muted-foreground">No data available.</p>;
  }

  return (
    <div className="space-y-3">
      {entries.map(([label, count]) => {
        const percentage = total ? (count / total) * 100 : 0;
        return (
          <div key={label} className="space-y-1">
            <div className="flex justify-between text-xs text-muted-foreground">
              <span className="truncate">{label}</span>
              <span>{count} ({percentage.toFixed(1)}%)</span>
            </div>
            <div className="h-2 rounded-full bg-muted/40">
              <div
                className="h-2 rounded-full bg-primary/80"
                style={{ width: `${percentage}%` }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
};

export default function TrainModels({ embedded = false }: { embedded?: boolean }) {
  const [config, setConfig] = useState(DEFAULT_CONFIG);
  const [file, setFile] = useState<File | null>(null);
  const [training, setTraining] = useState(false);
  const [runId, setRunId] = useState<string | null>(null);
  const [trainingStatus, setTrainingStatus] = useState<string | null>(null);
  const [datasetProfile, setDatasetProfile] = useState<DatasetProfile | null>(null);
  const [datasetError, setDatasetError] = useState<string | null>(null);
  const navigate = useNavigate();

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      setDatasetProfile(null);
      setDatasetError(null);

      try {
        const text = await selectedFile.text();
        const parsed = JSON.parse(text);
        let students: unknown = parsed;
        if (parsed && typeof parsed === 'object' && Array.isArray((parsed as { students?: unknown }).students)) {
          students = (parsed as { students?: unknown }).students;
        }
        if (!Array.isArray(students)) {
          setDatasetError('Dataset must be a JSON array of student records, or an object with a students array.');
          return;
        }

        const genderCounts: Record<string, number> = {};
        const departmentCounts: Record<string, number> = {};
        const gradeCounts: Record<string, number> = {};
        let sscSum = 0;
        let sscCount = 0;
        let sscMin = Number.POSITIVE_INFINITY;
        let sscMax = Number.NEGATIVE_INFINITY;
        let hscSum = 0;
        let hscCount = 0;
        let hscMin = Number.POSITIVE_INFINITY;
        let hscMax = Number.NEGATIVE_INFINITY;
        let birthSum = 0;
        let birthCount = 0;
        let birthMin = Number.POSITIVE_INFINITY;
        let birthMax = Number.NEGATIVE_INFINITY;
        let semesterSum = 0;
        let semesterCount = 0;
        let semesterMin = Number.POSITIVE_INFINITY;
        let semesterMax = Number.NEGATIVE_INFINITY;
        let attendanceSum = 0;
        let attendanceCount = 0;
        let attendanceMin = Number.POSITIVE_INFINITY;
        let attendanceMax = Number.NEGATIVE_INFINITY;

        for (const student of students) {
          const gender = typeof student?.gender === 'string' ? student.gender.trim() : null;
          if (gender) genderCounts[gender] = (genderCounts[gender] || 0) + 1;

          const department = typeof student?.department === 'string' ? student.department.trim() : null;
          if (department) departmentCounts[department] = (departmentCounts[department] || 0) + 1;

          if (typeof student?.ssc_gpa === 'number' && !Number.isNaN(student.ssc_gpa)) {
            sscSum += student.ssc_gpa;
            sscCount += 1;
            sscMin = Math.min(sscMin, student.ssc_gpa);
            sscMax = Math.max(sscMax, student.ssc_gpa);
          }

          if (typeof student?.hsc_gpa === 'number' && !Number.isNaN(student.hsc_gpa)) {
            hscSum += student.hsc_gpa;
            hscCount += 1;
            hscMin = Math.min(hscMin, student.hsc_gpa);
            hscMax = Math.max(hscMax, student.hsc_gpa);
          }

          if (typeof student?.birth_year === 'number' && !Number.isNaN(student.birth_year)) {
            birthSum += student.birth_year;
            birthCount += 1;
            birthMin = Math.min(birthMin, student.birth_year);
            birthMax = Math.max(birthMax, student.birth_year);
          }

          if (student?.semesters && typeof student.semesters === 'object') {
            const semesterKeys = Object.keys(student.semesters);
            if (semesterKeys.length > 0) {
              semesterSum += semesterKeys.length;
              semesterCount += 1;
              semesterMin = Math.min(semesterMin, semesterKeys.length);
              semesterMax = Math.max(semesterMax, semesterKeys.length);
            }

            for (const semester of Object.values(student.semesters)) {
              if (!semester || typeof semester !== 'object') continue;
              for (const [key, value] of Object.entries(semester)) {
                if (key === 'attendancePercentage') {
                  if (typeof value === 'number' && !Number.isNaN(value)) {
                    attendanceSum += value;
                    attendanceCount += 1;
                    attendanceMin = Math.min(attendanceMin, value);
                    attendanceMax = Math.max(attendanceMax, value);
                  }
                  continue;
                }
                if (typeof value === 'string') {
                  const grade = value.trim();
                  if (grade) gradeCounts[grade] = (gradeCounts[grade] || 0) + 1;
                }
              }
            }
          }
        }

        const profile: DatasetProfile = {
          studentCount: students.length,
          genderCounts,
          departmentCounts,
          gradeCounts,
          sscStats: sscCount
            ? { min: sscMin, max: sscMax, avg: sscSum / sscCount }
            : null,
          hscStats: hscCount
            ? { min: hscMin, max: hscMax, avg: hscSum / hscCount }
            : null,
          birthYearStats: birthCount
            ? { min: birthMin, max: birthMax, avg: birthSum / birthCount }
            : null,
          semesterStats: semesterCount
            ? { min: semesterMin, max: semesterMax, avg: semesterSum / semesterCount }
            : null,
          attendanceStats: attendanceCount
            ? { min: attendanceMin, max: attendanceMax, avg: attendanceSum / attendanceCount }
            : null
        };

        setDatasetProfile(profile);
      } catch (error: any) {
        setDatasetError(error?.message || 'Failed to parse dataset.');
      }
    }
  };

  const handleStartTraining = async () => {
    if (!file) {
      toast.error('Please select a training file');
      return;
    }

    setTraining(true);
    try {
      const result = await api.startTraining(file, config);
      setRunId(result.runId);
      toast.success('Training started!');
    } catch (error: any) {
      toast.error(error.message || 'Failed to start training');
      setTraining(false);
    }
  };

  const handleTrainingComplete = (status: string) => {
    setTrainingStatus(status);
    if (status === 'SUCCEEDED') {
      toast.success('Training completed successfully!');
    } else {
      toast.error('Training failed');
      setTraining(false);
    }
  };

  const handleCloseLogs = () => {
    setRunId(null);
    setTraining(false);
    setTrainingStatus(null);
    navigate('/dashboard/summary');
  };

  if (runId) {
    const token = localStorage.getItem('token') || '';
    const content = (
      <div className="space-y-6">
        <div className="text-center space-y-2">
          <h1 className="text-3xl font-bold text-foreground">Training in Progress</h1>
          <p className="text-muted-foreground">Watch your model learn in real-time</p>
        </div>

        <LogStream
          url={api.getTrainLogsUrl(runId)}
          token={token}
          runId={runId}
          onComplete={handleTrainingComplete}
        />
        {trainingStatus && (
          <div className="flex justify-center">
            <Button variant="outline" onClick={handleCloseLogs}>
              Close Logs
            </Button>
          </div>
        )}
      </div>
    );

    if (embedded) return content;

    return (
      <div className="min-h-screen bg-background p-6">
        <div className="container mx-auto max-w-4xl">
          {content}
        </div>
      </div>
    );
  }

  const content = (
    <Card className="p-8 bg-card/50 backdrop-blur-sm border-border/50 space-y-8">
      {/* Hyperparameters */}
      <div className="space-y-4">
        <div className="flex items-center gap-2">
          <Sparkles className="h-5 w-5 text-primary" />
          <h2 className="text-xl font-semibold text-foreground">Hyperparameters</h2>
        </div>

        <div className="space-y-6">
          {HYPERPARAMETER_GROUPS.map((group) => (
            <div
              key={group.title}
              className="rounded-lg border border-border/60 bg-background/40 p-4 space-y-4"
            >
              <div className="space-y-1">
                <h3 className="text-base font-semibold text-foreground">{group.title}</h3>
                <p className="text-xs text-muted-foreground">{group.description}</p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {group.fields.map((field) => {
                  const isDisabled =
                    group.enabledKey &&
                    field.key !== group.enabledKey &&
                    !config[group.enabledKey];
                  if (field.type === 'select') {
                    const selectKey = field.key;
                    return (
                      <div key={selectKey} className="space-y-2">
                        <Label className="text-xs text-muted-foreground">{field.label}</Label>
                        <select
                          value={config[selectKey] ? 'true' : 'false'}
                          onChange={(e) => setConfig({ ...config, [selectKey]: e.target.value === 'true' })}
                          className="w-full h-10 px-3 rounded-md border border-input bg-background/50 text-sm"
                        >
                          <option value="true">Yes</option>
                          <option value="false">No</option>
                        </select>
                      </div>
                    );
                  }

                  return (
                    <div key={field.key} className="space-y-2">
                      <Label className="text-xs text-muted-foreground">{field.label}</Label>
                      <Input
                        type="number"
                        value={config[field.key] as number}
                        onChange={(e) => setConfig({ ...config, [field.key]: parseFloat(e.target.value) || 0 })}
                        step={field.step}
                        min={field.min}
                        max={field.max}
                        className="bg-background/50"
                        disabled={isDisabled}
                      />
                    </div>
                  );
                })}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Grade Scale */}
      <GradeScaleEditor
        value={config.GRADE_POINTS}
        onChange={(gradePoints) => setConfig({ ...config, GRADE_POINTS: gradePoints })}
      />

      {/* File Upload */}
      <div className="space-y-4">
        <Label className="text-base font-semibold" htmlFor='file-upload'>Training Dataset (JSON)</Label>
        <CustomFileInput
          id="file-upload"
          file={file}
          onFileChange={handleFileChange}
          accept=".json,application/json"
        />
      </div>

      {/* Dataset Profile */}
      {(datasetProfile || datasetError) && (
        <div className="space-y-4">
          <div className="flex items-center gap-2">
            <BarChart3 className="h-5 w-5 text-primary" />
            <h2 className="text-xl font-semibold text-foreground">Dataset Profile</h2>
          </div>

          {datasetError ? (
            <div className="rounded-lg border border-destructive/40 bg-destructive/10 px-4 py-3 text-sm text-destructive">
              {datasetError}
            </div>
          ) : (
            datasetProfile && (
              <div className="space-y-6">
                <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                  <div className="rounded-lg border border-border/60 bg-background/40 p-3">
                    <p className="text-xs text-muted-foreground">Students</p>
                    <p className="text-xl font-semibold text-foreground">{datasetProfile.studentCount}</p>
                  </div>
                  <div className="rounded-lg border border-border/60 bg-background/40 p-3">
                    <p className="text-xs text-muted-foreground">Avg Semesters</p>
                    <p className="text-xl font-semibold text-foreground">
                      {datasetProfile.semesterStats ? datasetProfile.semesterStats.avg.toFixed(1) : '—'}
                    </p>
                  </div>
                  <div className="rounded-lg border border-border/60 bg-background/40 p-3">
                    <p className="text-xs text-muted-foreground">Avg Attendance</p>
                    <p className="text-xl font-semibold text-foreground">
                      {datasetProfile.attendanceStats ? datasetProfile.attendanceStats.avg.toFixed(1) : '—'}
                    </p>
                  </div>
                  <div className="rounded-lg border border-border/60 bg-background/40 p-3">
                    <p className="text-xs text-muted-foreground">Avg SSC GPA</p>
                    <p className="text-xl font-semibold text-foreground">
                      {datasetProfile.sscStats ? datasetProfile.sscStats.avg.toFixed(2) : '—'}
                    </p>
                  </div>
                  <div className="rounded-lg border border-border/60 bg-background/40 p-3">
                    <p className="text-xs text-muted-foreground">Avg HSC GPA</p>
                    <p className="text-xl font-semibold text-foreground">
                      {datasetProfile.hscStats ? datasetProfile.hscStats.avg.toFixed(2) : '—'}
                    </p>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <div className="rounded-lg border border-border/60 bg-background/40 p-4 space-y-3">
                    <p className="text-sm font-semibold text-foreground">Gender Distribution</p>
                    {renderDistribution(datasetProfile.genderCounts, datasetProfile.studentCount)}
                  </div>
                  <div className="rounded-lg border border-border/60 bg-background/40 p-4 space-y-3">
                    <p className="text-sm font-semibold text-foreground">Department Distribution</p>
                    {renderDistribution(datasetProfile.departmentCounts, datasetProfile.studentCount, 8)}
                  </div>
                  <div className="rounded-lg border border-border/60 bg-background/40 p-4 space-y-3">
                    <p className="text-sm font-semibold text-foreground">Grade Distribution</p>
                    {renderDistribution(datasetProfile.gradeCounts, sumCounts(datasetProfile.gradeCounts), 10)}
                  </div>
                </div>
              </div>
            )
          )}
        </div>
      )}

      {/* Submit */}
      <Button
        onClick={handleStartTraining}
        disabled={!file || training}
        size="lg"
        className="w-full gap-2"
      >
        {training && <Loader2 className="h-5 w-5 animate-spin" />}
        Start Training
      </Button>
    </Card>
  );

  if (embedded) return content;

  return (
    <div className="min-h-screen bg-background p-6">
      <div className="container mx-auto max-w-4xl space-y-8">
        <div className="text-center space-y-2 animate-fade-in">
          <h1 className="text-3xl font-bold text-foreground">Configure Training</h1>
          <p className="text-muted-foreground">Set parameters and upload your training dataset</p>
        </div>

        {content}
      </div>
    </div>
  );
}
