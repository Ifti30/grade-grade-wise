
import { useState } from 'react';
import { api } from '@/lib/api';
import { Layout } from '@/components/Layout';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Loader2, Zap } from 'lucide-react';
import { toast } from 'sonner';
import PredictionDetails from '@/components/predictions/PredictionDetails';
import { ChartContainer, ChartTooltip, ChartTooltipContent } from '@/components/ui/chart';
import { CartesianGrid, Line, LineChart, XAxis, YAxis } from 'recharts';
import { CustomFileInput } from '@/components/ui/CustomFileInput';

const DEFAULT_GRADE_POINTS: Record<string, number> = {
  'A+': 4.0,
  A: 3.75,
  'A-': 3.5,
  'B+': 3.25,
  B: 3.0,
  'B-': 2.75,
  'C+': 2.5,
  C: 2.25,
  D: 2.0,
  F: 0.0
};

type StudentProfile = {
  studentId: string;
  department?: string;
  sscGpa?: number;
  hscGpa?: number;
  gender?: string;
  cgpaTrend: { label: string; value: number }[];
};

const formatGpa = (value: unknown) =>
  typeof value === 'number' && Number.isFinite(value) ? value.toFixed(2) : '—';

const computeSemesterGpa = (semester: Record<string, any>, gradePoints: Record<string, number>) => {
  const scores: number[] = [];
  Object.entries(semester || {}).forEach(([key, value]) => {
    if (key === 'attendancePercentage') return;
    if (typeof value === 'string' && value in gradePoints) {
      scores.push(gradePoints[value]);
    }
  });
  if (!scores.length) return null;
  return scores.reduce((sum, val) => sum + val, 0) / scores.length;
};

const computeCgpaTrend = (student: any) => {
  const semesters = student?.semesters && typeof student.semesters === 'object' ? student.semesters : null;
  if (!semesters) return [];
  const semesterKeys = Object.keys(semesters)
    .map((key) => Number(key))
    .filter((key) => !Number.isNaN(key))
    .sort((a, b) => a - b);

  const trend: { label: string; value: number }[] = [];
  let total = 0;
  let count = 0;

  semesterKeys.forEach((semKey) => {
    const semester = semesters[String(semKey)];
    const semGpa = computeSemesterGpa(semester, DEFAULT_GRADE_POINTS);
    if (semGpa == null) return;
    total += semGpa;
    count += 1;
    trend.push({
      label: `Sem ${semKey}`,
      value: total / count
    });
  });

  return trend;
};

export default function DashboardPredict() {
  const [file, setFile] = useState<File | null>(null);
  const [predicting, setPredicting] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [profile, setProfile] = useState<StudentProfile | null>(null);
  const [profileError, setProfileError] = useState<string | null>(null);
  const [creditHours, setCreditHours] = useState<string>('');

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      setResult(null);
      setProfile(null);
      setProfileError(null);

      try {
        const content = await selectedFile.text();
        const parsed = JSON.parse(content);
        const student = Array.isArray(parsed) ? parsed[0] : parsed;
        if (!student || typeof student !== 'object') {
          setProfileError('Student profile JSON is not valid.');
          return;
        }

        const studentIdRaw = student.student_id ?? student.id ?? 'Unknown';
        const trend = computeCgpaTrend(student);

        setProfile({
          studentId: String(studentIdRaw),
          department: typeof student.department === 'string' ? student.department : undefined,
          sscGpa: typeof student.ssc_gpa === 'number' ? student.ssc_gpa : undefined,
          hscGpa: typeof student.hsc_gpa === 'number' ? student.hsc_gpa : undefined,
          gender: typeof student.gender === 'string' ? student.gender : undefined,
          cgpaTrend: trend
        });
      } catch (error: any) {
        setProfileError(error?.message || 'Failed to parse student profile.');
      }
    }
  };

  const handlePredict = async () => {
    if (!file) {
      toast.error('Please select a student file');
      return;
    }

    setPredicting(true);
    try {
      const parsedHours = creditHours.trim() ? Number(creditHours) : null;
      const creditHoursValue = parsedHours !== null && !Number.isNaN(parsedHours) ? parsedHours : null;
      const data = await api.predict(file, creditHoursValue);
      console.log('Prediction debug:', {
        creditHours: creditHoursValue,
        summary: data?.summary,
        loadAdjusted: data?.summary?.loadAdjusted ?? data?.summary?.load_adjusted
      });
      setResult(data);
      toast.success('Prediction completed!');
    } catch (error: any) {
      toast.error(error.message || 'Prediction failed');
    } finally {
      setPredicting(false);
    }
  };

  const handleReset = () => {
    setFile(null);
    setResult(null);
    setProfile(null);
    setProfileError(null);
    setCreditHours('');
  };

  return (
    <Layout>
      <div className="space-y-6">
        <div>
          <h2 className="text-3xl font-bold text-foreground mb-2">Make Prediction</h2>
          <p className="text-muted-foreground">Upload student data to predict academic performance</p>
        </div>

        {!result && (
          <Card className="p-8 bg-card/50 backdrop-blur-sm border-border/50">
            <div className="space-y-6">
              <div className="space-y-4">
                <Label className="text-base font-semibold">Student Data (JSON)</Label>
                <CustomFileInput
                    id="file-upload-predict"
                    file={file}
                    onFileChange={handleFileChange}
                    accept=".json,application/json"
                />
                <p className="text-xs text-muted-foreground">
                  Upload a JSON file containing student information for prediction
                </p>
              </div>

              <div className="space-y-2">
                <Label className="text-sm font-semibold">Upcoming Credit Hours</Label>
                <Input
                  type="number"
                  placeholder="e.g., 15"
                  value={creditHours}
                  onChange={(e) => setCreditHours(e.target.value)}
                  className="bg-background/50"
                />
                <p className="text-xs text-muted-foreground">
                  Optional: provide credit hours for the upcoming semester.
                </p>
              </div>

              <Button
                onClick={handlePredict}
                disabled={!file || predicting}
                size="lg"
                className="w-full gap-2"
              >
                {predicting ? (
                  <Loader2 className="h-5 w-5 animate-spin" />
                ) : (
                  <Zap className="h-5 w-5" />
                )}
                {predicting ? 'Predicting...' : 'Run Prediction'}
              </Button>
            </div>
          </Card>
        )}

        {!result && (profile || profileError) && (
          <Card className="p-6 bg-card/50 backdrop-blur-sm border-border/50">
            <div className="space-y-4">
              <h3 className="text-xl font-semibold text-foreground">Student Profile</h3>
              {profileError ? (
                <div className="rounded-lg border border-destructive/40 bg-destructive/10 px-4 py-3 text-sm text-destructive">
                  {profileError}
                </div>
              ) : (
                profile && (
                  <div className="grid grid-cols-1 lg:grid-cols-[1fr_1.4fr] gap-6">
                    <div className="grid grid-cols-2 gap-4">
                      <div className="p-4 rounded-lg bg-muted/30">
                        <p className="text-xs text-muted-foreground mb-1">Student ID</p>
                        <p className="text-lg font-semibold text-foreground">{profile.studentId}</p>
                      </div>
                      <div className="p-4 rounded-lg bg-muted/30">
                        <p className="text-xs text-muted-foreground mb-1">Department</p>
                        <p className="text-lg font-semibold text-foreground">{profile.department || '—'}</p>
                      </div>
                      <div className="p-4 rounded-lg bg-muted/30">
                        <p className="text-xs text-muted-foreground mb-1">SSC GPA</p>
                        <p className="text-lg font-semibold text-foreground">{formatGpa(profile.sscGpa)}</p>
                      </div>
                      <div className="p-4 rounded-lg bg-muted/30">
                        <p className="text-xs text-muted-foreground mb-1">HSC GPA</p>
                        <p className="text-lg font-semibold text-foreground">{formatGpa(profile.hscGpa)}</p>
                      </div>
                      <div className="p-4 rounded-lg bg-muted/30">
                        <p className="text-xs text-muted-foreground mb-1">Gender</p>
                        <p className="text-lg font-semibold text-foreground">{profile.gender || '—'}</p>
                      </div>
                      <div className="p-4 rounded-lg bg-muted/30">
                        <p className="text-xs text-muted-foreground mb-1">Semesters</p>
                        <p className="text-lg font-semibold text-foreground">
                          {profile.cgpaTrend.length || '—'}
                        </p>
                      </div>
                    </div>

                    <div className="p-4 rounded-lg bg-muted/20 border border-border/40 space-y-3">
                      <div className="flex items-center justify-between">
                        <p className="text-sm font-semibold text-foreground">CGPA Trend</p>
                        {profile.cgpaTrend.length > 0 && (
                          <p className="text-xs text-muted-foreground">
                            {profile.cgpaTrend[0].label} {formatGpa(profile.cgpaTrend[0].value)} →{' '}
                            {profile.cgpaTrend[profile.cgpaTrend.length - 1].label}{' '}
                            {formatGpa(profile.cgpaTrend[profile.cgpaTrend.length - 1].value)}
                          </p>
                        )}
                      </div>
                      {profile.cgpaTrend.length > 0 ? (
                        <ChartContainer
                          config={{ cgpa: { label: 'CGPA', color: '#6366f1' } }}
                          className="h-48 w-full aspect-auto"
                        >
                          <LineChart data={profile.cgpaTrend}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="label" />
                            <YAxis domain={[0, 4]} tickFormatter={(value) => formatGpa(value)} />
                            <ChartTooltip content={<ChartTooltipContent formatter={(value) => formatGpa(value)} />} />
                            <Line type="monotone" dataKey="value" stroke="var(--color-cgpa)" strokeWidth={2} dot={false} />
                          </LineChart>
                        </ChartContainer>
                      ) : (
                        <p className="text-xs text-muted-foreground">No CGPA trend available.</p>
                      )}
                    </div>
                  </div>
                )
              )}
            </div>
          </Card>
        )}

        {/* Results */}
        {result && (
          <Card className="p-8 bg-card/50 backdrop-blur-sm border-border/50 animate-fade-in">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-2xl font-bold text-foreground">Prediction Results</h3>
              <Button onClick={handleReset} variant="outline" className="gap-2">
                <Zap className="h-4 w-4" />
                Make Another Prediction
              </Button>
            </div>
            <PredictionDetails prediction={result} profile={profile || undefined} />
          </Card>
        )}
      </div>
    </Layout>
  );
}
