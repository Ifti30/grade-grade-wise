import { spawn, spawnSync } from 'child_process';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import { makeResultCatcher } from './makeResultCatcher';
import { TrainingResultSchema } from './schemas';
import { makeResultCatcher } from './makeResultCatcher'

const __dirname = path.dirname(fileURLToPath(import.meta.url));
function resolvePythonBinary() {
  if (process.env.PYTHON_BIN) {
    return process.env.PYTHON_BIN;
  }

  const candidates = process.platform === 'win32'
    ? ['python', 'py', 'python3']
    : ['python3', 'python'];

  for (const candidate of candidates) {
    const result = spawnSync(candidate, ['--version'], { stdio: 'ignore' });
    if (!result.error && result.status === 0) {
      return candidate;
    }
  }

  // Fall back to the first candidate; spawn will surface a clearer error later
  return candidates[0];
}

const PYTHON_BIN = resolvePythonBinary();

// Resolve script paths ABSOLUTELY so CWD doesn't matter
const TRAIN_SCRIPT = path.resolve(__dirname, '../../ml/train.py');
const PREDICT_SCRIPT = path.resolve(__dirname, '../../ml/predict.py');

export async function runPythonTrain(orgId, runId, trainJsonPath, configJsonPath, outDir, prisma) {
  const logPath = path.join(outDir, 'train.log');
  const logHandle = await fs.open(logPath, 'a'); // append; create if missing

  // Emit an immediate log line so SSE clients don't sit on an empty file
  try {
    await logHandle.write(`[INFO] Training started for run ${runId} at ${new Date().toISOString()}\n`);
  } catch (e) {
    console.error('Failed to write initial log line:', e);
  }

  // ALSO keep a copy in config.json under outDir (some UIs expect it there)
  try {
    await fs.copyFile(configJsonPath, path.join(outDir, 'config.json'));
  } catch { } // best effort

  const modelId = runId;
  const args = [
    TRAIN_SCRIPT,
    '--model-id', modelId,
    '--resume',
    '--org-id', orgId,
    '--train-json', trainJsonPath,
    '--config-json', configJsonPath,
    '--out-dir', outDir,
  ];

  // Ensure unbuffered output and inherit env
  const pythonProcess = spawn(PYTHON_BIN, args, {
    env: { ...process.env, PYTHONUNBUFFERED: '1' },
    stdio: ['ignore', 'pipe', 'pipe'],
  });

  let resultJson = null;

  const catcher = makeResultCatcher({
    onProgress: async (progress) => {
      await prisma.trainingProgress.upsert({
        where: { modelId },
        update: progress,
        create: { modelId, ...progress },
      });
    },
    onError: (err) => {
      console.error('Python error:', err);
    },
  });
  
  pythonProcess.stdout.on('data', (data) => {
    catcher.write(data.toString('utf8'));
  });

  pythonProcess.stderr.on('data', async (data) => {
    await logHandle.write(data.toString());
  });

  return new Promise((resolve, reject) => {
    pythonProcess.on('close', async (code) => {
      if (!resultJson || resultJson.status !== 'ok') {
        try {
          await logHandle.write(`Training process exited with code ${code}.\n`);
        } catch (writeErr) {
          console.error('Failed to write close error to log:', writeErr);
        }
      }
      await logHandle.close();

      // Default-safe payloads for Prisma JSON columns
      const metrics = (resultJson && resultJson.metrics) ? resultJson.metrics : {};
      const plots = (resultJson && resultJson.plots) ? resultJson.plots : [];

      if (resultJson && resultJson.status === 'ok') {
        try {
          await prisma.modelRun.update({
            where: { id: runId },
            data: {
              status: 'SUCCEEDED',
              metrics,
              plots,
              finishedAt: new Date(),
            },
          });
        } catch (e) {
          // Don't hide success from caller if DB write fails
          console.error('Prisma update (SUCCEEDED) failed:', e);
        }
        resolve(resultJson);
      } else {
        try {
          await prisma.modelRun.update({
            where: { id: runId },
            data: {
              status: 'FAILED',
              finishedAt: new Date(),
            },
          });
        } catch (e) {
          console.error('Prisma update (FAILED) failed:', e);
        }

        let stderr = '';

        pythonProcess.stderr.on('data', async (data) => {
          const text = data.toString();
          stderr += text;
          await logHandle.write(text);
        });

        const errMsg = resultJson?.error || `Training failed with code ${code}\n${stderr.slice(-4000)}`;
        reject(new Error(errMsg));
      }
    });

    pythonProcess.on('error', async (error) => {
      try {
        await logHandle.write(`Failed to start Python process (${PYTHON_BIN}): ${error.message}\n`);
      } catch (writeErr) {
        console.error('Failed to write spawn error to log:', writeErr);
      }
      await logHandle.close();
      try {
        await prisma.modelRun.update({
          where: { id: runId },
          data: {
            status: 'FAILED',
            finishedAt: new Date(),
          },
        });
      } catch (e) {
        console.error('Prisma update (FAILED on spawn error) failed:', e);
      }
      reject(error);
    });
  });
}

export async function runPythonPredict(orgId, studentJsonPath, artifactsDir, outFile, creditHours) {
  const args = [
    PREDICT_SCRIPT,
    '--org-id', orgId,
    '--student-json', studentJsonPath,
    '--artifacts-dir', artifactsDir,
    '--out-file', outFile,
  ];

  if (creditHours !== undefined && creditHours !== null && creditHours !== '') {
    args.push('--credit-hours', String(creditHours));
  }

  const pythonProcess = spawn(PYTHON_BIN, args, {
    env: { ...process.env, PYTHONUNBUFFERED: '1' },
    stdio: ['ignore', 'pipe', 'pipe'],
  });


  let stderr = '';

  const catcher = makeResultCatcher({
    onProgress: async (progress) => {
      await prisma.trainingProgress.upsert({
        where: { modelId },
        update: progress,
        create: { modelId, ...progress },
      });
    },
    onError: (err) => {
      console.error('Python error:', err);
    },
  });
  
  pythonProcess.stdout.on('data', (data) => {
    catcher.write(data.toString('utf8'));
  });

  pythonProcess.stderr.on('data', (data) => {
    stderr += data.toString();
  });

  return new Promise((resolve, reject) => {
    pythonProcess.on('close', (code) => {
      const { result, error } = catcher.getResult();
    
      if (error) {
        reject(new Error(error.message));
        return;
      }
    
      if (!result) {
        reject(
          new Error(
            `Python exited without result (code ${code})\n${stderr.slice(-4000)}`
          )
        );
        return;
      }
    
      const parsed = TrainingResultSchema.safeParse(result);
      if (!parsed.success) {
        reject(
          new Error('Invalid result schema: ' + parsed.error.message)
        );
        return;
      }
      const parsed = TrainingResultSchema.safeParse(resultJson);
      if (!parsed.success) {
        reject(new Error('Invalid training result: ' + parsed.error.message));
        return;
      }
      resolve(parsed.data);
    });
    

    pythonProcess.on('error', (error) => {
      reject(error);
    });
  });
}
