import { spawn, spawnSync } from 'child_process';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import { makeResultCatcher } from './makeResultCatcher.js';
import { TrainingResultSchema } from '../schemas/trainingResult.js';
import { PredictionResultSchema } from '../schemas/predictionResult.js';

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

async function loadTrainingMetadata(outDir) {
  try {
    const metadataPath = path.join(outDir, 'metadata.json');
    const raw = await fs.readFile(metadataPath, 'utf-8');
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

function buildMetricsFromMetadata(meta) {
  if (!meta || typeof meta !== 'object') return null;
  const bestModel = meta.best_model || meta.bestModel || null;
  let enabledModels = Array.isArray(meta.enabled_models)
    ? meta.enabled_models
    : Array.isArray(meta.enabledModels)
      ? meta.enabledModels
      : [];
  const models = meta.models && typeof meta.models === 'object' ? meta.models : {};
  if (!enabledModels.length && models) {
    enabledModels = Object.keys(models);
  }
  const bestMetrics = bestModel && models[bestModel] ? models[bestModel] : null;

  const summary = bestMetrics ? {
    rmse: Number(bestMetrics.rmse_te ?? bestMetrics.rmse ?? 0),
    r2: Number(bestMetrics.r2_te ?? bestMetrics.r2 ?? 0),
    mae: Number(bestMetrics.mae_te ?? bestMetrics.mae ?? 0),
    accuracy: meta.risk_accuracy != null ? Number(meta.risk_accuracy) : null
  } : null;

  const finalModels = Object.fromEntries(
    Object.entries(models).map(([name, values]) => [
      name,
      {
        rmse: Number(values.rmse_te ?? values.rmse ?? 0),
        r2: Number(values.r2_te ?? values.r2 ?? 0),
        mae: Number(values.mae_te ?? values.mae ?? 0)
      }
    ])
  );

  return {
    summary,
    bestModel,
    enabledModels,
    final: Object.keys(finalModels).length > 0 ? { models: finalModels } : undefined
  };
}

export async function runPythonTrain(orgId, runId, trainJsonPath, configJsonPath, outDir, prisma) {
  const logPath = path.join(outDir, 'train.log');
  const logHandle = await fs.open(logPath, 'a'); // append; create if missing

  let lastProgressHeader = null;

  function formatProgress(progress) {
    const phase = progress?.phase || 'progress';
    const model = progress?.model ? `model=${progress.model}` : null;
    const label = progress?.label ? `label=${progress.label}` : null;
    const epoch = progress?.epoch ? `epoch=${progress.epoch}` : null;
    const total = progress?.totalEpochs ? `total=${progress.totalEpochs}` : null;
    const samples = progress?.samples ? `samples=${progress.samples}` : null;
    const samplesFinal = progress?.samplesFinal ? `final=${progress.samplesFinal}` : null;
    const samplesNext = progress?.samplesNext ? `next=${progress.samplesNext}` : null;
    const valLoss = progress?.valLoss != null ? `valLoss=${Number(progress.valLoss).toFixed(6)}` : null;
    const parts = [model, label, epoch, total, samples, samplesFinal, samplesNext, valLoss].filter(Boolean);
    return `[PROGRESS] phase=${phase}${parts.length ? ' ' + parts.join(' ') : ''}`;
  }

  function formatProgressHeader(progress) {
    const phase = progress?.phase || 'progress';
    const model = progress?.model ? `model=${progress.model}` : null;
    const label = progress?.label ? `label=${progress.label}` : null;
    const total = progress?.totalEpochs ? `total=${progress.totalEpochs}` : null;
    const samples = progress?.samples ? `samples=${progress.samples}` : null;
    const samplesFinal = progress?.samplesFinal ? `final=${progress.samplesFinal}` : null;
    const samplesNext = progress?.samplesNext ? `next=${progress.samplesNext}` : null;
    const parts = [model, label, total, samples, samplesFinal, samplesNext].filter(Boolean);
    return `[PROGRESS] phase=${phase}${parts.length ? ' ' + parts.join(' ') : ''}`;
  }

  function formatProgressDetail(progress) {
    const epoch = progress?.epoch ? `epoch=${progress.epoch}` : null;
    const valLoss = progress?.valLoss != null ? `valLoss=${Number(progress.valLoss).toFixed(6)}` : null;
    const parts = [epoch, valLoss].filter(Boolean);
    return parts.length ? `  ${parts.join(' ')}` : null;
  }

  console.log('[train] starting', {
    orgId,
    runId,
    trainJsonPath,
    configJsonPath,
    outDir,
    pythonBin: PYTHON_BIN,
    script: TRAIN_SCRIPT
  });

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

  console.log('[train] spawn args', args);

  // Ensure unbuffered output and inherit env
  const pythonProcess = spawn(PYTHON_BIN, args, {
    env: { ...process.env, PYTHONUNBUFFERED: '1' },
    stdio: ['ignore', 'pipe', 'pipe'],
  });

  let resultJson = null;
  let stderr = '';

  const catcher = makeResultCatcher({
    onResult: (payload) => {
      resultJson = payload;
    },
    onProgress: async (progress) => {
      try {
        const header = formatProgressHeader(progress);
        if (header !== lastProgressHeader) {
          await logHandle.write(`${header}\n`);
          lastProgressHeader = header;
        }
        const detail = formatProgressDetail(progress);
        if (detail) {
          await logHandle.write(`${detail}\n`);
        }
      } catch (err) {
        console.error('Failed to write progress to log:', err);
      }
      if (!prisma?.trainingProgress?.upsert) return;
      try {
        await prisma.trainingProgress.upsert({
          where: { modelId },
          update: progress,
          create: { modelId, ...progress },
        });
      } catch (err) {
        console.error('Failed to persist training progress:', err);
      }
    },
    onError: (err) => {
      console.error('Python error:', err);
    },
  });

  pythonProcess.stdout.on('data', (data) => {
    const text = data.toString('utf8');
    if (text.includes('__RESULT__')) {
      console.log('[train] result received');
    }
    catcher.write(text);
    const cleaned = text
      .split('\n')
      .filter((line) => line && !line.startsWith('__PROGRESS__') && !line.startsWith('__RESULT__'))
      .join('\n');
    if (!cleaned) return;
    logHandle.write(cleaned + '\n').catch((err) => {
      console.error('Failed to write stdout to log:', err);
    });
  });

  pythonProcess.stderr.on('data', async (data) => {
    const text = data.toString();
    console.log('[train] stderr chunk', text.slice(0, 2000));
    stderr += text;
    await logHandle.write(text);
  });

  return new Promise((resolve, reject) => {
    pythonProcess.on('close', async (code) => {
      console.log('[train] process closed', { code });
      const { result, error } = catcher.getResult();
      if (!resultJson && result) {
        resultJson = result;
      }

      if (error) {
        resultJson = { status: 'error', error: error.message };
      }

      console.log('[train] result', resultJson);

      if (!resultJson || resultJson.status !== 'ok') {
        try {
          await logHandle.write(`Training process exited with code ${code}.\n`);
        } catch (writeErr) {
          console.error('Failed to write close error to log:', writeErr);
        }
      }
      await logHandle.close();

      if (resultJson && resultJson.status === 'ok') {
        const parsed = TrainingResultSchema.safeParse(resultJson);
        if (!parsed.success) {
          console.error('[train] result validation failed', parsed.error);
          reject(new Error('Invalid training result: ' + parsed.error.message));
          return;
        }

        const metadata = await loadTrainingMetadata(outDir);
        if (!metadata) {
          console.warn('[train] metadata.json missing or unreadable', { outDir });
        }
        const metricsFromMetadata = buildMetricsFromMetadata(metadata);
        const metrics = parsed.data.metrics || metricsFromMetadata || {};
        const plots = parsed.data.plots || {};

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

        const errMsg = resultJson?.error || `Training failed with code ${code}\n${stderr.slice(-4000)}`;
        console.error('[train] failed', errMsg);
        reject(new Error(errMsg));
      }
    });

    pythonProcess.on('error', async (error) => {
      console.error('[train] spawn error', error);
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

      if (result.status === 'error') {
        reject(new Error(result.error || 'Prediction failed'));
        return;
      }

      const parsedResJson = PredictionResultSchema.safeParse(result);
      if (!parsedResJson.success) {
        reject(new Error('Invalid prediction result: ' + parsedResJson.error.message));
        return;
      }
      resolve(parsedResJson.data);
    });


    pythonProcess.on('error', (error) => {
      reject(error);
    });
  });
}
