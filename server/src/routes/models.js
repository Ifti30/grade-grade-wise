import express from 'express';
import path from 'path';
import fs from 'fs/promises';
import multer from 'multer';
import { authenticateToken } from '../auth.js';
import prisma from '../lib/prisma.js';
import { STORAGE_ROOT, normalizePlotList, normalizePlotObject } from '../lib/storage.js';
import { createOrgUploadStorage, jsonFileFilter } from '../lib/uploads.js';
import { validateConfig } from '../utils/validate-config.js';
import { runPythonTrain, terminateTrainingRun } from '../utils/python-runner.js';
import { readLogChunk } from '../utils/log-stream.js';

const router = express.Router();
const SUMMARY_CHUNK_SIZE = 200;

const chunkArray = (items, size) => {
  if (!Array.isArray(items) || items.length === 0) return [];
  const chunks = [];
  for (let i = 0; i < items.length; i += size) {
    chunks.push(items.slice(i, i + size));
  }
  return chunks;
};

const writeSse = (res, payload) => {
  res.write(`data: ${JSON.stringify(payload)}\n\n`);
};

function normalizePlotMap(plots) {
  if (!plots) return {};

  // If an array of paths, convert to keyed object based on filename
  if (Array.isArray(plots)) {
    return normalizePlotList(plots);
  }

  if (typeof plots === 'object') {
    return normalizePlotObject(plots);
  }

  return {};
}

// Configure multer for file uploads
const upload = multer({
  storage: createOrgUploadStorage('uploads', 'train'),
  fileFilter: jsonFileFilter
});

// Get model status
router.get('/status', authenticateToken, async (req, res) => {
  try {
    const lastRun = await prisma.modelRun.findFirst({
      where: { orgId: req.orgId },
      orderBy: { createdAt: 'desc' }
    });

    const hasModel = lastRun && lastRun.status === 'SUCCEEDED';

    res.json({
      hasModel,
      lastRun: lastRun ? {
        id: lastRun.id,
        status: lastRun.status,
        createdAt: lastRun.createdAt,
        finishedAt: lastRun.finishedAt
      } : null
    });
  } catch (error) {
    console.error('Status check error:', error);
    res.status(500).json({ error: 'Failed to check model status' });
  }
});

// Start training
router.post('/train', authenticateToken, upload.single('trainFile'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'Training file is required' });
    }

    console.log('[train] request', {
      orgId: req.orgId,
      filename: req.file?.filename,
      path: req.file?.path,
      size: req.file?.size
    });

    // Parse and validate config
    let config;
    try {
      config = JSON.parse(req.body.config || '{}');
      config = validateConfig(config);
    } catch (error) {
      return res.status(400).json({ error: error.message });
    }

    // Capture any previous runs so we can remove them after creating the new one
    const previousRuns = await prisma.modelRun.findMany({
      where: { orgId: req.orgId },
      select: { id: true, artifactsDir: true }
    });

    // Create model run record
    const modelRun = await prisma.modelRun.create({
      data: {
        orgId: req.orgId,
        status: 'PENDING',
        config
      }
    });

    // Create run directory
    const runDir = path.join(STORAGE_ROOT, 'models', req.orgId, modelRun.id);
    await fs.mkdir(runDir, { recursive: true });

    // Save config to file
    const configPath = path.join(runDir, 'config.json');
    await fs.writeFile(configPath, JSON.stringify(config, null, 2));

    // Update status to RUNNING
    await prisma.modelRun.update({
      where: { id: modelRun.id },
      data: { status: 'RUNNING', artifactsDir: runDir }
    });

    // Remove previous runs and their artifacts so only the latest remains
    if (previousRuns.length > 0) {
      const oldIds = previousRuns.map(run => run.id);
      await prisma.modelRun.deleteMany({ where: { id: { in: oldIds } } });

      await Promise.all(previousRuns.map(async (run) => {
        if (run.artifactsDir) {
          try {
            await fs.rm(run.artifactsDir, { recursive: true, force: true });
          } catch (err) {
            console.error(`Failed to delete artifacts for run ${run.id}:`, err);
          }
        }
      }));

      const orgModelsDir = path.join(STORAGE_ROOT, 'models', req.orgId);
      try {
        const entries = await fs.readdir(orgModelsDir, { withFileTypes: true });
        await Promise.all(entries.map(async (entry) => {
          if (entry.isDirectory() && entry.name !== modelRun.id) {
            try {
              await fs.rm(path.join(orgModelsDir, entry.name), { recursive: true, force: true });
            } catch (err) {
              console.error(`Failed to delete leftover directory ${entry.name}:`, err);
            }
          }
        }));
      } catch (err) {
        if (err.code !== 'ENOENT') {
          console.error('Failed to clean model directory:', err);
        }
      }
    }

    // Start training asynchronously
    runPythonTrain(req.orgId, modelRun.id, req.file.path, configPath, runDir, prisma)
      .catch(error => {
        console.error('Training failed:', error);
      });

    res.json({ runId: modelRun.id });
  } catch (error) {
    console.error('Train start error:', error);
    res.status(500).json({ error: 'Failed to start training' });
  }
});

// Stream training logs (SSE)
router.get('/train/:runId/logs', async (req, res) => {
  try {
    const { runId } = req.params;
    const token = req.query.token;

    if (!token) {
      return res.status(401).json({ error: 'Access token required' });
    }

    // Verify token manually (EventSource doesn't support headers)
    let decoded;
    try {
      const jwt = await import('jsonwebtoken');
      decoded = jwt.default.verify(token, process.env.JWT_SECRET);
    } catch (error) {
      return res.status(403).json({ error: 'Invalid or expired token' });
    }

    // Fetch user to verify orgId
    const user = await prisma.user.findUnique({
      where: { id: decoded.userId }
    });

    if (!user) {
      return res.status(403).json({ error: 'User not found' });
    }

    // Verify run belongs to org
    const modelRun = await prisma.modelRun.findFirst({
      where: { id: runId, orgId: user.orgId }
    });

    if (!modelRun) {
      return res.status(404).json({ error: 'Training run not found' });
    }

    const logPath = path.join(modelRun.artifactsDir, 'train.log');

    // Set SSE headers
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    res.flushHeaders();

    const rawOffset = req.query.offset ?? req.headers['last-event-id'];
    const parsedOffset = Number(rawOffset);
    let lastSize = Number.isFinite(parsedOffset) && parsedOffset >= 0 ? Math.floor(parsedOffset) : 0;
    let attempts = 0;
    const maxAttempts = 600; // 10 minutes

    const sendLogs = async () => {
      try {
        const { chunk, nextOffset } = await readLogChunk(logPath, lastSize);
        if (chunk) {
          res.write(`id: ${nextOffset}\n`);
          res.write(`data: ${JSON.stringify({ content: chunk })}\n\n`);
          lastSize = nextOffset;
          attempts = 0;
        }

        // Check if training is complete
        const currentRun = await prisma.modelRun.findUnique({
          where: { id: runId }
        });

        if (currentRun.status === 'SUCCEEDED' || currentRun.status === 'FAILED') {
          res.write(`id: ${lastSize}\n`);
          res.write(`data: ${JSON.stringify({ status: currentRun.status, complete: true })}\n\n`);
          res.end();
          return;
        }

        attempts++;
        if (attempts < maxAttempts) {
          setTimeout(sendLogs, 1000);
        } else {
          res.end();
        }
      } catch (error) {
        if (error.code !== 'ENOENT') {
          console.error('Log streaming error:', error);
        }
        if (attempts < maxAttempts) {
          setTimeout(sendLogs, 1000);
        } else {
          res.end();
        }
      }
    };

    // Start sending logs
    sendLogs();

    // Handle client disconnect
    req.on('close', () => {
      res.end();
    });
  } catch (error) {
    console.error('SSE setup error:', error);
    res.status(500).json({ error: 'Failed to stream logs' });
  }
});

// Terminate training run
router.post('/train/:runId/terminate', authenticateToken, async (req, res) => {
  try {
    const { runId } = req.params;
    const modelRun = await prisma.modelRun.findFirst({
      where: { id: runId, orgId: req.orgId }
    });

    if (!modelRun) {
      return res.status(404).json({ error: 'Training run not found' });
    }

    if (modelRun.status !== 'RUNNING' && modelRun.status !== 'PENDING') {
      return res.status(409).json({ error: 'Training is not running' });
    }

    const terminated = terminateTrainingRun(runId);
    if (!terminated) {
      return res.status(409).json({ error: 'Training process not active on server' });
    }

    try {
      const logPath = path.join(modelRun.artifactsDir, 'train.log');
      await fs.appendFile(logPath, `[WARN] Termination requested at ${new Date().toISOString()}\n`);
    } catch (error) {
      console.error('Failed to append termination log:', error);
    }

    res.json({ status: 'terminating' });
  } catch (error) {
    console.error('Terminate training error:', error);
    res.status(500).json({ error: 'Failed to terminate training' });
  }
});

// Get training summary
router.get('/summary', authenticateToken, async (req, res) => {
  try {
    const lastSucceeded = await prisma.modelRun.findFirst({
      where: {
        orgId: req.orgId,
        status: 'SUCCEEDED'
      },
      orderBy: { createdAt: 'desc' }
    });

    if (!lastSucceeded) {
      return res.json({ hasModel: false });
    }

    // Standardize metrics shape for frontend
    const metrics = (() => {
      const m = lastSucceeded.metrics;
      if (!m) return {};
      if (!Array.isArray(m) && typeof m === 'object') return m;
      if (Array.isArray(m)) {
        // Pick best model by smallest RMSE (or first as fallback)
        const pick = [...m].sort((a, b) => {
          const ra = Number(a.RMSE ?? a.rmse ?? Infinity);
          const rb = Number(b.RMSE ?? b.rmse ?? Infinity);
          return ra - rb;
        })[0] || m[0];
        return {
          rmse: Number(pick?.RMSE ?? pick?.rmse ?? 0),
          mae: Number(pick?.MAE ?? pick?.mae ?? 0),
          r2: Number(pick?.R2 ?? pick?.r2 ?? 0) || null,
          accuracy: Number(pick?.accuracy ?? pick?.Accuracy ?? 0) || null
        };
      }
      return {};
    })();

    res.json({
      hasModel: true,
      metrics,
      plots: normalizePlotMap(lastSucceeded.plots),
      artifactsDir: lastSucceeded.artifactsDir,
      report: await (async () => {
        try {
          const reportPath = path.join(lastSucceeded.artifactsDir, 'report.json');
          const raw = await fs.readFile(reportPath, 'utf-8');
          return JSON.parse(raw);
        } catch {
          return null;
        }
      })(),
      createdAt: lastSucceeded.createdAt,
      config: lastSucceeded.config
    });
  } catch (error) {
    console.error('Summary error:', error);
    res.status(500).json({ error: 'Failed to get summary' });
  }
});

// Stream training summary (SSE)
router.get('/summary/stream', async (req, res) => {
  try {
    const token = req.query.token;
    const requestedRunId = req.query.runId;

    if (!token) {
      return res.status(401).json({ error: 'Access token required' });
    }

    // Verify token manually (EventSource doesn't support headers)
    let decoded;
    try {
      const jwt = await import('jsonwebtoken');
      decoded = jwt.default.verify(token, process.env.JWT_SECRET);
    } catch (error) {
      return res.status(403).json({ error: 'Invalid or expired token' });
    }

    // Fetch user to verify orgId
    const user = await prisma.user.findUnique({
      where: { id: decoded.userId }
    });

    if (!user) {
      return res.status(403).json({ error: 'User not found' });
    }

    const runLookup = requestedRunId
      ? { id: String(requestedRunId), orgId: user.orgId }
      : { orgId: user.orgId, status: 'SUCCEEDED' };

    const lastSucceeded = await prisma.modelRun.findFirst({
      where: runLookup,
      orderBy: { createdAt: 'desc' }
    });

    // Set SSE headers
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    res.flushHeaders();

    if (!lastSucceeded) {
      writeSse(res, { type: 'complete', totalChunks: 0, index: 0, hasModel: false });
      res.end();
      return;
    }

    const metrics = (() => {
      const m = lastSucceeded.metrics;
      if (!m) return {};
      if (!Array.isArray(m) && typeof m === 'object') return m;
      if (Array.isArray(m)) {
        const pick = [...m].sort((a, b) => {
          const ra = Number(a.RMSE ?? a.rmse ?? Infinity);
          const rb = Number(b.RMSE ?? b.rmse ?? Infinity);
          return ra - rb;
        })[0] || m[0];
        return {
          rmse: Number(pick?.RMSE ?? pick?.rmse ?? 0),
          mae: Number(pick?.MAE ?? pick?.mae ?? 0),
          r2: Number(pick?.R2 ?? pick?.r2 ?? 0) || null,
          accuracy: Number(pick?.accuracy ?? pick?.Accuracy ?? 0) || null
        };
      }
      return {};
    })();

    const plots = normalizePlotMap(lastSucceeded.plots);
    let report = null;
    try {
      const reportPath = path.join(lastSucceeded.artifactsDir, 'report.json');
      const raw = await fs.readFile(reportPath, 'utf-8');
      report = JSON.parse(raw);
    } catch {
      report = null;
    }

    const chunks = [];
    chunks.push({ type: 'meta', payload: {
      hasModel: true,
      createdAt: lastSucceeded.createdAt,
      artifactsDir: lastSucceeded.artifactsDir
    }});
    chunks.push({ type: 'metrics', payload: metrics });
    chunks.push({ type: 'plots', payload: plots });
    chunks.push({ type: 'config', payload: lastSucceeded.config || {} });

    if (report && typeof report === 'object') {
      chunks.push({ type: 'report', path: 'schema', payload: {
        schema_version: report.schema_version,
        created_at: report.created_at,
        splitting: report.splitting
      }});

      const dataset = report.dataset || {};
      chunks.push({ type: 'report', path: 'dataset.stats', payload: dataset.stats || {} });
      for (const [key, values] of Object.entries({
        final_cgpa_hist: dataset.final_cgpa_hist,
        next_sem_cgpa_hist: dataset.next_sem_cgpa_hist
      })) {
        const parts = chunkArray(values, SUMMARY_CHUNK_SIZE);
        parts.forEach((part, idx) => {
          chunks.push({
            type: 'report',
            path: `dataset.${key}`,
            index: idx,
            total: parts.length,
            payload: part
          });
        });
      }

      const regression = report.regression || {};
      for (const [taskKey, task] of Object.entries(regression)) {
        if (!task || typeof task !== 'object') continue;
        if (task.bestModel) {
          chunks.push({ type: 'report', path: `regression.${taskKey}.bestModel`, payload: task.bestModel });
        }
        const metricsBlock = task.metrics || {};
        if (metricsBlock.models) {
          chunks.push({ type: 'report', path: `regression.${taskKey}.metrics.models`, payload: metricsBlock.models });
        }
        if (metricsBlock.testSize != null) {
          chunks.push({ type: 'report', path: `regression.${taskKey}.metrics.testSize`, payload: metricsBlock.testSize });
        }
        const predictions = metricsBlock.predictions || {};
        for (const [modelName, values] of Object.entries(predictions)) {
          const parts = chunkArray(values, SUMMARY_CHUNK_SIZE);
          parts.forEach((part, idx) => {
            chunks.push({
              type: 'report',
              path: `regression.${taskKey}.metrics.predictions.${modelName}`,
              index: idx,
              total: parts.length,
              payload: part
            });
          });
        }
        const featureImportance = metricsBlock.featureImportance || {};
        for (const [modelName, values] of Object.entries(featureImportance)) {
          const parts = chunkArray(values, SUMMARY_CHUNK_SIZE);
          parts.forEach((part, idx) => {
            chunks.push({
              type: 'report',
              path: `regression.${taskKey}.metrics.featureImportance.${modelName}`,
              index: idx,
              total: parts.length,
              payload: part
            });
          });
        }
        const learningCurves = metricsBlock.learningCurves || {};
        for (const [modelName, curves] of Object.entries(learningCurves)) {
          if (!curves || typeof curves !== 'object') continue;
          for (const [curveKey, values] of Object.entries(curves)) {
            const parts = chunkArray(values, SUMMARY_CHUNK_SIZE);
            parts.forEach((part, idx) => {
              chunks.push({
                type: 'report',
                path: `regression.${taskKey}.metrics.learningCurves.${modelName}.${curveKey}`,
                index: idx,
                total: parts.length,
                payload: part
              });
            });
          }
        }
        if (task.residualSamples) {
          const parts = chunkArray(task.residualSamples, SUMMARY_CHUNK_SIZE);
          parts.forEach((part, idx) => {
            chunks.push({
              type: 'report',
              path: `regression.${taskKey}.residualSamples`,
              index: idx,
              total: parts.length,
              payload: part
            });
          });
        }
        if (task.split) {
          chunks.push({ type: 'report', path: `regression.${taskKey}.split`, payload: task.split });
        }
      }

      const classification = report.classification || {};
      chunks.push({ type: 'report', path: 'classification.summary', payload: {
        risk_target: classification.risk_target,
        thresholds: classification.thresholds,
        labels: classification.labels,
        accuracy: classification.accuracy,
        precision_macro: classification.precision_macro,
        recall_macro: classification.recall_macro,
        f1_macro: classification.f1_macro,
        precision_weighted: classification.precision_weighted,
        recall_weighted: classification.recall_weighted,
        f1_weighted: classification.f1_weighted
      }});
      if (classification.confusion_matrix) {
        chunks.push({ type: 'report', path: 'classification.confusion_matrix', payload: classification.confusion_matrix });
      }
    }

    writeSse(res, { type: 'start', totalChunks: chunks.length, index: 0 });
    if (typeof res.flush === 'function') {
      res.flush();
    }

    let idx = 0;
    const intervalMs = Number(process.env.SUMMARY_STREAM_INTERVAL_MS || 20);
    const timer = setInterval(() => {
      if (idx >= chunks.length) {
        writeSse(res, { type: 'complete', totalChunks: chunks.length, index: chunks.length });
        if (typeof res.flush === 'function') {
          res.flush();
        }
        clearInterval(timer);
        res.end();
        return;
      }
      const chunk = chunks[idx];
      writeSse(res, { ...chunk, index: idx + 1, totalChunks: chunks.length });
      if (typeof res.flush === 'function') {
        res.flush();
      }
      idx += 1;
    }, Math.max(5, intervalMs));

    req.on('close', () => {
      clearInterval(timer);
    });
  } catch (error) {
    console.error('Summary stream error:', error);
    res.status(500).json({ error: 'Failed to stream summary' });
  }
});

export default router;
