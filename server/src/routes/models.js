import express from 'express';
import multer from 'multer';
import path from 'path';
import fs from 'fs/promises';
import { fileURLToPath } from 'url';
import { PrismaClient } from '@prisma/client';
import { authenticateToken } from '../auth.js';
import { validateConfig } from '../utils/validate-config.js';
import { runPythonTrain } from '../utils/python-runner.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const router = express.Router();
const prisma = new PrismaClient();
const STORAGE_ROOT = path.join(__dirname, '../../storage');

function toStaticPath(absPath) {
  if (!absPath || typeof absPath !== 'string') return absPath;
  const normalized = path.normalize(absPath);
  if (normalized.startsWith(STORAGE_ROOT)) {
    const rel = normalized.slice(STORAGE_ROOT.length).replace(/\\/g, '/');
    return `/static${rel}`;
  }
  return absPath;
}

function normalizePlotMap(plots) {
  if (!plots) return {};

  // If an array of paths, convert to keyed object based on filename
  if (Array.isArray(plots)) {
    const entries = plots
      .filter(Boolean)
      .map((p) => {
        const key = path.basename(p, path.extname(p)) || 'plot';
        return [key, toStaticPath(p)];
      });
    return Object.fromEntries(entries);
  }

  if (typeof plots === 'object') {
    return Object.fromEntries(
      Object.entries(plots).map(([key, value]) => [key, toStaticPath(value)])
    );
  }

  return {};
}

// Configure multer for file uploads
const storage = multer.diskStorage({
  destination: async (req, file, cb) => {
    const uploadDir = path.join(__dirname, '../../storage/uploads', req.orgId);
    await fs.mkdir(uploadDir, { recursive: true });
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    const timestamp = Date.now();
    cb(null, `train_${timestamp}.json`);
  }
});

const upload = multer({
  storage,
  fileFilter: (req, file, cb) => {
    if (file.mimetype === 'application/json' || file.originalname.endsWith('.json')) {
      cb(null, true);
    } else {
      cb(new Error('Only JSON files are allowed'));
    }
  }
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
    const runDir = path.join(__dirname, '../../storage/models', req.orgId, modelRun.id);
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

      const orgModelsDir = path.join(__dirname, '../../storage/models', req.orgId);
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

    let lastSize = 0;
    let attempts = 0;
    const maxAttempts = 600; // 10 minutes

    const sendLogs = async () => {
      try {
        const stats = await fs.stat(logPath);
        if (stats.size > lastSize) {
          const stream = await fs.readFile(logPath, 'utf-8');
          const newContent = stream.slice(lastSize);
          lastSize = stats.size;

          res.write(`data: ${JSON.stringify({ content: newContent })}\n\n`);
        }

        // Check if training is complete
        const currentRun = await prisma.modelRun.findUnique({
          where: { id: runId }
        });

        if (currentRun.status === 'SUCCEEDED' || currentRun.status === 'FAILED') {
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
      createdAt: lastSucceeded.createdAt,
      config: lastSucceeded.config
    });
  } catch (error) {
    console.error('Summary error:', error);
    res.status(500).json({ error: 'Failed to get summary' });
  }
});

export default router;
