import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import cookieParser from 'cookie-parser';

import authRoutes from './routes/auth.js';
import modelRoutes from './routes/models.js';
import predictRoutes from './routes/predict.js';
import exportRoutes from './routes/export.js';
import { STORAGE_ROOT } from './lib/storage.js';
import { runCleanup } from './utils/cleanup.js';

dotenv.config();

const app = express();
const PORT = process.env.PORT || 3000;
const CLEANUP_INTERVAL_MS = Number(process.env.CLEANUP_INTERVAL_MS || 60 * 60 * 1000);
const UPLOADS_KEEP_LATEST = Number(process.env.UPLOADS_KEEP_LATEST || 1);
const TRAIN_LOG_KEEP = Number(process.env.TRAIN_LOG_KEEP || 5);

// Middleware
app.use(cors({ origin: true, credentials: true }));
app.use(cookieParser());
app.use(express.json());

// Static files - serve storage directory
app.use('/static', express.static(STORAGE_ROOT));

// Routes
app.use('/api/auth', authRoutes);
app.use('/api/models', modelRoutes);
app.use('/api/predict', predictRoutes);
app.use('/api/export', exportRoutes);

// Health check
app.get('/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

const scheduleCleanup = () => {
  const intervalMs = Number.isFinite(CLEANUP_INTERVAL_MS) && CLEANUP_INTERVAL_MS > 0
    ? CLEANUP_INTERVAL_MS
    : 60 * 60 * 1000;
  const uploadsKeep = Number.isFinite(UPLOADS_KEEP_LATEST) && UPLOADS_KEEP_LATEST >= 0
    ? UPLOADS_KEEP_LATEST
    : 1;
  const trainLogsKeep = Number.isFinite(TRAIN_LOG_KEEP) && TRAIN_LOG_KEEP >= 0
    ? TRAIN_LOG_KEEP
    : 5;

  runCleanup({
    storageRoot: STORAGE_ROOT,
    uploadsKeepLatest: uploadsKeep,
    trainLogsDir: process.env.TRAIN_LOG_DIR,
    trainLogsKeep
  }).catch((err) => {
    console.error('Cleanup failed:', err);
  });

  setInterval(() => {
    runCleanup({
      storageRoot: STORAGE_ROOT,
      uploadsKeepLatest: uploadsKeep,
      trainLogsDir: process.env.TRAIN_LOG_DIR,
      trainLogsKeep
    }).catch((err) => {
      console.error('Cleanup failed:', err);
    });
  }, intervalMs);
};

// Error handling
app.use((err, req, res, next) => {
  console.error('Error:', err);
  res.status(err.status || 500).json({
    error: err.message || 'Internal server error',
    ...(process.env.NODE_ENV === 'development' && { stack: err.stack })
  });
});

const server = app.listen(PORT, () => {
  console.log(`🚀 Server running on http://localhost:${PORT}`);
  console.log(`📊 Static files served from /static`);
  console.log(`🐍 Python binary: ${process.env.PYTHON_BIN || 'python3'}`);
});

scheduleCleanup();

server.keepAliveTimeout = 300000;
server.headersTimeout = 305000;
