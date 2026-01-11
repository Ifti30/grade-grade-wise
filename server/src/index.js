import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import path from 'path';
import { fileURLToPath } from 'url';

import authRoutes from './routes/auth.js';
import modelRoutes from './routes/models.js';
import predictRoutes from './routes/predict.js';

dotenv.config();

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());

// Static files - serve storage directory
const storagePath = path.join(__dirname, '../storage');
app.use('/static', express.static(storagePath));

// Routes
app.use('/api/auth', authRoutes);
app.use('/api/models', modelRoutes);
app.use('/api/predict', predictRoutes);

// Health check
app.get('/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

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

server.keepAliveTimeout = 300000;
server.headersTimeout = 305000;