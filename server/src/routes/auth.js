import express from 'express';
import bcrypt from 'bcrypt';
import jwt from 'jsonwebtoken';
import crypto from 'crypto';
import { authenticateToken } from '../auth.js';
import prisma from '../lib/prisma.js';

const router = express.Router();
const ACCESS_TOKEN_TTL = '7d';
const REFRESH_TOKEN_DAYS = 30;

function createAccessToken(userId, orgId) {
  return jwt.sign(
    { userId, orgId },
    process.env.JWT_SECRET,
    { expiresIn: ACCESS_TOKEN_TTL }
  );
}

function createRefreshToken() {
  return crypto.randomBytes(48).toString('hex');
}

function hashRefreshToken(token) {
  return crypto.createHash('sha256').update(token).digest('hex');
}

function addRefreshCookie(res, token) {
  res.cookie('refresh_token', token, {
    httpOnly: true,
    sameSite: 'lax',
    secure: process.env.NODE_ENV === 'production',
    path: '/api/auth',
  });
}

async function issueTokensForUser(res, user) {
  const token = createAccessToken(user.id, user.orgId);
  const refreshToken = createRefreshToken();
  const refreshTokenHash = hashRefreshToken(refreshToken);
  const refreshTokenExpires = new Date(Date.now() + REFRESH_TOKEN_DAYS * 24 * 60 * 60 * 1000);

  await prisma.user.update({
    where: { id: user.id },
    data: { refreshTokenHash, refreshTokenExpires }
  });

  addRefreshCookie(res, refreshToken);
  return token;
}

// Sign up
router.post('/signup', async (req, res) => {
  try {
    const { orgName, email, password } = req.body;

    if (!orgName || !email || !password) {
      return res.status(400).json({ error: 'orgName, email, and password are required' });
    }

    // Check if user exists
    const existingUser = await prisma.user.findUnique({ where: { email } });
    if (existingUser) {
      return res.status(400).json({ error: 'Email already registered' });
    }

    // Hash password
    const passwordHash = await bcrypt.hash(password, 10);

    // Create org and user in transaction
    const result = await prisma.$transaction(async (tx) => {
      const org = await tx.organization.create({
        data: { name: orgName }
      });

      const user = await tx.user.create({
        data: {
          orgId: org.id,
          email,
          passwordHash
        },
        include: { org: true }
      });

      return { org, user };
    });

    // Generate JWT
    const token = await issueTokensForUser(res, result.user);

    res.status(201).json({
      token,
      user: {
        id: result.user.id,
        email: result.user.email,
        createdAt: result.user.createdAt
      },
      org: {
        id: result.org.id,
        name: result.org.name
      }
    });
  } catch (error) {
    console.error('Signup error:', error);
    res.status(500).json({ error: 'Signup failed' });
  }
});

// Sign in
router.post('/signin', async (req, res) => {
  try {
    const { email, password } = req.body;

    if (!email || !password) {
      return res.status(400).json({ error: 'Email and password are required' });
    }

    // Find user with org
    const user = await prisma.user.findUnique({
      where: { email },
      include: { org: true }
    });

    if (!user) {
      return res.status(401).json({ error: 'Invalid credentials' });
    }

    // Verify password
    const validPassword = await bcrypt.compare(password, user.passwordHash);
    if (!validPassword) {
      return res.status(401).json({ error: 'Invalid credentials' });
    }

    // Generate JWT
    const token = await issueTokensForUser(res, user);

    res.json({
      token,
      user: {
        id: user.id,
        email: user.email,
        createdAt: user.createdAt
      },
      org: {
        id: user.org.id,
        name: user.org.name
      }
    });
  } catch (error) {
    console.error('Signin error:', error);
    res.status(500).json({ error: 'Signin failed' });
  }
});

// Refresh access token
router.post('/refresh', async (req, res) => {
  try {
    const refreshToken = req.cookies?.refresh_token;
    if (!refreshToken) {
      return res.status(401).json({ error: 'Refresh token missing' });
    }

    const refreshTokenHash = hashRefreshToken(refreshToken);
    const user = await prisma.user.findFirst({
      where: { refreshTokenHash }
    });

    if (!user || !user.refreshTokenExpires || user.refreshTokenExpires < new Date()) {
      return res.status(401).json({ error: 'Invalid or expired refresh token' });
    }

    const token = await issueTokensForUser(res, user);
    res.json({ token });
  } catch (error) {
    console.error('Refresh error:', error);
    res.status(500).json({ error: 'Refresh failed' });
  }
});

// Sign out
router.post('/signout', async (req, res) => {
  try {
    const refreshToken = req.cookies?.refresh_token;
    if (refreshToken) {
      const refreshTokenHash = hashRefreshToken(refreshToken);
      await prisma.user.updateMany({
        where: { refreshTokenHash },
        data: { refreshTokenHash: null, refreshTokenExpires: null }
      });
    }
    res.clearCookie('refresh_token', { path: '/api/auth' });
    res.json({ status: 'ok' });
  } catch (error) {
    console.error('Signout error:', error);
    res.status(500).json({ error: 'Signout failed' });
  }
});

// Get current user
router.get('/me', authenticateToken, async (req, res) => {
  res.json({
    user: {
      id: req.user.id,
      email: req.user.email,
      createdAt: req.user.createdAt
    },
    org: {
      id: req.user.org.id,
      name: req.user.org.name
    }
  });
});

export default router;
