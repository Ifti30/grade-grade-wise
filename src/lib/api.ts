// Point frontend to NAS-hosted backend
// const API_URL = 'http://thisispersonal.technospective.se:1000/api';
const API_URL = 'http://localhost:3000/api'
const API_BASE = API_URL.replace(/\/api$/, '');

export const getApiBase = () => API_BASE;

const decodeJwtPayload = (token: string) => {
  const parts = token.split('.');
  if (parts.length !== 3) return null;
  const payload = parts[1].replace(/-/g, '+').replace(/_/g, '/');
  const padded = payload + '='.repeat((4 - (payload.length % 4)) % 4);
  try {
    return JSON.parse(atob(padded));
  } catch {
    return null;
  }
};

const isTokenExpired = (token: string, skewSeconds = 60) => {
  const payload = decodeJwtPayload(token);
  if (!payload || typeof payload.exp !== 'number') return false;
  const expMs = payload.exp * 1000;
  return Date.now() + skewSeconds * 1000 >= expMs;
};

export const buildStaticUrl = (path?: string | null) => {
  if (!path) return null;
  if (path.startsWith('http://') || path.startsWith('https://')) {
    return path;
  }
  if (path.startsWith('/static/')) {
    return `${API_BASE}${path}`;
  }
  if (path.startsWith('static/')) {
    return `${API_BASE}/${path}`;
  }
  return path;
};

interface AuthResponse {
  token: string;
  user: {
    id: string;
    email: string;
  };
  org: {
    id: string;
    name: string;
  };
}

class ApiClient {
  private token: string | null = null;

  constructor() {
    this.token = localStorage.getItem('token');
  }

  setToken(token: string) {
    this.token = token;
    localStorage.setItem('token', token);
  }

  clearToken() {
    this.token = null;
    localStorage.removeItem('token');
  }

  async getValidToken() {
    const token = this.token || localStorage.getItem('token');
    if (token && !isTokenExpired(token)) {
      this.token = token;
      return token;
    }
    const refreshed = await this.refreshToken();
    return refreshed ? this.token : null;
  }

  private async fetch(endpoint: string, options: RequestInit = {}) {
    const retry = (options as { _retry?: boolean })._retry !== true;
    const headers: HeadersInit = {
      ...options.headers,
    };

    if (this.token && !headers['Authorization']) {
      headers['Authorization'] = `Bearer ${this.token}`;
    }

    const response = await fetch(`${API_URL}${endpoint}`, {
      ...options,
      credentials: 'include',
      headers,
    });

    if (response.status === 401 && retry) {
      const refreshed = await this.refreshToken();
      if (refreshed) {
        return this.fetch(endpoint, { ...options, _retry: true } as RequestInit);
      }
      this.clearToken();
      window.location.href = '/signin';
      throw new Error('Unauthorized');
    }

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || 'Request failed');
    }

    return data;
  }

  // Auth
  async signup(orgName: string, email: string, password: string): Promise<AuthResponse> {
    const data = await this.fetch('/auth/signup', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ orgName, email, password }),
    });
    this.setToken(data.token);
    return data;
  }

  async signin(email: string, password: string): Promise<AuthResponse> {
    const data = await this.fetch('/auth/signin', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
    });
    this.setToken(data.token);
    return data;
  }

  async refreshToken() {
    try {
      const response = await fetch(`${API_URL}/auth/refresh`, {
        method: 'POST',
        credentials: 'include',
      });
      if (!response.ok) {
        if (response.status === 401) {
          this.clearToken();
        }
        return false;
      }
      const data = await response.json();
      if (data?.token) {
        this.setToken(data.token);
        return true;
      }
      return false;
    } catch {
      return false;
    }
  }

  async signout() {
    try {
      await fetch(`${API_URL}/auth/signout`, {
        method: 'POST',
        credentials: 'include',
      });
    } catch {
      // best-effort; still clear local state
    } finally {
      this.clearToken();
    }
  }

  async me() {
    return this.fetch('/auth/me');
  }

  // Models
  async getModelStatus() {
    return this.fetch('/models/status');
  }

  async startTraining(trainFile: File, config: any) {
    const formData = new FormData();
    formData.append('trainFile', trainFile);
    formData.append('config', JSON.stringify(config));

    return this.fetch('/models/train', {
      method: 'POST',
      body: formData,
    });
  }

  getTrainLogsUrl(runId: string) {
    return `${API_URL}/models/train/${runId}/logs`;
  }

  async terminateTraining(runId: string) {
    return this.fetch(`/models/train/${runId}/terminate`, { method: 'POST' });
  }

  async getModelSummary() {
    return this.fetch('/models/summary');
  }

  getModelSummaryStreamUrl(runId?: string) {
    const base = `${API_URL}/models/summary/stream`;
    if (!runId) return base;
    return `${base}?runId=${encodeURIComponent(runId)}`;
  }

  // Predictions
  async predict(studentFile: File, creditHours?: number | null) {
    const formData = new FormData();
    formData.append('studentFile', studentFile);
    if (creditHours !== undefined && creditHours !== null && !Number.isNaN(creditHours)) {
      formData.append('creditHours', String(creditHours));
    }

    return this.fetch('/predict', {
      method: 'POST',
      body: formData,
    });
  }

  async getPredictions() {
    return this.fetch('/predict');
  }

  async getPrediction(id: string) {
    return this.fetch(`/predict/${id}`);
  }

  async clearPredictions() {
    return this.fetch('/predict', { method: 'DELETE' });
  }
}

export const api = new ApiClient();
