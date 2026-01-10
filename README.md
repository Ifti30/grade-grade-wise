# Empirical Garde Prediction - A systematic study on features influencing grade forecasting using multiple basic ML algorithms

A modern fullstack web application that orchestrates Python ML scripts for grade prediction with real-time training logs, predictions, and comprehensive dashboard.

## 🚀 Quick Start

### Prerequisites

- Node.js 18+ and npm
- PostgreSQL 15+ running locally (or reachable connection string)
- Python 3.8+ with required scientific libraries (`numpy`, `pandas`, `scikit-learn`, `lightgbm`, `torch`, etc.)
- Your `ml/train.py` and `ml/predict.py` scripts

### 1. Database Setup

Choose one of the two options below. The `DATABASE_URL` in `server/.env` must match whichever option you use; otherwise you will see errors like `User postgres was denied access on the database grades.public`.

**Option A: Use the bundled Docker database (recommended for a quick start)**

```bash
docker compose up -d db
```

This starts Postgres with:

- database: `grades`
- user: `postgres`
- password: `postgres`

Your `server/.env` should contain:

```
DATABASE_URL="postgresql://postgres:postgres@localhost:5432/grades?schema=public"
JWT_SECRET="replace-with-long-random-string"
PYTHON_BIN="python3" # override if needed
```

**Option B: Use an existing/local Postgres instance**

Create a database and user you control, then grant privileges:

```bash
psql -U postgres -c "CREATE DATABASE grades;"
psql -U postgres -c "CREATE USER grade_app WITH PASSWORD \'change-me\';"
psql -U postgres -c "GRANT ALL PRIVILEGES ON DATABASE grades TO grade_app;"
```

Then point `server/.env` to that user:

```
DATABASE_URL="postgresql://grade_app:change-me@localhost:5432/grades?schema=public"
JWT_SECRET="replace-with-long-random-string"
PYTHON_BIN="python3"
```

### 2. Backend Setup

```bash
cd server
cp .env.example .env
npm install
npx prisma migrate dev --name init
npm run dev
```

Backend runs on http://localhost:3000

**Important**: Update `PYTHON_BIN` in `server/.env` if your Python path is different (e.g., `/opt/homebrew/bin/python3` on M1 Macs).

### 3. Frontend Setup

In a new terminal:

```bash
npm install
npm run dev
```

Frontend runs on http://localhost:5173

### 4. Python Scripts

Place your ML scripts at the repo root:
```
./ml/train.py
./ml/predict.py
./ml/samples/train_small.json
```

Ensure they follow the contract specified in the spec:

**train.py** - Trains model and outputs:
```bash
python3 ml/train.py \
  --org-id <ORG_ID> \
  --train-json <PATH> \
  --config-json <PATH> \
  --out-dir <DIR>
```

Must print on completion:
```
__RESULT__{"status":"ok","metrics":{...}, "plots": {...}}
```

**predict.py** - Makes prediction:
```bash
python3 ml/predict.py \
  --org-id <ORG_ID> \
  --student-json <PATH> \
  --artifacts-dir <DIR> \
  --out-file <PATH>
```

Must print on completion:
```
__RESULT__{"status":"ok","prediction":{...}}
```

## 🏗️ Architecture

### Tech Stack

- **Backend**: Node.js + Express + Prisma + PostgreSQL
- **Frontend**: React + Vite + TailwindCSS
- **Database**: PostgreSQL with JSONB support
- **ML Integration**: Python subprocess with SSE streaming
- **Auth**: JWT with bcrypt

### Project Structure

```
.
├── server/              # Backend API
│   ├── src/
│   │   ├── index.js           # Express server
│   │   ├── auth.js            # JWT middleware
│   │   ├── routes/
│   │   │   ├── auth.js        # Auth endpoints
│   │   │   ├── models.js      # Training endpoints
│   │   │   └── predict.js     # Prediction endpoints
│   │   └── utils/
│   │       ├── validate-config.js
│   │       └── python-runner.js
│   ├── prisma/
│   │   └── schema.prisma
│   └── storage/         # File uploads & model artifacts
│       ├── uploads/
│       ├── models/
│       └── predictions/
├── src/                 # Frontend React app
│   ├── pages/          # Route pages
│   ├── components/     # Reusable components
│   └── lib/            # API client
├── ml/                 # Your Python scripts
│   ├── train.py
│   ├── predict.py
│   └── samples/
└── docker-compose.yml  # Optional helper (legacy); not required if PostgreSQL is already available
```

## 🔑 Features

### Authentication
- Signup with organization creation
- JWT-based authentication
- Protected routes

### Training Pipeline
- Upload training dataset (JSON)
- Configure hyperparameters with validation
- Custom grade scale editor
- Real-time SSE log streaming
- Metrics and plot visualization

### Prediction System
- Upload student data for prediction
- View prediction results with visualizations
- Prediction history with detailed views
- Retrain capability

### Dashboard
- Training summary with metrics
- Prediction interface
- History view
- Retrain workflow

## 📡 API Endpoints

### Auth
- `POST /api/auth/signup` - Create account
- `POST /api/auth/signin` - Login
- `GET /api/auth/me` - Get current user

### Models
- `POST /api/models/train` - Start training
- `GET /api/models/train/:runId/logs` - Stream logs (SSE)
- `GET /api/models/summary` - Get latest model
- `GET /api/models/status` - Check model status

### Predictions
- `POST /api/predict` - Make prediction
- `GET /api/predict` - List predictions
- `GET /api/predict/:id` - Get prediction details

### Static Files
- `GET /static/*` - Access artifacts, plots, logs

## ⚙️ Configuration

### Training Config Defaults

```json
{
  "RANDOM_SEED": 42,
  "TEST_SIZE": 0.2,
  "THREADS": 4,
  "RF_TREES": 300,
  "LGBM_N_ESTIMATORS": 2000,
  "MLP_HIDDEN": 64,
  "MLP_EPOCHS": 300,
  "MLP_PATIENCE": 40,
  "SVR_ENABLE": true,
  "RISK_HIGH_MAX": 3.30,
  "RISK_MED_MAX": 3.50,
  "GRADE_POINTS": {
    "A+": 4.0,
    "A": 3.75,
    "A-": 3.5,
    "B+": 3.25,
    "B": 3.0,
    "B-": 2.75,
    "C+": 2.5,
    "C": 2.25,
    "D": 2.0,
    "F": 0.0
  }
}
```

### Validation Bounds
- `TEST_SIZE`: 0.1 - 0.3
- `THREADS`: 1 - 16
- `RF_TREES`: 50 - 1000
- `LGBM_N_ESTIMATORS`: 200 - 4000
- `MLP_HIDDEN`: 16 - 256
- `MLP_EPOCHS`: 50 - 600
- `MLP_PATIENCE`: 10 - 100

## 🐳 Database Management

Access Adminer at http://localhost:8080

**Credentials**:
- System: PostgreSQL
- Server: db
- Username: postgres
- Password: postgres
- Database: grades

## 🛠️ Development

### Backend Development
```bash
cd server
npm run dev  # Runs with nodemon for auto-reload
```

### Frontend Development
```bash
npm run dev  # Runs Vite dev server
```

### Database Migrations
```bash
cd server
npx prisma migrate dev --name <migration_name>
npx prisma studio  # Visual database editor
```

## 📦 Storage Structure

```
server/storage/
├── uploads/<orgId>/
│   └── train_<timestamp>.json
├── models/<orgId>/<runId>/
│   ├── artifacts/
│   ├── train.log
│   ├── config.json
│   └── plots/
└── predictions/<orgId>/
    ├── student_<timestamp>.json
    └── prediction_<timestamp>.json
```

## 🔒 Security Notes

- JWT tokens stored in localStorage (frontend)
- All API routes except auth require JWT
- File uploads validated server-side
- Config parameters clamped to safe bounds
- CORS enabled for local development

## 🐛 Troubleshooting

### Python not found
Update `PYTHON_BIN` in `server/.env`:
```
PYTHON_BIN=/usr/bin/python3
# or
PYTHON_BIN=/opt/homebrew/bin/python3
```

### Database connection failed
Confirm PostgreSQL is reachable:
```bash
pg_isready --dbname grade_predictor --host localhost --port 5432
# or inspect the service logs
journalctl -u postgresql --since "5 minutes ago"
```
Also re-check the `DATABASE_URL` in `server/.env`.

### Training logs not streaming
- Check SSE headers are correct
- Verify train.log exists in runDir
- Check browser DevTools Network tab

### Prediction fails
- Ensure a SUCCEEDED model exists
- Verify artifacts directory is accessible
- Check Python script outputs __RESULT__ line

## 📝 License

MIT
