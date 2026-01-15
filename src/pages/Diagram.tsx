import { useEffect } from 'react';

export default function Diagram() {
  useEffect(() => {
    document.title = 'Diagram';
  }, []);

  useEffect(() => {
    const handleAfterPrint = () => {
      document.body.dataset.printTarget = '';
    };
    window.addEventListener('afterprint', handleAfterPrint);
    return () => window.removeEventListener('afterprint', handleAfterPrint);
  }, []);

  const handleDownload = (target: 'presentation' | 'thesis' | 'pipeline') => {
    document.body.dataset.printTarget = target;
    window.print();
  };

  return (
    <div className="diagram-page">
      <style>{`
        :root {
          --diagram-bg: #f7f4ef;
          --diagram-ink: #1c1a17;
          --diagram-muted: #6d645a;
          --diagram-line: #9a8f82;
          --diagram-data: #4b6f8a;
          --diagram-features: #3b8c83;
          --diagram-targets: #6b5aa8;
          --diagram-model: #4a8b4a;
          --diagram-eval: #b07d29;
          --diagram-interpret: #2f6f7b;
          --diagram-risk: #9b3d2f;
          --diagram-artifacts: #5f4b3b;
          --diagram-serving: #4b4f8a;
          --diagram-monitor: #555b65;
        }

        .diagram-page {
          min-height: 100vh;
          background:
            radial-gradient(circle at top, rgba(255, 255, 255, 0.85) 0%, rgba(243, 239, 233, 0.9) 45%),
            linear-gradient(120deg, #f4efe7 0%, #efe8df 60%, #f6f1ea 100%);
          color: var(--diagram-ink);
          padding: 32px 20px 56px;
          font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
        }

        .diagram-wrap {
          max-width: 1000px;
          margin: 0 auto;
        }

        .diagram-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 16px;
          margin-bottom: 20px;
        }

        .diagram-header h1 {
          font-size: 28px;
          letter-spacing: 0.3px;
          margin: 0 0 6px 0;
        }

        .diagram-header p {
          margin: 0;
          color: var(--diagram-muted);
          font-size: 14px;
        }

        .diagram-btn {
          background: #1c1a17;
          color: #fffaf2;
          border: none;
          border-radius: 999px;
          padding: 10px 18px;
          font-size: 13px;
          letter-spacing: 0.2px;
          cursor: pointer;
          box-shadow: 0 6px 14px rgba(28, 26, 23, 0.18);
        }

        .diagram-btn:hover {
          transform: translateY(-1px);
        }

        .diagram-section {
          background: rgba(255, 255, 255, 0.78);
          border: 1px solid rgba(28, 26, 23, 0.12);
          border-radius: 20px;
          padding: 20px;
          box-shadow: 0 18px 40px rgba(27, 25, 22, 0.12);
          margin-bottom: 24px;
          position: relative;
          overflow: hidden;
        }

        .diagram-section::after {
          content: "";
          position: absolute;
          inset: -40% 60% auto -30%;
          height: 200px;
          background: radial-gradient(circle, rgba(28, 26, 23, 0.08), transparent 70%);
          pointer-events: none;
        }

        .diagram-section-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 12px;
          margin-bottom: 18px;
        }

        .diagram-section h2 {
          margin: 0;
          font-size: 20px;
          letter-spacing: 0.2px;
        }

        .diagram-legend {
          display: flex;
          flex-wrap: wrap;
          gap: 10px 16px;
          font-size: 12px;
          color: var(--diagram-muted);
          margin-bottom: 20px;
        }

        .legend-item {
          display: inline-flex;
          align-items: center;
          gap: 6px;
        }

        .legend-dot {
          width: 10px;
          height: 10px;
          border-radius: 999px;
          display: inline-block;
        }

        .flow {
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 12px;
        }

        .node {
          width: min(820px, 92vw);
          background: #fffdf9;
          border: 2px solid rgba(28, 26, 23, 0.18);
          border-left: 8px solid var(--diagram-line);
          border-radius: 16px;
          padding: 14px 18px;
          box-shadow: 0 10px 24px rgba(28, 26, 23, 0.12);
          position: relative;
        }

        .node-title {
          display: block;
          font-weight: 700;
          font-size: 16px;
        }

        .node-detail {
          display: block;
          margin-top: 6px;
          color: #2f2a24;
          font-size: 12px;
        }

        .node.data { border-left-color: var(--diagram-data); }
        .node.features { border-left-color: var(--diagram-features); }
        .node.targets { border-left-color: var(--diagram-targets); }
        .node.model { border-left-color: var(--diagram-model); }
        .node.eval { border-left-color: var(--diagram-eval); }
        .node.interpret { border-left-color: var(--diagram-interpret); }
        .node.risk { border-left-color: var(--diagram-risk); }
        .node.artifacts { border-left-color: var(--diagram-artifacts); }
        .node.serving { border-left-color: var(--diagram-serving); }
        .node.monitor { border-left-color: var(--diagram-monitor); }

        .arrow-down {
          width: 3px;
          height: 22px;
          background: var(--diagram-line);
        }

        .split {
          width: min(820px, 92vw);
          height: 24px;
          position: relative;
        }

        .split::before {
          content: "";
          position: absolute;
          top: 0;
          left: 50%;
          width: 3px;
          height: 12px;
          background: var(--diagram-line);
        }

        .split::after {
          content: "";
          position: absolute;
          top: 12px;
          left: 12%;
          right: 12%;
          height: 3px;
          background: var(--diagram-line);
        }

        .branch {
          width: min(820px, 92vw);
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 16px;
        }

        .branch .node {
          width: 100%;
        }

        .branch .node::before {
          content: "";
          position: absolute;
          top: -12px;
          left: 50%;
          width: 3px;
          height: 12px;
          background: var(--diagram-line);
        }

        .merge {
          width: min(820px, 92vw);
          height: 24px;
          position: relative;
        }

        .merge::before {
          content: "";
          position: absolute;
          top: 12px;
          left: 12%;
          right: 12%;
          height: 3px;
          background: var(--diagram-line);
        }

        .merge::after {
          content: "";
          position: absolute;
          top: 12px;
          left: 50%;
          width: 3px;
          height: 12px;
          background: var(--diagram-line);
        }

        .diagram-footer {
          margin-top: 24px;
          color: var(--diagram-muted);
          font-size: 12px;
          text-align: center;
        }

        .pipeline-grid {
          display: flex;
          flex-direction: column;
          gap: 20px;
          align-items: center;
        }

        .pipeline-panel {
          background: transparent;
          border: none;
          border-radius: 0;
          padding: 0;
          box-shadow: none;
          width: fit-content;
          justify-self: center;
        }

        .pipeline-title {
          font-size: 16px;
          font-weight: 700;
          margin: 0 0 12px 0;
          text-align: center;
        }

        .pipeline-flow {
          display: flex;
          flex-direction: column;
          gap: 8px;
          align-items: center;
        }

        .pipeline-step {
          border: 1px solid rgba(28, 26, 23, 0.12);
          border-radius: 14px;
          padding: 12px 14px;
          background: #f9f7f3;
          width: fit-content;
          min-width: 260px;
          text-align: center;
        }

        .pipeline-arrow {
          display: flex;
          align-items: center;
          justify-content: center;
          color: #1c1a17;
          font-weight: 700;
          font-size: 14px;
          height: 16px;
        }

        .pipeline-connector {
          display: flex;
          align-items: center;
          justify-content: center;
          height: 22px;
          font-size: 18px;
          color: #1c1a17;
        }

        .pipeline-step-title {
          display: block;
          font-weight: 600;
          font-size: 13px;
          margin-bottom: 6px;
          text-align: center;
        }

        .pipeline-step ul {
          margin: 0;
          padding-left: 0;
          font-size: 12px;
          color: #2f2a24;
          list-style: none;
          text-align: center;
        }

        .diagram-thesis {
          background: #ffffff;
        }

        .diagram-thesis .diagram-footer {
          color: #1c1a17;
        }

        .thesis-layout {
          position: relative;
          padding: 10px 12px 24px;
          min-height: 220px;
        }

        .thesis-row {
          display: grid;
          grid-template-columns: repeat(3, minmax(170px, 1fr));
          grid-template-rows: repeat(3, minmax(0, 1fr));
          grid-auto-flow: row;
          gap: 24px;
          overflow-x: auto;
          padding-bottom: 8px;
          position: relative;
        }

        .thesis-box {
          border: 2px solid #1c1a17;
          border-radius: 16px;
          padding: 14px 16px;
          min-height: 70px;
          min-width: 0;
          display: flex;
          align-items: center;
          justify-content: center;
          text-align: center;
          font-weight: 600;
          color: #1c1a17;
          background: #ffffff;
          font-size: 13px;
          position: relative;
          z-index: 1;
        }

        .thesis-row .thesis-box:nth-child(1)::after,
        .thesis-row .thesis-box:nth-child(2)::after,
        .thesis-row .thesis-box:nth-child(4)::after,
        .thesis-row .thesis-box:nth-child(5)::after,
        .thesis-row .thesis-box:nth-child(7)::after,
        .thesis-row .thesis-box:nth-child(8)::after {
          content: "";
          position: absolute;
          right: -22px;
          top: 50%;
          width: 18px;
          height: 2px;
          background: #1c1a17;
          transform: translateY(-50%);
        }

        .thesis-row .thesis-box:nth-child(1)::before,
        .thesis-row .thesis-box:nth-child(2)::before,
        .thesis-row .thesis-box:nth-child(4)::before,
        .thesis-row .thesis-box:nth-child(5)::before,
        .thesis-row .thesis-box:nth-child(7)::before,
        .thesis-row .thesis-box:nth-child(8)::before {
          content: "";
          position: absolute;
          right: -24px;
          top: 50%;
          transform: translateY(-50%);
          border-left: 6px solid #1c1a17;
          border-top: 4px solid transparent;
          border-bottom: 4px solid transparent;
        }

        .thesis-row .thesis-box:nth-child(4)::after,
        .thesis-row .thesis-box:nth-child(5)::after {
          right: auto;
          left: -22px;
        }

        .thesis-row .thesis-box:nth-child(4)::before,
        .thesis-row .thesis-box:nth-child(5)::before {
          right: auto;
          left: -24px;
          border-left: 0;
          border-right: 6px solid #1c1a17;
        }

        .thesis-row .thesis-box:nth-child(3)::after,
        .thesis-row .thesis-box:nth-child(6)::after {
          content: "";
          position: absolute;
          left: 50%;
          top: auto;
          bottom: -18px;
          width: 2px;
          height: 12px;
          background: #1c1a17;
          transform: translateX(-50%);
        }

        .thesis-row .thesis-box:nth-child(3)::before,
        .thesis-row .thesis-box:nth-child(6)::before {
          content: "";
          position: absolute;
          left: 50%;
          top: auto;
          bottom: -22px;
          transform: translateX(-50%);
          border-left: 4px solid transparent;
          border-right: 4px solid transparent;
          border-top: 6px solid #1c1a17;
          border-bottom: 0;
        }

        .thesis-row .thesis-box:nth-child(4) {
          grid-column: 3;
          grid-row: 2;
        }

        .thesis-row .thesis-box:nth-child(5) {
          grid-column: 2;
          grid-row: 2;
        }

        .thesis-row .thesis-box:nth-child(6) {
          grid-column: 1;
          grid-row: 2;
        }

        .thesis-row .thesis-box:nth-child(7) {
          grid-column: 1;
          grid-row: 3;
        }

        .thesis-row .thesis-box:nth-child(8) {
          grid-column: 2;
          grid-row: 3;
        }

        .thesis-row .thesis-box:nth-child(9) {
          grid-column: 3;
          grid-row: 3;
        }


        .thesis-input,
        .thesis-process,
        .thesis-output {
          background: #f1f1f1;
        }

        .thesis-caption {
          margin-top: 16px;
          text-align: center;
          font-size: 13px;
          color: #1c1a17;
          font-family: "Times New Roman", serif;
        }

        @media (max-width: 720px) {
          .diagram-header {
            flex-direction: column;
            align-items: flex-start;
          }

          .branch {
            grid-template-columns: 1fr;
          }

          .split::after,
          .merge::before {
            left: 20%;
            right: 20%;
          }

          .thesis-layout {
            min-height: auto;
          }

          .thesis-links {
            display: none;
          }
        }

        @media print {
          .diagram-page {
            background: #ffffff;
            padding: 10mm;
          }

          .diagram-btn {
            display: none;
          }

          body[data-print-target="presentation"] .diagram-thesis {
            display: none;
          }

          body[data-print-target="thesis"] .diagram-presentation {
            display: none;
          }

          body[data-print-target="pipeline"] .diagram-presentation,
          body[data-print-target="pipeline"] .diagram-thesis {
            display: none;
          }

          body[data-print-target="pipeline"] .diagram-section:not(:has(.pipeline-grid)) {
            display: none;
          }

          body[data-print-target="thesis"] .diagram-thesis * {
            visibility: hidden;
          }

          body[data-print-target="thesis"] .diagram-thesis .thesis-row,
          body[data-print-target="thesis"] .diagram-thesis .thesis-row * {
            visibility: visible;
          }

          body[data-print-target="thesis"] .diagram-thesis {
            padding: 0;
            box-shadow: none;
            border: none;
          }
        }
      `}</style>

      <div className="diagram-wrap">
        <div className="diagram-header">
          <div>
            <h1>Multi-Target Academic Performance Prediction</h1>
            <p>Flowchart of the data, modeling, and delivery pipeline.</p>
          </div>
        </div>

        <div className="diagram-section diagram-presentation">
          <div className="diagram-section-header">
            <h2>Presentation Diagram (High Contrast)</h2>
            <button className="diagram-btn" onClick={() => handleDownload('presentation')}>
              Download PDF
            </button>
          </div>

          <div className="diagram-legend">
            <span className="legend-item"><span className="legend-dot" style={{ background: 'var(--diagram-data)' }} /> Data</span>
            <span className="legend-item"><span className="legend-dot" style={{ background: 'var(--diagram-features)' }} /> Features</span>
            <span className="legend-item"><span className="legend-dot" style={{ background: 'var(--diagram-targets)' }} /> Targets</span>
            <span className="legend-item"><span className="legend-dot" style={{ background: 'var(--diagram-model)' }} /> Modeling</span>
            <span className="legend-item"><span className="legend-dot" style={{ background: 'var(--diagram-eval)' }} /> Evaluation</span>
            <span className="legend-item"><span className="legend-dot" style={{ background: 'var(--diagram-interpret)' }} /> Interpretation</span>
            <span className="legend-item"><span className="legend-dot" style={{ background: 'var(--diagram-risk)' }} /> Risk</span>
            <span className="legend-item"><span className="legend-dot" style={{ background: 'var(--diagram-artifacts)' }} /> Artifacts</span>
            <span className="legend-item"><span className="legend-dot" style={{ background: 'var(--diagram-serving)' }} /> Serving</span>
            <span className="legend-item"><span className="legend-dot" style={{ background: 'var(--diagram-monitor)' }} /> Ops</span>
          </div>

          <div className="flow">
            <div className="node data">
              <span className="node-title">1. Data Ingestion & Cleaning</span>
              <span className="node-detail">De-duplication, missing values, grade and credit clamping</span>
            </div>
            <div className="arrow-down" />

            <div className="node features">
              <span className="node-title">2. Feature Engineering</span>
              <span className="node-detail">Trends, volatility, attendance, credit hour aggregates</span>
            </div>
            <div className="arrow-down" />

            <div className="node targets">
              <span className="node-title">3. Targets Split</span>
              <span className="node-detail">Final CGPA and Next-Sem CGPA regression targets</span>
            </div>
            <div className="split" />

            <div className="branch">
              <div className="node targets">
                <span className="node-title">3A. Final CGPA Target</span>
                <span className="node-detail">All semesters included</span>
              </div>
              <div className="node targets">
                <span className="node-title">3B. Next-Sem CGPA Target</span>
                <span className="node-detail">Rolling prefix windows</span>
              </div>
            </div>

            <div className="merge" />

            <div className="node model">
              <span className="node-title">4. Model Suite (per target)</span>
              <span className="node-detail">DecisionTree, RandomForest, SVR, LightGBM, MLP</span>
            </div>
            <div className="arrow-down" />

            <div className="node model">
              <span className="node-title">5. Training & Progress Logs</span>
              <span className="node-detail">Step-level events via SSE log stream</span>
            </div>
            <div className="arrow-down" />

            <div className="node eval">
              <span className="node-title">6. Evaluation & Selection</span>
              <span className="node-detail">RMSE, MAE, R²; choose best model</span>
            </div>
            <div className="arrow-down" />

            <div className="node interpret">
              <span className="node-title">7. Interpretability</span>
              <span className="node-detail">Feature importance, learning curves, residual samples</span>
            </div>
            <div className="arrow-down" />

            <div className="node risk">
              <span className="node-title">8. Risk Classification</span>
              <span className="node-detail">Quantile thresholds to High/Medium/Low</span>
            </div>
            <div className="arrow-down" />

            <div className="node artifacts">
              <span className="node-title">9. Artifacts & Reports</span>
              <span className="node-detail">Models, plots, report.json, metadata.json</span>
            </div>
            <div className="arrow-down" />

            <div className="node serving">
              <span className="node-title">10. Serving & Auth</span>
              <span className="node-detail">API endpoints, JWT + refresh, summary stream</span>
            </div>
            <div className="arrow-down" />

            <div className="node monitor">
              <span className="node-title">Ops & Monitoring</span>
              <span className="node-detail">Resource logs, training status, termination hooks</span>
            </div>
          </div>

          <div className="diagram-footer">
            Group-wise split by student; dual regression targets.
          </div>
        </div>

        <div className="diagram-section diagram-thesis">
          <div className="diagram-section-header">
            <h2>Thesis Diagram (Conceptual Framework)</h2>
            <button className="diagram-btn" onClick={() => handleDownload('thesis')}>
              Download PDF
            </button>
          </div>

          <div className="thesis-layout">
            <div className="thesis-row">
              <div className="thesis-box thesis-input">1. Data ingestion & cleaning</div>
              <div className="thesis-box thesis-input">2. Feature engineering</div>
              <div className="thesis-box thesis-input">3. Targets split</div>
              <div className="thesis-box thesis-process">4. Model suite (per target)</div>
              <div className="thesis-box thesis-process">5. Training & progress logs</div>
              <div className="thesis-box thesis-process">6. Evaluation & selection</div>
              <div className="thesis-box thesis-process">7. Interpretability</div>
              <div className="thesis-box thesis-output">8. Risk classification</div>
              <div className="thesis-box thesis-output">9. Artifacts & reports</div>
            </div>
          </div>

          <div className="thesis-caption">
            Figure 1.1: Conceptual framework of the proposed multi-target academic performance prediction system.
          </div>
        </div>

        <div className="diagram-section">
          <div className="diagram-section-header">
            <h2>Pipeline Charts</h2>
            <button className="diagram-btn" onClick={() => handleDownload('pipeline')}>
              Download PDF
            </button>
          </div>

          <div className="pipeline-grid">
            <div className="pipeline-panel">
              <h3 className="pipeline-title">Dataforge Pipeline</h3>
              <div className="pipeline-flow">
                <div className="pipeline-step">
                  <span className="pipeline-step-title">JSON Params</span>
                  <ul>
                    <li>Config parameters</li>
                    <li>Data generation specs</li>
                  </ul>
                </div>
                <div className="pipeline-arrow">↓</div>
                <div className="pipeline-step">
                  <span className="pipeline-step-title">DataForge Generation</span>
                  <ul>
                    <li>Synthetic data creation</li>
                    <li>Student records generation</li>
                  </ul>
                </div>
                <div className="pipeline-arrow">↓</div>
                <div className="pipeline-step">
                  <span className="pipeline-step-title">Post-process & Persist</span>
                  <ul>
                    <li>Validation & formatting</li>
                    <li>Save to database/files</li>
                  </ul>
                </div>
              </div>
            </div>

            <div className="pipeline-connector">↓</div>

            <div className="pipeline-panel">
              <h3 className="pipeline-title">Empirical Prediction System Pipeline</h3>
              <div className="pipeline-flow">
                <div className="pipeline-step">
                  <span className="pipeline-step-title">Clean Dataset</span>
                  <ul>
                    <li>De-duplication</li>
                    <li>Missing values, clamping</li>
                  </ul>
                </div>
                <div className="pipeline-arrow">↓</div>
                <div className="pipeline-step">
                  <span className="pipeline-step-title">Feature Engineering</span>
                  <ul>
                    <li>Trends & volatility</li>
                    <li>Attendance, credit hours</li>
                  </ul>
                </div>
                <div className="pipeline-arrow">↓</div>
                <div className="pipeline-step">
                  <span className="pipeline-step-title">Target Creation</span>
                  <ul>
                    <li>Final CGPA target</li>
                    <li>Next-Sem CGPA target</li>
                  </ul>
                </div>
                <div className="pipeline-arrow">↓</div>
                <div className="pipeline-step">
                  <span className="pipeline-step-title">Student-wise Split</span>
                  <ul>
                    <li>Group-based split</li>
                    <li>Train/validation/test sets</li>
                  </ul>
                </div>
                <div className="pipeline-arrow">↓</div>
                <div className="pipeline-step">
                  <span className="pipeline-step-title">Train Model Suite</span>
                  <ul>
                    <li>DT, RF, SVR, LightGBM, MLP</li>
                    <li>Dual target training</li>
                  </ul>
                </div>
                <div className="pipeline-arrow">↓</div>
                <div className="pipeline-step">
                  <span className="pipeline-step-title">Evaluate & Select</span>
                  <ul>
                    <li>RMSE, MAE, R² metrics</li>
                    <li>Best model selection</li>
                  </ul>
                </div>
                <div className="pipeline-arrow">↓</div>
                <div className="pipeline-step">
                  <span className="pipeline-step-title">Interpretability</span>
                  <ul>
                    <li>Feature importance</li>
                    <li>Learning curves</li>
                  </ul>
                </div>
                <div className="pipeline-arrow">↓</div>
                <div className="pipeline-step">
                  <span className="pipeline-step-title">Risk Classification</span>
                  <ul>
                    <li>High/Medium/Low bins</li>
                    <li>Quantile thresholds</li>
                  </ul>
                </div>
                <div className="pipeline-arrow">↓</div>
                <div className="pipeline-step">
                  <span className="pipeline-step-title">Save Artifacts & Reports</span>
                  <ul>
                    <li>Model artifacts, plots</li>
                    <li>report.json, metadata.json</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
