export default function ModelPerformance() {
  return (
    <section className="section model-performance-section">

      <div className="section-title">
        <div>
          <span className="eyebrow">MODEL VALIDATION</span>
          <h2>Model Performance</h2>
          <p className="section-description">
            Final Random Forest evaluation on the unseen 2025 test set.
          </p>
        </div>

        <div className="evaluation-period">
          2023 Train → 2024 Validate → 2025 Test
        </div>
      </div>

      <div className="metric-grid">

        <div className="metric-card">
          <span>Accuracy</span>
          <strong>81.48%</strong>
        </div>

        <div className="metric-card">
          <span>Balanced Accuracy</span>
          <strong>79.35%</strong>
        </div>

        <div className="metric-card">
          <span>Precision</span>
          <strong>66.78%</strong>
        </div>

        <div className="metric-card">
          <span>Recall</span>
          <strong>74.15%</strong>
        </div>

        <div className="metric-card">
          <span>F1 Score</span>
          <strong>70.27%</strong>
        </div>

        <div className="metric-card">
          <span>ROC-AUC</span>
          <strong>87.79%</strong>
        </div>

        <div className="metric-card">
          <span>PR-AUC</span>
          <strong>79.26%</strong>
        </div>

        <div className="metric-card">
          <span>Test Samples</span>
          <strong>2,700</strong>
        </div>

      </div>

      <div className="evaluation-grid">

        <div className="evaluation-card">
          <div className="evaluation-card-header">
            <h3>ROC Curve</h3>
            <span>ROC-AUC 87.79%</span>
          </div>

          <img
            src="/evaluation/roc_curve.png"
            alt="Random Forest ROC curve on 2025 unseen test set"
          />
        </div>

        <div className="evaluation-card">
          <div className="evaluation-card-header">
            <h3>Precision–Recall Curve</h3>
            <span>PR-AUC 79.26%</span>
          </div>

          <img
            src="/evaluation/pr_curve.png"
            alt="Random Forest precision recall curve on 2025 unseen test set"
          />
        </div>

      </div>

      <div className="evaluation-grid">

        <div className="evaluation-card">
          <div className="evaluation-card-header">
            <h3>Confusion Matrix</h3>
            <span>2025 unseen test set</span>
          </div>

          <img
            src="/evaluation/confusion_matrix.png"
            alt="Random Forest confusion matrix"
          />
        </div>

        <div className="evaluation-card">
          <div className="evaluation-card-header">
            <h3>Feature Importance</h3>
            <span>Random Forest</span>
          </div>

          <img
            src="/evaluation/feature_importance.png"
            alt="Random Forest feature importance"
          />
        </div>

      </div>

      <div className="evaluation-note">

        <strong>Evaluation design</strong>

        <p>
          The model was trained on 2023 data, the decision threshold
          was selected using the 2024 validation set, and final
          performance was measured on the unseen 2025 test set.
        </p>

        <p>
          The 2026 dashboard operates as a live deployment layer;
          its future accuracy is not claimed until corresponding
          observations become available for verification.
        </p>

      </div>

    </section>
  );
}