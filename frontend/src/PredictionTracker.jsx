import { useEffect, useState } from "react";

export default function PredictionTracker() {
  const [tracker, setTracker] = useState({
    rows: [],
    summary: {
      total: 0,
      pending: 0,
      verified: 0,
      correct: 0,
      incorrect: 0,
    },
  });

  const [loading, setLoading] = useState(true);

  const loadTracker = async () => {
    try {
      const response = await fetch("http://127.0.0.1:8000/api/tracker");

      if (!response.ok) {
        throw new Error("Unable to load prediction tracker");
      }

      const data = await response.json();
      setTracker(data);
    } catch (error) {
      console.error("Tracker error:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadTracker();
  }, []);

  const {
    total,
    pending,
    verified,
    correct,
    incorrect,
  } = tracker.summary;

  return (
    <section className="section prediction-tracker-section">

      <div className="section-title">

        <div>
          <span className="eyebrow">
            LIVE MODEL MONITORING
          </span>

          <h2>2026 Prediction Monitoring</h2>

          <p className="section-description">
            Daily predictions are compared with subsequent
            satellite-derived active-fire observations when available.
          </p>
        </div>

        <div className="evaluation-period">
          Live-history deployment
        </div>

      </div>

      <div className="tracker-summary-grid">

        <div className="tracker-card">
          <span>Predictions</span>
          <strong>{total}</strong>
        </div>

        <div className="tracker-card">
          <span>Pending</span>
          <strong>{pending}</strong>
        </div>

        <div className="tracker-card">
          <span>Verified</span>
          <strong>{verified}</strong>
        </div>

        <div className="tracker-card">
          <span>Correct</span>
          <strong>{correct}</strong>
        </div>

        <div className="tracker-card">
          <span>Incorrect</span>
          <strong>{incorrect}</strong>
        </div>

      </div>

      <div className="tracker-table-container">

        <table className="tracker-table">

          <thead>
            <tr>
              <th>Date</th>
              <th>District</th>
              <th>Prediction</th>
              <th>Probability</th>
              <th>Actual</th>
              <th>Result</th>
            </tr>
          </thead>

          <tbody>

            {loading && (
              <tr>
                <td colSpan="6" className="tracker-empty">
                  Loading prediction history...
                </td>
              </tr>
            )}

            {!loading && tracker.rows.length === 0 && (
              <tr>
                <td colSpan="6" className="tracker-empty">
                  No prediction records available.
                </td>
              </tr>
            )}

            {!loading &&
              tracker.rows.map((row, index) => {

                const probability =
                  row.risk_probability != null
                    ? `${(Number(row.risk_probability) * 100).toFixed(1)}%`
                    : "—";

                const actual =
                  row.actual_risk ||
                  row.actual_label ||
                  "Pending";

                const result =
                  row.verification_status || "Pending";

                return (
                  <tr
                    key={`${row.prediction_date}-${row.district}-${index}`}
                  >

                    <td>
                      {row.prediction_date || "—"}
                    </td>

                    <td>
                      <strong>
                        {row.district || "—"}
                      </strong>

                      <span className="tracker-state">
                        {row.state || ""}
                      </span>
                    </td>

                    <td>
                      <span
                        className={
                          row.risk_label === "Elevated"
                            ? "tracker-badge tracker-elevated"
                            : "tracker-badge tracker-normal"
                        }
                      >
                        {row.risk_label || "—"}
                      </span>
                    </td>

                    <td>
                      {probability}
                    </td>

                    <td>
                      {actual}
                    </td>

                    <td>
                      <span
                        className={
                          result === "Correct"
                            ? "tracker-result tracker-correct"
                            : result === "Incorrect"
                              ? "tracker-result tracker-incorrect"
                              : "tracker-result tracker-pending"
                        }
                      >
                        {result}
                      </span>
                    </td>

                  </tr>
                );
              })}

          </tbody>

        </table>

      </div>

      <div className="tracker-note">

        <strong>How verification works</strong>

        <p>
          Each next-day prediction is stored first as pending.
          After corresponding FIRMS observations become available,
          the observed district fire activity is compared with the
          predicted Normal/Elevated classification.
        </p>

        <p>
          2026 performance metrics are calculated only from verified
          observations and are not inferred from pending predictions.
        </p>

      </div>

    </section>
  );
}