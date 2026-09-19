
import { useEffect, useMemo, useState } from "react";
import axios from "axios";
import "./App.css";
import RiskMap from "./RiskMap";
import ModelPerformance from "./ModelPerformance";
import PredictionTracker from "./PredictionTracker";
const API_URL =
  import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";
const TRIGGER_KEY =
  import.meta.env.VITE_PREDICTION_TRIGGER_KEY;

const processingStages = [
  "Connecting to live data",
  "Building fire-history features",
  "Processing weather inputs",
  "Running Random Forest inference",
  "Generating district risk predictions",
];

function getHighestRisk(predictions) {
  if (!Array.isArray(predictions) || predictions.length === 0) {
    return null;
  }

  return predictions.reduce(
    (max, item) =>
      Number(item.risk_probability) >
        Number(max.risk_probability)
        ? item
        : max,
    predictions[0]
  );
}


function App() {
  const [summary, setSummary] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [selectedDistrict, setSelectedDistrict] = useState(null);
  const [search, setSearch] = useState("");
  const [loading, setLoading] = useState(true);
  const [runningPrediction, setRunningPrediction] = useState(false);
  const [processingStage, setProcessingStage] = useState(0);
  const [error, setError] = useState("");
  const [showMap, setShowMap] = useState(false);
  const [predictionDates, setPredictionDates] = useState([]);
  const [selectedDate, setSelectedDate] = useState("");
  const [loadingDate, setLoadingDate] = useState(false);
  async function loadDashboard() {
    try {
    setError("");

    const [summaryResponse, predictionsResponse, datesResponse] =
      await Promise.all([
        axios.get(`${API_URL}/api/summary`),
        axios.get(`${API_URL}/api/predictions`),
        axios.get(`${API_URL}/api/prediction-dates`),
      ]);

    setSummary(summaryResponse.data);

    const predictionData =
      predictionsResponse.data?.predictions;

    if (!Array.isArray(predictionData)) {
      throw new Error(
        "Invalid prediction data received from backend."
      );
    }

    const dates = datesResponse.data?.dates;

    if (!Array.isArray(dates)) {
      throw new Error(
        "Invalid prediction dates received from backend."
      );
    }

    setPredictions(predictionData);
    setPredictionDates(dates);

    if (predictionsResponse.data?.prediction_date) {
      setSelectedDate(
        predictionsResponse.data.prediction_date
      );
    }

    const highest = getHighestRisk(predictionData);
    setSelectedDistrict(highest);

  } catch (err) {
    console.error(err);

    setError(
      "Unable to connect to StubbleAI. Make sure the FastAPI backend is running."
    );
  } finally {
    setLoading(false);
  }
}
async function loadPredictionDate(date) {
  if (!date) {
    return;
  }

  try {
    setLoadingDate(true);
    setError("");

    const response = await axios.get(
      `${API_URL}/api/predictions`,
      {
        params: {
          date,
        },
      }
    );

    const predictionData =
      response.data?.predictions;

    if (!Array.isArray(predictionData)) {
      throw new Error(
        "Invalid historical prediction data received."
      );
    }

    if (predictionData.length !== 45) {
      throw new Error(
        `Expected 45 district predictions, received ${predictionData.length}.`
      );
    }

    setPredictions(predictionData);

    const highest = getHighestRisk(predictionData);

    setSelectedDistrict(highest);

    const normalCount = predictionData.filter(
      (item) => item.risk_label === "Normal"
    ).length;

    const elevatedCount = predictionData.filter(
      (item) => item.risk_label === "Elevated"
    ).length;

    setSummary({
      prediction_date:
        response.data.prediction_date,
      total_districts:
        response.data.total_districts,
      normal_count: normalCount,
      elevated_count: elevatedCount,
      highest_risk_district:
        highest?.district || null,
      highest_risk_state:
        highest?.state || null,
      highest_risk_probability:
        highest?.risk_probability || 0,
      highest_risk_label:
        highest?.risk_label || null,
    });

    setSelectedDate(date);
    setSearch("");
    setShowMap(false);

  } catch (err) {
    console.error(err);

    const message =
      err.response?.data?.detail ||
      "Unable to load the selected prediction date.";

    setError(String(message));

  } finally {
    setLoadingDate(false);
  }
}
useEffect(() => {
  loadDashboard();
}, []);

  // Cycle through the processing stages while the REAL backend request runs.
  useEffect(() => {
    if (!runningPrediction) {
      setProcessingStage(0);
      return;
    }

    const interval = setInterval(() => {
      setProcessingStage((current) => {
        return (current + 1) % processingStages.length;
      });
    }, 1800);

    return () => clearInterval(interval);
  }, [runningPrediction]);

  async function runLivePrediction() {
    try {
      setRunningPrediction(true);
      setError("");
      const predictionEndpoint = import.meta.env.PROD
  ? "/api/run-prediction"
  : `${API_URL}/api/run-prediction`;

const requestConfig = import.meta.env.PROD
  ? {
      timeout: 180000,
    }
  : {
      timeout: 180000,
      headers: {
        "X-Trigger-Key": import.meta.env.VITE_PREDICTION_TRIGGER_KEY,
      },
    };

const response = await axios.post(
  predictionEndpoint,
  {},
  requestConfig
);

//       const response = await axios.post(
//   `${API_URL}/api/run-prediction`,
//   {},
//   {
//     timeout: 180000,
//     headers: {
//       "X-Trigger-Key": TRIGGER_KEY,
//     },
//   }
// );

      const data = response.data;
      if (!Array.isArray(data.predictions)) {
        throw new Error(
          "Invalid prediction response received from backend."
        );
      }

      const highest = getHighestRisk(data.predictions);

      if (!highest) {
        throw new Error(
          "Backend returned no district predictions."
        );
      }

      setSummary({
        prediction_date: data.prediction_date,
        total_districts: data.total_districts,
        normal_count: data.normal_count,
        elevated_count: data.elevated_count,
        highest_risk_district: highest.district,
        highest_risk_state: highest.state,
        highest_risk_probability: highest.risk_probability,
        highest_risk_label: highest.risk_label,
      });

      setPredictions(data.predictions);
      setSelectedDistrict(highest);
      setSelectedDate(data.prediction_date);

setPredictionDates((currentDates) => {
  const updatedDates = [
    ...new Set([
      data.prediction_date,
      ...currentDates,
    ]),
  ];

  return updatedDates.sort().reverse();
});


    } catch (err) {
      console.error(err);

      const message =
        err.response?.data?.detail?.message ||
        err.response?.data?.detail ||
        "Live prediction failed. Check the FastAPI terminal.";

      setError(String(message));
    } finally {
      setRunningPrediction(false);
    }
  }

  const filteredPredictions = useMemo(() => {
    const query = search.trim().toLowerCase();

    if (!query) {
      return predictions;
    }

    return predictions.filter(
      (item) =>
        item.district.toLowerCase().includes(query) ||
        item.state.toLowerCase().includes(query)
    );
  }, [predictions, search]);

  function selectDistrict(district) {
    setSelectedDistrict(district);
  }

  if (loading) {
    return (
      <div className="initial-loader">
        <div className="loader-symbol">🌾</div>
        <div className="loader-ring"></div>
        <h2>StubbleAI</h2>
        <p>Loading risk intelligence...</p>
      </div>
    );
  }

  if (error && !summary) {
    return (
      <div className="error-screen">
        <div className="error-box">
          <div className="error-icon">!</div>
          <h2>StubbleAI could not connect</h2>
          <p>{error}</p>
          <button onClick={loadDashboard}>Retry</button>
        </div>
      </div>
    );
  }

  return (
    <div className="app">
      {/* ------------------------------------------------ */}
      {/* AI PROCESSING OVERLAY */}
      {/* ------------------------------------------------ */}

      {runningPrediction && (
        <div className="processing-overlay">
          <div className="processing-card">
            <div className="processing-orbit">
              <div className="orbit-ring orbit-one"></div>
              <div className="orbit-ring orbit-two"></div>
              <div className="orbit-core"></div>
            </div>

            {/* <div className="processing-label">STUBBLEAI</div> */}
            <div className="processing-label">STUBBLEAI ENGINE</div>

            <h2>AI Risk Analysis</h2>

            <p className="processing-description">
              {/* Processing live district-level fire risk */}
              StubbleAI is analyzing live district-level conditions
              for next-day elevated fire activity

            </p>

            <div className="processing-stage">
              <span className="stage-dot"></span>
              {processingStages[processingStage]}
              <span className="animated-dots">...</span>
            </div>

            <div className="processing-bar">
              <div
                className="processing-progress"
                style={{
                  width: `${((processingStage + 1) / processingStages.length) * 100
                    }%`,
                }}
              ></div>
            </div>

            <div className="processing-meta">
              <span>LIVE INFERENCE</span>
              <span>45 DISTRICTS</span>
            </div>
          </div>
        </div>
      )}

      {/* ------------------------------------------------ */}
      {/* HEADER */}
      {/* ------------------------------------------------ */}

      {/* ------------------------------------------------ */}
      {/* STUBBLEAI HEADER */}
      {/* ------------------------------------------------ */}

      <header className="header">

        {/* BRAND */}
        <div className="brand">

          <div className="brand-icon">
            <svg
              viewBox="0 0 64 64"
              className="stubble-logo"
              xmlns="http://www.w3.org/2000/svg"
            >
              {/* AI signal */}
              <circle cx="45" cy="18" r="2.8" />
              <circle cx="52" cy="29" r="2.8" />
              <circle cx="46" cy="41" r="2.8" />

              <path d="M45 18L52 29L46 41" />

              {/* Main crop stalk */}
              <path className="stalk-main" d="M25 51V16" />

              {/* Crop leaves */}
              <path className="stalk-left" d="M25 27C19 24 16 20 16 15" />
              <path className="stalk-right" d="M25 34C31 31 34 26 34 20" />
              <path className="stalk-left" d="M25 40C19 37 16 33 16 29" />
              <path className="stalk-right" d="M25 46C31 43 34 39 34 34" />

              {/* Ground */}
              <path className="ground" d="M14 52H49" />
            </svg>
          </div>

          <div className="brand-text">
            <h1>
              Stubble<span>AI</span>
            </h1>

            <p>
              AI-Powered Crop Residue Risk Intelligence
            </p>
          </div>

        </div>


        {/* HEADER ACTIONS */}
        <div className="header-actions">

          <div className="date">
  <span>
    {selectedDate === predictionDates[0]
      ? "LATEST PREDICTION"
      : "HISTORICAL PREDICTION"}
  </span>

  <select
    value={selectedDate}
    onChange={(event) =>
      loadPredictionDate(event.target.value)
    }
    disabled={loadingDate || predictionDates.length === 0}
    aria-label="Select prediction date"
  >
    {predictionDates.map((date) => (
      <option
        key={date}
        value={date}
      >
        {date}
      </option>
    ))}
  </select>
</div>

          <button
            className="run-button"
            onClick={runLivePrediction}
            disabled={runningPrediction}
          >
            <span className="run-icon">
              {runningPrediction ? "◌" : "✦"}
            </span>

            {runningPrediction
              ? "Running Analysis..."
              : "Run Live Prediction"}
          </button>

        </div>

      </header>

      <main className="dashboard">
        {error && (
          <div className="inline-error">
            <span>!</span>
            {error}
          </div>
        )}

        {/* ------------------------------------------------ */}
        {/* SUMMARY CARDS */}
        {/* ------------------------------------------------ */}

        <section className="cards">
          <div className="card">
            <span>Total Districts</span>
            <strong>{summary.total_districts}</strong>
            <small>Punjab + Haryana</small>
          </div>

          <div className="card normal">
            <span>Normal Risk</span>
            <strong>{summary.normal_count}</strong>
            <small>Districts</small>
          </div>

          <div className="card elevated">
            <span>Elevated Risk</span>
            <strong>{summary.elevated_count}</strong>
            <small>Districts</small>
          </div>

          <div className="card highest">
            <span>Highest Risk</span>
            <strong>{summary.highest_risk_district}</strong>
            <small>
              {(summary.highest_risk_probability * 100).toFixed(1)}% probability
            </small>
          </div>
        </section>

        {/* ------------------------------------------------ */}
        {/* DISTRICT TABLE */}
        {/* ------------------------------------------------ */}



        <section className="section">

          <div className="section-title">

            <div>
              <span className="eyebrow">DISTRICT MONITORING</span>
              <h2>Risk Overview</h2>
            </div>

            <div className="district-header-actions">

              <button
                className="map-toggle-button"
                type="button"
                onClick={() => setShowMap(!showMap)}
                title={showMap ? "Show district table" : "Show risk map"}
                aria-label={showMap ? "Show district table" : "Show risk map"}
              >
                <span>{showMap ? "▦" : "◎"}</span>
              </button>

              <span className="district-count">
                {filteredPredictions.length} districts
              </span>

            </div>

          </div>



          {showMap ? (

            <div className="inline-map-view">

              <div className="inline-map-header">

                <div>
                  <span className="eyebrow">
                    GEOSPATIAL RISK INTELLIGENCE
                  </span>

                  <h3>District Risk Map</h3>

                  <p>
                    Live crop-residue fire risk across monitored districts
                  </p>
                </div>

                <button
                  className="map-close-button"
                  type="button"
                  onClick={() => setShowMap(false)}
                >
                  ▦ Table View
                </button>

              </div>

              <RiskMap
                predictions={predictions}
                onSelectDistrict={(district) => {
                  setSelectedDistrict(district);
                  setShowMap(false);
                }}
                onClose={() => setShowMap(false)}
                inline={true}
              />

            </div>

          ) : (

            <>

              <div className="toolbar">

                <div className="search-box">

                  <span>⌕</span>

                  <input
                    type="text"
                    placeholder="Search district or state..."
                    value={search}
                    onChange={(event) => setSearch(event.target.value)}
                  />

                  {search && (
                    <button
                      className="clear-search"
                      onClick={() => setSearch("")}
                    >
                      ×
                    </button>
                  )}

                </div>

              </div>

              <div className="table-container">

                <table>

                  <thead>
                    <tr>
                      <th>District</th>
                      <th>State</th>
                      <th>Risk Probability</th>
                      <th>Status</th>
                      <th>Mode</th>
                    </tr>
                  </thead>

                  <tbody>

                    {filteredPredictions.map((prediction) => (

                      <tr
                        key={`${prediction.state}-${prediction.district}`}
                        className={
                          selectedDistrict?.district === prediction.district &&
                            selectedDistrict?.state === prediction.state
                            ? "selected-row"
                            : ""
                        }
                        onClick={() => selectDistrict(prediction)}
                      >

                        <td className="district-name">
                          {prediction.district}
                        </td>

                        <td>
                          {prediction.state}
                        </td>

                        <td>

                          <div className="probability-cell">

                            <div className="probability-bar">
                              <div
                                style={{
                                  width: `${prediction.risk_probability * 100}%`,
                                }}
                              ></div>
                            </div>

                            <span>
                              {(prediction.risk_probability * 100).toFixed(1)}%
                            </span>

                          </div>

                        </td>

                        <td>

                          <span
                            className={`badge ${prediction.risk_label === "Elevated"
                                ? "badge-elevated"
                                : "badge-normal"
                              }`}
                          >

                            <span className="badge-dot"></span>

                            {prediction.risk_label}

                          </span>

                        </td>

                        <td>

                          <span className="mode-label">
                            {prediction.prediction_mode}
                          </span>

                        </td>

                      </tr>

                    ))}

                    {filteredPredictions.length === 0 && (

                      <tr>
                        <td colSpan="5" className="empty-state">
                          No district found for "{search}"
                        </td>
                      </tr>

                    )}

                  </tbody>

                </table>

              </div>

            </>

          )}
        </section>
        {/* ------------------------------------------------ */}
        {/* DISTRICT DETAIL */}
        {/* ------------------------------------------------ */}

        {selectedDistrict && (
          <section className="section district-detail-section">
            <div className="section-title">
              <div>
                <span className="eyebrow">DISTRICT ANALYSIS</span>
                <h2>
                  {selectedDistrict.district},{" "}
                  {selectedDistrict.state}
                </h2>
              </div>

              <span
                className={`large-status ${selectedDistrict.risk_label === "Elevated"
                    ? "large-status-elevated"
                    : "large-status-normal"
                  }`}
              >
                {selectedDistrict.risk_label}
              </span>
            </div>

            <div className="detail-grid">
              <div className="risk-panel">
                <span>Predicted next-day risk</span>

                <div className="big-probability">
                  {(selectedDistrict.risk_probability * 100).toFixed(1)}
                  <small>%</small>
                </div>

                <div className="risk-meter">
                  <div
                    style={{
                      width: `${selectedDistrict.risk_probability * 100}%`,
                    }}
                  ></div>
                </div>

                <p>
                  Model output probability for elevated active-fire activity.
                </p>

                <div className="prediction-mode">
                  <span>●</span>
                  {selectedDistrict.prediction_mode}
                </div>
              </div>

              <div className="detail-panel">
                <h3>Weather inputs</h3>

                <div className="metric-grid">
                  <div>
                    <span>Temperature</span>
                    <strong>{selectedDistrict.T2M}°C</strong>
                  </div>

                  <div>
                    <span>Humidity</span>
                    <strong>{selectedDistrict.RH2M}%</strong>
                  </div>

                  <div>
                    <span>Wind speed</span>
                    <strong>{selectedDistrict.WS2M} m/s</strong>
                  </div>

                  <div>
                    <span>Rainfall</span>
                    <strong>{selectedDistrict.PRECTOTCORR} mm</strong>
                  </div>
                </div>
              </div>

              <div className="detail-panel">
                <h3>Recent fire activity</h3>

                <div className="metric-grid">
                  <div>
                    <span>1-day lag</span>
                    <strong>{selectedDistrict.fire_lag_1d}</strong>
                  </div>

                  <div>
                    <span>3-day lag</span>
                    <strong>{selectedDistrict.fire_lag_3d}</strong>
                  </div>

                  <div>
                    <span>7-day lag</span>
                    <strong>{selectedDistrict.fire_lag_7d}</strong>
                  </div>

                  <div>
                    <span>3-day mean</span>
                    <strong>
                      {Number(selectedDistrict.fire_mean_3d).toFixed(2)}
                    </strong>
                  </div>

                  <div>
                    <span>7-day mean</span>
                    <strong>
                      {Number(selectedDistrict.fire_mean_7d).toFixed(2)}
                    </strong>
                  </div>
                </div>
              </div>

              <div className="detail-panel explanation-panel">
                <h3>About this prediction</h3>

                <p>
                  StubbleAI combines recent satellite-derived active-fire
                  activity, weather conditions, and seasonal information to
                  estimate next-day elevated fire activity at district level.
                </p>

                <small>
                  This is a decision-support prediction. FIRMS active-fire
                  detections are not themselves confirmation of crop-residue
                  burning.
                </small>
              </div>
            </div>
          </section>
        )}
        {/* ------------------------------------------------ */}
        {/* MODEL PERFORMANCE */}
        {/* ------------------------------------------------ */}

        <ModelPerformance />
        <PredictionTracker />
      </main>

      {/* {showMap && (
        <RiskMap
          predictions={predictions}
          onSelectDistrict={(district) => {
            setSelectedDistrict(district);
            setShowMap(false);
          }}
          onClose={() => setShowMap(false)}
        />
      )} */}


      <footer>
        <p>
          StubbleAI is a district-level decision-support prototype using
          satellite-derived active-fire activity and weather information.
        </p>

        <span>AI-assisted early warning • 2026 operational prototype</span>
      </footer>
    </div>
  );
}

export default App;