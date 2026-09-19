import { useEffect, useState } from "react";
import {
  MapContainer,
  TileLayer,
  GeoJSON,
  useMap,
} from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";

const INDIA_CENTER = [29.0, 77.5];

function MapController({ geojson }) {
  const map = useMap();

  useEffect(() => {
    if (!geojson) return;

    const layer = L.geoJSON(geojson);

    if (layer.getBounds().isValid()) {
      map.fitBounds(layer.getBounds(), {
        padding: [25, 25],
      });
    }
  }, [geojson, map]);

  return null;
}

function getRiskColor(probability) {
  if (probability >= 0.60) return "#c62828";
  if (probability >= 0.40) return "#e67e22";
  if (probability >= 0.20) return "#e5b93f";
  return "#3f8f63";
}

export default function RiskMap({
  predictions,
  onSelectDistrict,
  onClose,
  inline = false,
}) {
  const [geojson, setGeojson] = useState(null);
  const [selectedRisk, setSelectedRisk] = useState("All");

  useEffect(() => {
    fetch("/districts_punjab_haryana.geojson")
      .then((response) => {
        if (!response.ok) {
          throw new Error("Unable to load district map");
        }

        return response.json();
      })
      .then(setGeojson)
      .catch((error) => {
        console.error(error);
      });
  }, []);

  const predictionMap = new Map(
    predictions.map((item) => [
      `${item.state}|${item.district}`.toLowerCase(),
      item,
    ])
  );

  const filteredPredictions = predictions.filter((item) => {
    if (selectedRisk === "All") return true;

    return item.risk_label === selectedRisk;
  });

  const visibleKeys = new Set(
    filteredPredictions.map(
      (item) =>
        `${item.state}|${item.district}`.toLowerCase()
    )
  );

  const styleFeature = (feature) => {
    const props = feature.properties || {};

    const district =
  props.name ||
  props.lgd_districtname ||
  props.district ||
  props.DISTRICT ||
  props.District ||
  props.dtname ||
  props.DTNAME ||
  props.DIST_NAME ||
  props.district_name ||
  "";

    const state =
      props.state ||
      props.STATE ||
      props.State ||
      props.stname ||
      props.STNAME ||
      props.state_name ||
      "";

    const key = `${state}|${district}`.toLowerCase();

    const prediction = predictionMap.get(key);

    if (!prediction) {
      return {
        fillColor: "#e9eeeb",
        fillOpacity: 0.35,
        color: "#9eaaa3",
        weight: 1,
      };
    }

    const visible = visibleKeys.has(key);

    return {
      fillColor: getRiskColor(
        prediction.risk_probability
      ),
      fillOpacity: visible ? 0.78 : 0.10,
      color: "#ffffff",
      weight: visible ? 1.2 : 0.6,
    };
  };

  const onEachFeature = (feature, layer) => {
    const props = feature.properties || {};

    const district =
  props.name ||
  props.lgd_districtname ||
  props.district ||
  props.DISTRICT ||
  props.District ||
  props.dtname ||
  props.DTNAME ||
  props.DIST_NAME ||
  props.district_name ||
  "";

    const state =
      props.state ||
      props.STATE ||
      props.State ||
      props.stname ||
      props.STNAME ||
      props.state_name ||
      "";

    const key = `${state}|${district}`.toLowerCase();

    const prediction = predictionMap.get(key);

    if (!prediction) return;

    layer.bindTooltip(
      `<strong>${prediction.district}</strong><br/>
       ${prediction.state}<br/>
       Risk: ${(prediction.risk_probability * 100).toFixed(1)}%<br/>
       Status: ${prediction.risk_label}`,
      {
        sticky: true,
      }
    );

    layer.on({
      click: () => {
        onSelectDistrict(prediction);
        onClose();
      },

      mouseover: (event) => {
        event.target.setStyle({
          weight: 3,
          color: "#123d2b",
          fillOpacity: 0.9,
        });

        event.target.bringToFront();
      },

      mouseout: (event) => {
        event.target.setStyle(
          styleFeature(feature)
        );
      },
    });
  };

  return (
  <div className={inline ? "inline-risk-map" : "map-modal-overlay"}>

    <div className={inline ? "inline-risk-map-content" : "map-modal"}>

      {!inline && (
        <div className="map-modal-header">

          <div>
            <span className="eyebrow">
              STUBBLEAI GEOSPATIAL VIEW
            </span>

            <h2>District Risk Map</h2>

            <p>
              Live crop-residue fire risk across monitored
              districts
            </p>
          </div>

          <button
            className="map-close-button"
            onClick={onClose}
          >
            ×
          </button>

        </div>
      )}

      <div className="map-controls">

        <div className="map-filter-group">

          <button
            className={
              selectedRisk === "All"
                ? "map-filter active"
                : "map-filter"
            }
            onClick={() => setSelectedRisk("All")}
          >
            All
          </button>

          <button
            className={
              selectedRisk === "Normal"
                ? "map-filter active"
                : "map-filter"
            }
            onClick={() => setSelectedRisk("Normal")}
          >
            Normal
          </button>

          <button
            className={
              selectedRisk === "Elevated"
                ? "map-filter active"
                : "map-filter"
            }
            onClick={() => setSelectedRisk("Elevated")}
          >
            Elevated
          </button>

        </div>

        <div className="map-legend">

          <span>
            <i className="legend-dot low"></i>
            &lt;20%
          </span>

          <span>
            <i className="legend-dot moderate"></i>
            20–40%
          </span>

          <span>
            <i className="legend-dot elevated"></i>
            40–60%
          </span>

          <span>
            <i className="legend-dot high"></i>
            &gt;60%
          </span>

        </div>

      </div>

      <div className="risk-map-container">

        <MapContainer
          center={INDIA_CENTER}
          zoom={5}
          scrollWheelZoom={true}
          className="risk-map"
        >

          <TileLayer
            attribution='&copy; OpenStreetMap contributors'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />

          {geojson && (
            <>
              <MapController geojson={geojson} />

              <GeoJSON
                key={selectedRisk}
                data={geojson}
                style={styleFeature}
                onEachFeature={onEachFeature}
              />
            </>
          )}

        </MapContainer>

        {!geojson && (
          <div className="map-loading">
            Loading district boundaries...
          </div>
        )}

      </div>

      <div className="map-footer">

        <span>
          Showing {filteredPredictions.length} of{" "}
          {predictions.length} monitored districts
        </span>

        <span>
          Click a district for detailed analysis
        </span>

      </div>

    </div>
  </div>
);
}