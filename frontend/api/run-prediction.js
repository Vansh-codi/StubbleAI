export default async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({
      detail: "Method not allowed",
    });
  }

  const backendUrl = process.env.BACKEND_URL;
  const triggerKey = process.env.PREDICTION_TRIGGER_KEY;

  if (!backendUrl || !triggerKey) {
    return res.status(500).json({
      detail: "Prediction service is not configured.",
    });
  }

  try {
    const response = await fetch(
      `${backendUrl}/api/run-prediction`,
      {
        method: "POST",
        headers: {
          "X-Trigger-Key": triggerKey,
        },
      }
    );

    const data = await response.json();

    return res.status(response.status).json(data);

  } catch (error) {
    console.error("Prediction proxy error:", error);

    return res.status(502).json({
      detail: "Unable to reach prediction backend.",
    });
  }
}