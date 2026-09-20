# StubbleAI Security Notes

## Secrets

- Store NASA FIRMS credentials only in `.env` or the deployment platform's secret manager.
- Store the production `PREDICTION_TRIGGER_KEY` only in server-side environment variables or the deployment platform's secret manager.
- Never commit `.env`.
- Never place `FIRMS_MAP_KEY` or `PREDICTION_TRIGGER_KEY` in React/Vite `VITE_*` variables because frontend variables are exposed to browsers.
- If a key is ever exposed in a public repository, rotate/revoke it immediately.

## Model Security

- `*.pkl` / Joblib model files should be treated as trusted-code artifacts.
- Do not load model files from untrusted downloads.
- Keep the model artifact versioned and verify its source/checksum when distributing it.
- Do not expose model files or internal filesystem paths through the frontend API.
- The FastAPI API should return prediction outputs, not arbitrary model objects.

## API and Deployment

- Restrict CORS to the real frontend origin in production.
- Use HTTPS in production.
- Keep API keys and trigger secrets in the hosting provider's secret/environment-variable store.
- Protect the live prediction endpoint using the configured `X-Trigger-Key`.
- Apply rate limiting to prediction-trigger endpoints.
- Do not log API keys, tokens, trigger secrets, or full environment variables.

## Frontend Security

- Never expose private API credentials through `VITE_*` environment variables.
- The production frontend communicates with the prediction trigger through the server-side deployment proxy.
- The browser should not receive the `PREDICTION_TRIGGER_KEY`.

## Data and Privacy

- The prototype operates on district-level environmental data.
- No personally identifiable farmer information is required by the prediction pipeline.
- District-level predictions should not be used to identify, penalize, or attribute responsibility to individual farmers or communities.

## Responsible Use

StubbleAI is an environmental decision-support prototype.

Predictions represent model-estimated probabilities of elevated satellite-derived active-fire activity and should not be treated as confirmed crop-residue-burning events.

The system should not be used as the sole basis for enforcement, penalties, or decisions affecting individuals or communities.
