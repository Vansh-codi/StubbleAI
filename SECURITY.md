StubbleAI Security Notes

Secrets

Store NASA FIRMS credentials only in .env or the deployment platform's secret manager.

Never commit .env.

Never place FIRMS_MAP_KEY in React/Vite VITE_* variables because frontend variables are exposed to browsers.

If a key is ever exposed in a public repository, rotate/revoke it immediately.

Model security

*.pkl / Joblib model files should be treated as trusted-code artifacts.

Do not load model files from untrusted downloads.

Keep the model artifact versioned and verify its source/checksum when distributing it.

Do not expose model files or internal filesystem paths through the frontend API.

The FastAPI API should return prediction outputs, not arbitrary model objects.

Deployment

Restrict CORS to the real frontend origin in production.

Use HTTPS in production.

Keep API keys in the hosting provider's secret/environment-variable store.

Do not log API keys, tokens, or full environment variables.