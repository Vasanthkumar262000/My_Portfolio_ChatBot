function prewarmChatbotBackend(apiBaseUrl, timeoutMs = 8000) {
  if (!apiBaseUrl || typeof apiBaseUrl !== 'string') return;

  const url = apiBaseUrl.replace(/\/$/, '') + '/health';

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  fetch(url, {
    method: 'GET',
    mode: 'cors',
    signal: controller.signal,
  })
    .then(() => clearTimeout(timeoutId))
    .catch(() => clearTimeout(timeoutId));
}

if (typeof window !== 'undefined' && window.CHATBOT_API_URL) {
  prewarmChatbotBackend(window.CHATBOT_API_URL);
}
