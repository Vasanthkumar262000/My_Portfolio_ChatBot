import { useEffect } from 'react';

export function usePrewarmChatbot(apiBaseUrl, timeoutMs = 8000) {
  useEffect(() => {
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

    return () => {
      clearTimeout(timeoutId);
      controller.abort();
    };
  }, [apiBaseUrl, timeoutMs]);
}
