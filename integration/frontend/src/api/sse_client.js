/**
 * Connects to /api/stream via EventSource and dispatches events to callbacks.
 *
 * onEvent(parsedData)  — called for every "data:" SSE line
 * onError(err)         — called on EventSource error
 *
 * Returns a cleanup function that closes the connection.
 */
export function startStream(onEvent, onError) {
  const es = new EventSource('/api/stream')

  es.onmessage = (e) => {
    try {
      const data = JSON.parse(e.data)
      onEvent(data)
    } catch {
      // ignore malformed lines
    }
  }

  es.onerror = (err) => {
    es.close()
    onError(err)
  }

  return () => es.close()
}
