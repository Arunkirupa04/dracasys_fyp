/**
 * Resolve what to show for a module when the user has scrubbed to a sample.
 * M1 is a one-shot forecast (same for all samples). M2–M4 use per-sample history.
 */

export function findHistoryAtSample(history, sample) {
  if (!history?.length || !sample) return null
  return history.find((r) => r.sample === sample) || null
}

export function historyUpToSample(history, sample) {
  if (!history?.length) return []
  if (!sample) return history
  return history.filter((r) => r.sample <= sample)
}

/**
 * Build card state for overview/detail at a viewed sample.
 * liveState = latest module state from App; history = per-sample results.
 */
export function dataAtSample(moduleKey, liveState, history, viewedSample, m1State) {
  if (moduleKey === 'm1') {
    return m1State || liveState
  }

  if (!viewedSample) return liveState

  const hit = findHistoryAtSample(history, viewedSample)
  if (!hit) {
    // Sample not yet available for this module
    if (liveState?.status === 'pending' && (!history || history.length === 0)) {
      return liveState
    }
    return {
      status: 'idle',
      result: null,
      sampleIndex: viewedSample,
    }
  }

  return {
    status: hit.status === 'error' ? 'error' : 'ready',
    result: hit,
    sampleIndex: viewedSample,
  }
}
