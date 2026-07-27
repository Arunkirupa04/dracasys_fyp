/**
 * Build display rows for every field a module exposes over SSE.
 * Used on the overview cards (live outputs) and reusable elsewhere.
 */

const fmt = (v, digits = 2) =>
  typeof v === 'number' && !Number.isNaN(v) ? v.toFixed(digits) : v ?? '—'

export function getModuleOutputs(moduleKey, data, config) {
  const result = data?.result
  const hLabels = config?.ui?.horizon_labels || { h1: '15s', h2: '30s', h3: '45s' }

  if (!result || data.status === 'idle') {
    return { rows: [{ label: 'Status', value: 'Waiting — start stream' }], table: null }
  }
  if (data.status === 'pending') {
    return { rows: [{ label: 'Status', value: result.message || 'Processing…' }], table: null }
  }
  if (data.status === 'error') {
    return { rows: [{ label: 'Error', value: result.error || 'Unknown error', alert: true }], table: null }
  }

  switch (moduleKey) {
    case 'm1':
      return buildM1Outputs(result)
    case 'm2':
      return buildM2Outputs(result, hLabels)
    case 'm3':
      return buildM3Outputs(result)
    case 'm4':
      return buildM4Outputs(result)
    default:
      return { rows: [], table: null }
  }
}

function buildM1Outputs(r) {
  const pred = r.predicted_cpu_percent || []
  const prophet = r.prophet_component_percent || []
  const gru = r.gru_residual_component_percent || []
  const first = pred[0]
  const last = pred[pred.length - 1]
  const min = pred.length ? Math.min(...pred) : null
  const max = pred.length ? Math.max(...pred) : null

  return {
    rows: [
      { label: 'Container ID', value: r.container_id },
      { label: 'Model version', value: r.model_version },
      { label: 'Horizon steps', value: `${r.horizon_steps} × 15 min (24 h)` },
      { label: 'History used', value: `${r.history_steps_used} steps` },
      { label: 'Scaler mode', value: r.scaler_mode },
      { label: 'Latency', value: `${fmt(r.processing_time_ms / 1000, 1)} s` },
      { label: 'Forecast start CPU', value: `${fmt(first, 1)}%` },
      { label: 'Forecast end CPU', value: `${fmt(last, 1)}%`, highlight: true },
      { label: 'Forecast min / max', value: `${fmt(min, 1)}% / ${fmt(max, 1)}%` },
      { label: 'Prophet end', value: `${fmt(prophet[prophet.length - 1], 1)}%` },
      { label: 'GRU residual end', value: `${fmt(gru[gru.length - 1], 1)}%` },
      { label: 'History tail (chart)', value: `${(r.history_tail_cpu_percent || []).length} points` },
      { label: 'Forecast points', value: String(pred.length) },
    ],
    table: null,
  }
}

function buildM2Outputs(r, hLabels) {
  const actual = r.actual || {}
  const horizons = r.horizons || {}
  const keys = ['h1', 'h2', 'h3']

  const rows = [
    { label: 'Sample', value: `#${r.sample}` },
    { label: 'Actual CPU', value: `${fmt(actual.cpu_usage, 1)} s` },
    { label: 'Actual Mem', value: `${fmt(actual.mem_usage_mb, 1)} MB`, highlight: true },
    { label: 'Actual WSS', value: `${fmt(actual.mem_wss_mb, 1)} MB` },
    { label: 'Actual RSS', value: `${fmt(actual.mem_rss_mb, 1)} MB` },
  ]

  keys.forEach((h) => {
    const hz = horizons[h]
    if (!hz) return
    rows.push(
      { label: `${hLabels[h]} CPU forecast`, value: `${fmt(hz.cpu_usage, 1)} s` },
      { label: `${hLabels[h]} Mem forecast`, value: `${fmt(hz.mem_usage_mb, 1)} MB` },
      { label: `${hLabels[h]} WSS forecast`, value: `${fmt(hz.mem_wss_mb, 1)} MB` },
      { label: `${hLabels[h]} RSS forecast`, value: `${fmt(hz.mem_rss_mb, 1)} MB` },
    )
  })

  const table = {
    headers: ['Metric', 'Actual', ...keys.map((h) => hLabels[h])],
    rows: [
      ['CPU (s)', fmt(actual.cpu_usage, 1), ...keys.map((h) => fmt(horizons[h]?.cpu_usage, 1))],
      ['Mem (MB)', fmt(actual.mem_usage_mb, 1), ...keys.map((h) => fmt(horizons[h]?.mem_usage_mb, 1))],
      ['WSS (MB)', fmt(actual.mem_wss_mb, 1), ...keys.map((h) => fmt(horizons[h]?.mem_wss_mb, 1))],
      ['RSS (MB)', fmt(actual.mem_rss_mb, 1), ...keys.map((h) => fmt(horizons[h]?.mem_rss_mb, 1))],
    ],
  }

  return { rows, table }
}

function buildM3Outputs(r) {
  return {
    rows: [
      { label: 'Sample', value: `#${r.sample}` },
      { label: 'Decision', value: r.is_anomaly ? 'ANOMALY' : 'Normal', alert: r.is_anomaly },
      { label: 'Reconstruction MSE', value: fmt(r.reconstruction_mse, 5), highlight: r.is_anomaly },
      { label: 'Threshold', value: fmt(r.threshold, 5) },
      { label: 'MSE / threshold', value: r.threshold ? fmt(r.reconstruction_mse / r.threshold, 2) + '×' : '—' },
      { label: 'Anomaly type', value: r.anomaly_type || (r.is_anomaly ? '—' : 'N/A') },
    ],
    table: null,
  }
}

function buildM4Outputs(r) {
  return {
    rows: [
      { label: 'Sample', value: `#${r.sample}` },
      { label: 'Decision', value: r.is_anomaly ? 'THREAT' : 'Normal', alert: r.is_anomaly },
      { label: 'Anomaly score', value: fmt(r.score, 5), highlight: r.is_anomaly },
      { label: 'Threshold', value: fmt(r.threshold, 5) },
      { label: 'Score − threshold', value: fmt((r.score ?? 0) - (r.threshold ?? 0), 5) },
      { label: 'Attack type', value: r.attack_type || (r.is_anomaly ? '—' : 'N/A') },
      { label: 'Ground-truth label', value: r.ground_truth_label != null ? String(r.ground_truth_label) : '—' },
    ],
    table: null,
  }
}
