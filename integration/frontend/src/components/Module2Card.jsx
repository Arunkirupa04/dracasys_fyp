import ModuleCard from './ModuleCard'
import { ActualPredictedChart, GroupedBarChart } from './charts'
import { moduleConfig } from '../hooks/useConfig'

export default function Module2Card({ data, history = [], config }) {
  const mod = moduleConfig(config, 'm2')
  const result = data.result
  const horizons = result?.horizons
  const hLabels = config?.ui?.horizon_labels || { h1: '15s', h2: '30s', h3: '45s' }
  const horizonKeys = ['h1', 'h2', 'h3']

  const actualMem = history.map((r) => r.actual?.mem_usage_mb)
  const actualCpu = history.map((r) => r.actual?.cpu_usage)
  const predictedMemTail = result
    ? horizonKeys.map((h) => result.horizons[h]?.mem_usage_mb)
    : []
  const predictedCpuTail = result
    ? horizonKeys.map((h) => result.horizons[h]?.cpu_usage)
    : []

  const latestActual = result?.actual

  return (
    <ModuleCard
      moduleId={mod.id}
      title={mod.title}
      subtitle={mod.subtitle}
      layer={config?.ui?.layers?.prediction?.label}
      layerColor="var(--accent)"
      status={data.status}
    >
      {!horizons ? (
        <Placeholder mod={mod} />
      ) : (
        <>
          <div style={styles.metricsRow}>
            <Metric label="Actual CPU now" value={`${latestActual?.cpu_usage?.toFixed(1)} s`} />
            <Metric label="Actual Mem now" value={`${latestActual?.mem_usage_mb?.toFixed(1)} MB`} highlight />
            <Metric label="H1 forecast Mem" value={`${horizons.h1.mem_usage_mb.toFixed(1)} MB`} />
            <Metric label="Sample" value={`#${result.sample}`} />
          </div>

          <div className="detail-chart-grid" style={styles.chartGrid}>
            <ActualPredictedChart
              title="Memory — Actual Stream vs H1/H2/H3 Forecast"
              yLabel="Memory (MB)"
              unit=" MB"
              actual={actualMem}
              predicted={predictedMemTail}
              actualLabel="Actual (streamed samples)"
              predictedLabel="Forecast tail (H1→H3)"
              width={640}
              height={260}
            />
            <ActualPredictedChart
              title="CPU Usage — Actual Stream vs Forecast"
              yLabel="CPU (seconds)"
              actual={actualCpu}
              predicted={predictedCpuTail}
              actualLabel="Actual (streamed)"
              predictedLabel="Forecast (H1→H3)"
              width={640}
              height={260}
            />
          </div>

          <GroupedBarChart
            title="Current Sample — Actual vs H1 Forecast by Metric"
            yLabel="Value"
            categories={['CPU (s)', 'Mem (MB)', 'WSS (MB)', 'RSS (MB)']}
            actual={[
              latestActual?.cpu_usage,
              latestActual?.mem_usage_mb,
              latestActual?.mem_wss_mb,
              latestActual?.mem_rss_mb,
            ]}
            predicted={[
              horizons.h1.cpu_usage,
              horizons.h1.mem_usage_mb,
              horizons.h1.mem_wss_mb,
              horizons.h1.mem_rss_mb,
            ]}
            actualLabel="Actual (now)"
            predictedLabel="Predicted H1"
            width={640}
            height={240}
          />

          <table style={styles.table}>
            <thead>
              <tr>
                <th style={styles.th}>Metric</th>
                <th style={styles.th}>Actual (now)</th>
                {horizonKeys.map((h) => (
                  <th key={h} style={styles.th}>{hLabels[h]}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {[
                { key: 'cpu_usage', label: 'CPU (s)', fmt: (v) => v?.toFixed(1), actual: latestActual?.cpu_usage },
                { key: 'mem_usage_mb', label: 'Mem (MB)', fmt: (v) => v?.toFixed(1), actual: latestActual?.mem_usage_mb },
                { key: 'mem_wss_mb', label: 'WSS (MB)', fmt: (v) => v?.toFixed(1), actual: latestActual?.mem_wss_mb },
                { key: 'mem_rss_mb', label: 'RSS (MB)', fmt: (v) => v?.toFixed(1), actual: latestActual?.mem_rss_mb },
              ].map(({ key, label, fmt, actual }) => (
                <tr key={key}>
                  <td style={styles.tdLabel}>{label}</td>
                  <td style={{ ...styles.td, color: 'var(--text)' }}>{fmt(actual)}</td>
                  {horizonKeys.map((h) => (
                    <td key={h} style={styles.td}>{fmt(horizons[h][key])}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </ModuleCard>
  )
}

function Metric({ label, value, highlight }) {
  return (
    <div style={styles.metric}>
      <div style={{ ...styles.metricVal, color: highlight ? 'var(--accent)' : 'var(--text)' }}>{value}</div>
      <div style={styles.metricLabel}>{label}</div>
    </div>
  )
}

function Placeholder({ mod }) {
  return (
    <div style={styles.placeholder}>
      <div style={styles.phIcon}>📈</div>
      <div style={styles.phText}>Waiting for stream data</div>
      <div style={styles.phSub}>
        Forecasts {mod?.outputs?.targets?.join(', ')} at {mod?.outputs?.horizons?.join(', ')}
      </div>
    </div>
  )
}

const styles = {
  metricsRow: { display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12, marginBottom: 16 },
  metric: { background: 'var(--surface2)', borderRadius: 8, padding: '12px 14px', border: '1px solid var(--border)' },
  metricVal: { fontSize: 18, fontWeight: 700, fontFamily: 'var(--mono)' },
  metricLabel: { fontSize: 10, color: 'var(--text-muted)', marginTop: 4, textTransform: 'uppercase' },
  chartGrid: { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 8 },
  table: { width: '100%', borderCollapse: 'collapse', marginTop: 12 },
  th: {
    fontSize: 10, fontWeight: 700, color: 'var(--accent)', textAlign: 'right',
    padding: '8px 6px', textTransform: 'uppercase', borderBottom: '1px solid var(--border)',
  },
  tdLabel: { fontSize: 12, color: 'var(--text-dim)', padding: '6px 6px', textAlign: 'left' },
  td: {
    fontSize: 12, fontFamily: 'var(--mono)', color: 'var(--accent)', textAlign: 'right',
    padding: '6px 6px', borderBottom: '1px solid var(--border)',
  },
  placeholder: {
    display: 'flex', flexDirection: 'column', alignItems: 'center',
    justifyContent: 'center', gap: 8, textAlign: 'center', padding: '40px 8px',
  },
  phIcon:  { fontSize: 28, opacity: 0.35 },
  phText:  { fontSize: 13, fontWeight: 600, color: 'var(--text-dim)' },
  phSub:   { fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.5 },
}
