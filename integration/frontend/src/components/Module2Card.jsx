import ModuleCard from './ModuleCard'

const HORIZON_LABELS = { h1: '15 s ahead', h2: '30 s ahead', h3: '45 s ahead' }

export default function Module2Card({ data }) {
  const result = data.result
  const horizons = result?.horizons

  return (
    <ModuleCard
      moduleId="M2"
      title="Short-Term Prediction"
      subtitle="Adaptive GRU — CPU & Memory"
      layer="Prediction Layer"
      layerColor="var(--accent)"
      status={data.status}
      isAnomaly={false}
    >
      {!horizons ? (
        <Placeholder />
      ) : (
        <div style={styles.horizonsWrap}>
          {Object.entries(horizons).map(([key, h]) => (
            <HorizonBlock key={key} label={HORIZON_LABELS[key]} h={h} />
          ))}
        </div>
      )}
      {result && (
        <div style={styles.footer}>
          Sample {result.sample} — 3 horizon forecasts
        </div>
      )}
    </ModuleCard>
  )
}

function HorizonBlock({ label, h }) {
  return (
    <div style={styles.hBlock}>
      <div style={styles.hLabel}>{label}</div>
      <MetricRow icon="⚡" name="CPU" value={`${h.cpu_usage?.toFixed(1)} s`} />
      <MetricRow icon="💾" name="Memory" value={`${h.mem_usage_mb?.toFixed(1)} MB`} />
      <MetricRow icon="📊" name="WSS" value={`${h.mem_wss_mb?.toFixed(1)} MB`} />
      <MetricRow icon="🗃" name="RSS" value={`${h.mem_rss_mb?.toFixed(1)} MB`} />
    </div>
  )
}

function MetricRow({ icon, name, value }) {
  return (
    <div style={styles.metricRow}>
      <span style={styles.metricIcon}>{icon}</span>
      <span style={styles.metricName}>{name}</span>
      <span style={styles.metricVal}>{value}</span>
    </div>
  )
}

function Placeholder() {
  return (
    <div style={styles.placeholder}>
      <div style={styles.phIcon}>📈</div>
      <div style={styles.phText}>Waiting for stream data</div>
      <div style={styles.phSub}>Will forecast CPU & memory at 15 s / 30 s / 45 s horizons</div>
    </div>
  )
}

const styles = {
  horizonsWrap: { display: 'flex', flexDirection: 'column', gap: 12 },
  hBlock: {
    background: 'var(--surface2)', borderRadius: 8, padding: '10px 12px',
    border: '1px solid var(--border)',
  },
  hLabel: {
    fontSize: 10, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.08em',
    color: 'var(--accent)', marginBottom: 6,
  },
  metricRow: { display: 'flex', alignItems: 'center', gap: 6, padding: '2px 0' },
  metricIcon: { fontSize: 12, width: 16, textAlign: 'center' },
  metricName: { fontSize: 12, color: 'var(--text-dim)', flex: 1 },
  metricVal:  { fontSize: 12, fontFamily: 'var(--mono)', color: 'var(--text)', fontWeight: 500 },
  placeholder: {
    flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center',
    justifyContent: 'center', gap: 8, textAlign: 'center', padding: '16px 8px',
  },
  phIcon:  { fontSize: 28, opacity: 0.35 },
  phText:  { fontSize: 13, fontWeight: 600, color: 'var(--text-dim)' },
  phSub:   { fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.5 },
  footer:  { fontSize: 11, color: 'var(--text-muted)', fontFamily: 'var(--mono)', marginTop: 'auto' },
}
