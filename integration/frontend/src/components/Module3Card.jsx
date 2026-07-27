import ModuleCard from './ModuleCard'

export default function Module3Card({ data }) {
  const result   = data.result
  const isAnom   = result?.is_anomaly ?? false
  const mse      = result?.reconstruction_mse
  const threshold= result?.threshold ?? 0.098

  // Fill bar: ratio of MSE to 2× threshold (capped at 100%)
  const fillPct = mse != null ? Math.min(100, (mse / (threshold * 2)) * 100) : 0
  const barColor = mse > threshold ? 'var(--red)' : 'var(--green)'

  return (
    <ModuleCard
      moduleId="M3"
      title="System Anomaly Detector"
      subtitle="Variational Autoencoder (VAE)"
      layer="Anomaly Detection"
      layerColor="var(--red)"
      status={data.status}
      isAnomaly={isAnom}
    >
      {result == null ? (
        <Placeholder />
      ) : (
        <>
          {/* Alert / Normal badge */}
          <div style={{ ...styles.alertBadge, background: isAnom ? 'rgba(239,68,68,0.15)' : 'rgba(34,197,94,0.12)', borderColor: isAnom ? 'rgba(239,68,68,0.4)' : 'rgba(34,197,94,0.3)', color: isAnom ? 'var(--red)' : 'var(--green)' }}>
            <span style={styles.alertIcon}>{isAnom ? '⚠' : '✓'}</span>
            {isAnom ? 'System Anomaly Detected' : 'Normal Behavior'}
          </div>

          {isAnom && result.anomaly_type && (
            <div style={styles.anomalyType}>{result.anomaly_type}</div>
          )}

          {/* MSE score bar */}
          <div style={styles.scoreSection}>
            <div style={styles.scoreHeader}>
              <span style={styles.scoreLabel}>Reconstruction MSE</span>
              <span style={{ ...styles.scoreVal, color: barColor }}>
                {mse?.toFixed(5)}
              </span>
            </div>
            <div style={styles.barTrack}>
              <div style={{ ...styles.barFill, width: `${fillPct}%`, background: barColor }} />
              {/* Threshold marker */}
              <div style={{ ...styles.thresholdLine, left: `${Math.min(50, 50)}%` }} />
            </div>
            <div style={styles.thresholdLabel}>
              Threshold: <span style={{ fontFamily: 'var(--mono)' }}>{threshold.toFixed(5)}</span>
            </div>
          </div>

          <div style={styles.footer}>
            Sample {result.sample} · 30-step VAE window
          </div>
        </>
      )}
    </ModuleCard>
  )
}

function Placeholder() {
  return (
    <div style={styles.placeholder}>
      <div style={styles.phIcon}>🛡</div>
      <div style={styles.phText}>Waiting for container metrics</div>
      <div style={styles.phSub}>Detects CPU saturation, memory leaks, pod failures via reconstruction error</div>
    </div>
  )
}

const styles = {
  alertBadge: {
    display: 'flex', alignItems: 'center', gap: 8,
    padding: '10px 14px', borderRadius: 8,
    border: '1px solid', fontSize: 14, fontWeight: 700,
    transition: 'all 0.3s',
  },
  alertIcon: { fontSize: 16 },
  anomalyType: {
    fontSize: 12, color: 'var(--text-dim)', fontStyle: 'italic',
    padding: '2px 4px',
  },
  scoreSection: { marginTop: 4 },
  scoreHeader:  { display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 },
  scoreLabel:   { fontSize: 11, color: 'var(--text-dim)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.06em' },
  scoreVal:     { fontSize: 13, fontFamily: 'var(--mono)', fontWeight: 700 },
  barTrack: {
    position: 'relative', height: 8, background: 'var(--surface2)',
    borderRadius: 4, overflow: 'visible',
  },
  barFill: {
    height: '100%', borderRadius: 4,
    transition: 'width 0.6s ease, background 0.3s',
    maxWidth: '100%',
  },
  thresholdLine: {
    position: 'absolute', top: -4, width: 2, height: 16,
    background: 'var(--yellow)', borderRadius: 1,
  },
  thresholdLabel: { marginTop: 6, fontSize: 11, color: 'var(--text-muted)' },
  placeholder: {
    flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center',
    justifyContent: 'center', gap: 8, textAlign: 'center', padding: '16px 8px',
  },
  phIcon:  { fontSize: 28, opacity: 0.35 },
  phText:  { fontSize: 13, fontWeight: 600, color: 'var(--text-dim)' },
  phSub:   { fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.5 },
  footer:  { fontSize: 11, color: 'var(--text-muted)', fontFamily: 'var(--mono)', marginTop: 'auto' },
}
