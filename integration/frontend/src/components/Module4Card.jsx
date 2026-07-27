import ModuleCard from './ModuleCard'

export default function Module4Card({ data }) {
  const result    = data.result
  const isAnom    = result?.is_anomaly ?? false
  const score     = result?.score
  const threshold = result?.threshold ?? -0.3475
  const attackType = result?.attack_type

  // Score is inverted: higher = more anomalous.
  // Normalize for display: map [threshold-1 .. threshold+1] → [0..100%]
  const fillPct = score != null
    ? Math.min(100, Math.max(0, ((score - (threshold - 0.5)) / 1.0) * 100))
    : 0
  const barColor = isAnom ? 'var(--red)' : 'var(--green)'

  return (
    <ModuleCard
      moduleId="M4"
      title="Security Anomaly Detector"
      subtitle="Sequence Bottleneck Autoencoder"
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
            <span style={styles.alertIcon}>{isAnom ? '🚨' : '✓'}</span>
            {isAnom ? 'Security Threat Detected' : 'Normal Traffic'}
          </div>

          {/* Attack type label */}
          {isAnom && attackType && (
            <div style={styles.attackBadge}>
              <span style={styles.attackLabel}>Attack type</span>
              <span style={styles.attackName}>{attackType}</span>
            </div>
          )}

          {/* Score bar */}
          <div style={styles.scoreSection}>
            <div style={styles.scoreHeader}>
              <span style={styles.scoreLabel}>Anomaly Score</span>
              <span style={{ ...styles.scoreVal, color: barColor }}>
                {score?.toFixed(5)}
              </span>
            </div>
            <div style={styles.barTrack}>
              <div style={{ ...styles.barFill, width: `${fillPct}%`, background: barColor }} />
            </div>
            <div style={styles.thresholdLabel}>
              Threshold: <span style={{ fontFamily: 'var(--mono)' }}>{threshold.toFixed(5)}</span>
              &nbsp;·&nbsp;Higher score = more anomalous
            </div>
          </div>

          {/* Model info */}
          <div style={styles.modelInfo}>
            <Tag label="ROC-AUC" value="0.845" />
            <Tag label="F1" value="0.803" />
            <Tag label="HPO_best" />
          </div>

          <div style={styles.footer}>
            Sample {result.sample} · CICFlowMeter · 163 features · T=10
          </div>
        </>
      )}
    </ModuleCard>
  )
}

function Tag({ label, value }) {
  return (
    <span style={styles.tag}>
      {label}{value ? `: ${value}` : ''}
    </span>
  )
}

function Placeholder() {
  return (
    <div style={styles.placeholder}>
      <div style={styles.phIcon}>🔒</div>
      <div style={styles.phText}>Waiting for network flow data</div>
      <div style={styles.phSub}>Detects CVEs, RCE, recon, and container escapes via reconstruction error on CICFlowMeter flows</div>
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
  attackBadge: {
    background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.25)',
    borderRadius: 8, padding: '8px 12px',
    display: 'flex', flexDirection: 'column', gap: 2,
  },
  attackLabel: { fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--red)', fontWeight: 700 },
  attackName:  { fontSize: 13, fontWeight: 600, color: 'var(--text)' },
  scoreSection: { marginTop: 4 },
  scoreHeader:  { display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 },
  scoreLabel:   { fontSize: 11, color: 'var(--text-dim)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.06em' },
  scoreVal:     { fontSize: 13, fontFamily: 'var(--mono)', fontWeight: 700 },
  barTrack: {
    height: 8, background: 'var(--surface2)', borderRadius: 4, overflow: 'hidden',
  },
  barFill: {
    height: '100%', borderRadius: 4,
    transition: 'width 0.6s ease, background 0.3s',
    maxWidth: '100%',
  },
  thresholdLabel: { marginTop: 6, fontSize: 11, color: 'var(--text-muted)' },
  modelInfo: { display: 'flex', gap: 6, flexWrap: 'wrap', marginTop: 4 },
  tag: {
    background: 'var(--surface2)', border: '1px solid var(--border)',
    borderRadius: 4, padding: '2px 8px', fontSize: 10,
    color: 'var(--text-dim)', fontFamily: 'var(--mono)',
  },
  placeholder: {
    flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center',
    justifyContent: 'center', gap: 8, textAlign: 'center', padding: '16px 8px',
  },
  phIcon:  { fontSize: 28, opacity: 0.35 },
  phText:  { fontSize: 13, fontWeight: 600, color: 'var(--text-dim)' },
  phSub:   { fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.5 },
  footer:  { fontSize: 11, color: 'var(--text-muted)', fontFamily: 'var(--mono)', marginTop: 'auto' },
}
