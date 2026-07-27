export default function SystemStatus({ anyAnomaly, streamState }) {
  if (streamState === 'idle') {
    return (
      <div style={{ ...styles.badge, ...styles.badgeIdle }}>
        <span style={styles.dot} /> System Ready
      </div>
    )
  }
  if (anyAnomaly) {
    return (
      <div style={{ ...styles.badge, ...styles.badgeAlert }}>
        <span style={{ ...styles.dot, ...styles.dotPulse }} /> Anomaly Detected
      </div>
    )
  }
  return (
    <div style={{ ...styles.badge, ...styles.badgeNormal }}>
      <span style={styles.dot} /> All Systems Normal
    </div>
  )
}

const styles = {
  badge: {
    display: 'flex', alignItems: 'center', gap: 8,
    padding: '8px 16px', borderRadius: 999,
    fontSize: 13, fontWeight: 600,
    border: '1px solid transparent',
  },
  badgeIdle:   { background: 'var(--surface2)', color: 'var(--text-dim)', borderColor: 'var(--border)' },
  badgeNormal: { background: 'rgba(34,197,94,0.12)', color: 'var(--green)', borderColor: 'rgba(34,197,94,0.3)' },
  badgeAlert:  { background: 'rgba(239,68,68,0.12)', color: 'var(--red)',   borderColor: 'rgba(239,68,68,0.3)' },
  dot: {
    width: 8, height: 8, borderRadius: '50%', background: 'currentColor',
    display: 'inline-block',
  },
  dotPulse: {
    animation: 'pulse 1.2s infinite',
    boxShadow: '0 0 0 3px rgba(239,68,68,0.3)',
  },
}
