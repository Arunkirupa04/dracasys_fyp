import ModuleCard from './ModuleCard'

export default function Module1Card({ data }) {
  return (
    <ModuleCard
      moduleId="M1"
      title="Long-Term Forecasting"
      subtitle="Prophet + LSTM"
      layer="Prediction Layer"
      layerColor="var(--accent)"
      status={data.status}
    >
      <div style={styles.pending}>
        <div style={styles.icon}>🔮</div>
        <div style={styles.msg}>Module 1 not yet implemented</div>
        <div style={styles.sub}>Prophet + LSTM hybrid will provide long-term capacity trend forecasts</div>
        <div style={styles.badge}>Coming Soon</div>
      </div>
    </ModuleCard>
  )
}

const styles = {
  pending: {
    display: 'flex', flexDirection: 'column', alignItems: 'center',
    justifyContent: 'center', gap: 10, padding: '24px 12px', textAlign: 'center',
    flex: 1,
  },
  icon:  { fontSize: 32, opacity: 0.4 },
  msg:   { fontSize: 13, fontWeight: 600, color: 'var(--text-dim)' },
  sub:   { fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.5 },
  badge: {
    marginTop: 6, padding: '4px 12px', borderRadius: 999,
    background: 'var(--surface2)', border: '1px solid var(--border)',
    fontSize: 11, fontWeight: 600, color: 'var(--text-muted)',
    textTransform: 'uppercase', letterSpacing: '0.06em',
  },
}
