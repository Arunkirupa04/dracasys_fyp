import { layerColor, moduleConfig } from '../hooks/useConfig'
import ModuleOutputsDisplay from './ModuleOutputsDisplay'

const STATUS = {
  idle:    { label: 'Waiting', color: 'var(--text-muted)' },
  pending: { label: 'Processing', color: 'var(--accent)' },
  error:   { label: 'Error', color: 'var(--yellow)' },
  ready:   { label: 'Ready', color: 'var(--green)' },
}

export default function ModuleSummaryCard({
  moduleKey, config, data, isAnomaly, onClick, streamActive, viewedSample,
}) {
  const mod = moduleConfig(config, moduleKey)
  const layerKey = mod.layer || 'prediction'
  const lc = layerColor(config, layerKey)
  const st = STATUS[data.status] || STATUS.idle
  const alert = isAnomaly && data.status === 'ready'
  const showOutputs = streamActive || data.status !== 'idle'

  return (
    <div
      className="module-summary-card"
      style={{
        ...styles.card,
        borderColor: alert ? 'rgba(239,68,68,0.5)' : 'var(--border)',
        boxShadow: alert ? '0 0 20px rgba(239,68,68,0.12)' : 'none',
      }}
    >
      <button type="button" onClick={onClick} style={styles.clickArea}>
        <div style={styles.top}>
          <span style={{ ...styles.id, color: lc }}>{mod.id}</span>
          <span style={{ ...styles.badge, color: st.color, borderColor: st.color + '44' }}>
            {alert ? 'Alert' : st.label}
          </span>
        </div>
        <div style={styles.title}>{mod.title}</div>
        <div style={styles.subtitle}>{mod.subtitle}</div>
      </button>

      <div style={styles.divider} />

      {showOutputs ? (
        <div style={styles.outputs}>
          <ModuleOutputsDisplay
            moduleKey={moduleKey}
            data={data}
            config={config}
            compact
            viewedSample={viewedSample}
          />
        </div>
      ) : (
        <div style={styles.idleHint}>Start stream to see live module outputs</div>
      )}

      <button type="button" onClick={onClick} className="detail-btn" style={styles.detailBtn}>
        Open full detail & charts →
      </button>
    </div>
  )
}

const styles = {
  card: {
    background: 'var(--surface)',
    border: '1px solid var(--border)',
    borderRadius: 12,
    padding: '16px 16px 12px',
    textAlign: 'left',
    display: 'flex',
    flexDirection: 'column',
    minHeight: 320,
  },
  clickArea: {
    background: 'none', border: 'none', padding: 0, cursor: 'pointer',
    textAlign: 'left', fontFamily: 'inherit', color: 'inherit', width: '100%',
  },
  top: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 },
  id: { fontSize: 11, fontWeight: 700, fontFamily: 'var(--mono)', letterSpacing: '0.1em' },
  badge: {
    fontSize: 10, fontWeight: 600, padding: '2px 8px', borderRadius: 4,
    border: '1px solid', textTransform: 'uppercase', letterSpacing: '0.05em',
  },
  title: { fontSize: 15, fontWeight: 700, marginBottom: 2 },
  subtitle: { fontSize: 11, color: 'var(--text-dim)' },
  divider: { height: 1, background: 'var(--border)', margin: '10px 0' },
  outputs: { flex: 1, minHeight: 0, marginBottom: 10 },
  idleHint: { flex: 1, fontSize: 11, color: 'var(--text-muted)', fontStyle: 'italic', padding: '12px 0' },
  detailBtn: {
    background: 'var(--surface2)', border: '1px solid var(--border)', borderRadius: 6,
    padding: '8px 12px', fontSize: 11, fontWeight: 600, color: 'var(--accent)',
    cursor: 'pointer', fontFamily: 'inherit', width: '100%', textAlign: 'center',
    transition: 'background 0.15s, border-color 0.15s',
  },
}
