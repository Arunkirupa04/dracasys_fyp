/**
 * Module documentation panel — approach, I/O, validation metrics from config.json.
 */
export default function ModuleInfoPanel({ mod }) {
  if (!mod?.approach) return null

  return (
    <div style={styles.panel}>
      <div style={styles.section}>
        <div style={styles.sectionTitle}>Current Approach</div>
        <p style={styles.body}>{mod.approach}</p>
      </div>

      <div style={styles.grid}>
        {mod.inputs && (
          <div style={styles.col}>
            <div style={styles.sectionTitle}>Inputs</div>
            <ul style={styles.list}>
              {Object.entries(mod.inputs).map(([k, v]) => (
                <li key={k}><span style={styles.key}>{k.replace(/_/g, ' ')}:</span> {v}</li>
              ))}
            </ul>
          </div>
        )}
        {mod.outputs && (
          <div style={styles.col}>
            <div style={styles.sectionTitle}>Outputs</div>
            <ul style={styles.list}>
              {Object.entries(mod.outputs).map(([k, v]) => (
                <li key={k}>
                  <span style={styles.key}>{k.replace(/_/g, ' ')}:</span>{' '}
                  {Array.isArray(v) ? v.join(', ') : v}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {mod.validation_metrics?.length > 0 && (
        <div style={styles.section}>
          <div style={styles.sectionTitle}>Validation Metrics</div>
          <div style={styles.metricsGrid}>
            {mod.validation_metrics.map((m) => (
              <div key={m.label} style={styles.metricCard}>
                <div style={styles.metricValue}>{m.value}</div>
                <div style={styles.metricLabel}>{m.label}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {mod.key_details?.length > 0 && (
        <div style={styles.section}>
          <div style={styles.sectionTitle}>Key Details</div>
          <ul style={styles.list}>
            {mod.key_details.map((d) => <li key={d}>{d}</li>)}
          </ul>
        </div>
      )}
    </div>
  )
}

const styles = {
  panel: {
    background: 'var(--surface)',
    border: '1px solid var(--border)',
    borderRadius: 10,
    padding: '16px 20px',
    marginBottom: 20,
  },
  section: { marginBottom: 16 },
  sectionTitle: {
    fontSize: 10,
    fontWeight: 700,
    textTransform: 'uppercase',
    letterSpacing: '0.08em',
    color: 'var(--text-dim)',
    marginBottom: 8,
  },
  body: { fontSize: 13, color: 'var(--text)', lineHeight: 1.65 },
  grid: { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 16 },
  col: {},
  list: { paddingLeft: 18, fontSize: 12, color: 'var(--text-dim)', lineHeight: 1.7 },
  key: { color: 'var(--text)', fontWeight: 600, textTransform: 'capitalize' },
  metricsGrid: { display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))', gap: 10 },
  metricCard: {
    background: 'var(--surface2)',
    border: '1px solid var(--border)',
    borderRadius: 8,
    padding: '12px 14px',
    textAlign: 'center',
  },
  metricValue: { fontSize: 18, fontWeight: 700, fontFamily: 'var(--mono)', color: 'var(--accent)', marginBottom: 4 },
  metricLabel: { fontSize: 10, color: 'var(--text-muted)', lineHeight: 1.4 },
}
