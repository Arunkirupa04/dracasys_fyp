/**
 * Base card container used by all 4 module cards.
 * Handles the border glow when anomaly is detected.
 */
export default function ModuleCard({
  moduleId, title, subtitle, layer, layerColor,
  status, isAnomaly, children,
}) {
  const alertBorder = isAnomaly
    ? '1px solid rgba(239,68,68,0.5)'
    : '1px solid var(--border)'

  const alertGlow = isAnomaly
    ? '0 0 20px rgba(239,68,68,0.15)'
    : 'none'

  const dotColor = {
    idle:    'var(--text-muted)',
    pending: 'var(--accent)',
    error:   'var(--yellow)',
    ready:   isAnomaly ? 'var(--red)' : 'var(--green)',
  }[status] || 'var(--text-muted)'

  const statusLabel = {
    idle:    'Waiting…',
    pending: 'Processing…',
    error:   'Error',
    ready:   isAnomaly ? 'Alert' : 'Normal',
  }[status] || ''

  return (
    <div style={{ ...styles.card, border: alertBorder, boxShadow: alertGlow }}>
      {/* Card header */}
      <div style={styles.header}>
        <div style={styles.headerTop}>
          <span style={{ ...styles.moduleId, color: layerColor }}>{moduleId}</span>
          <span style={{ ...styles.layerTag, color: layerColor, borderColor: layerColor + '44' }}>
            {layer}
          </span>
        </div>
        <div style={styles.title}>{title}</div>
        <div style={styles.subtitle}>{subtitle}</div>
      </div>

      {/* Status indicator */}
      <div style={styles.statusRow}>
        <span
          className={status === 'pending' ? 'pulse-dot' : undefined}
          style={{ ...styles.statusDot, background: dotColor }}
        />
        <span style={{ ...styles.statusText, color: status === 'idle' ? 'var(--text-muted)' : 'var(--text)' }}>
          {statusLabel}
        </span>
      </div>

      <div style={styles.divider} />

      {/* Body (module-specific content) */}
      <div style={styles.body}>{children}</div>
    </div>
  )
}

const styles = {
  card: {
    background: 'var(--surface)',
    borderRadius: 12,
    display: 'flex',
    flexDirection: 'column',
    overflow: 'hidden',
    transition: 'border-color 0.3s, box-shadow 0.3s',
    minHeight: 0,
  },
  header: { padding: '16px 16px 10px' },
  headerTop: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 },
  moduleId: { fontSize: 11, fontWeight: 700, fontFamily: 'var(--mono)', letterSpacing: '0.1em' },
  layerTag: {
    fontSize: 9, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.08em',
    border: '1px solid', borderRadius: 4, padding: '2px 6px',
  },
  title:    { fontSize: 15, fontWeight: 700, color: 'var(--text)', marginBottom: 2 },
  subtitle: { fontSize: 11, color: 'var(--text-dim)' },
  statusRow: { display: 'flex', alignItems: 'center', gap: 6, padding: '6px 16px' },
  statusDot: { width: 7, height: 7, borderRadius: '50%', flexShrink: 0, transition: 'background 0.3s' },
  statusText: { fontSize: 12, fontWeight: 600 },
  divider: { height: 1, background: 'var(--border)', margin: '0 16px' },
  body: { flex: 1, padding: 16, display: 'flex', flexDirection: 'column', gap: 10 },
}
