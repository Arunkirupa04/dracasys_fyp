const TYPE_COLORS = {
  info:    'var(--text-dim)',
  success: 'var(--green)',
  alert:   'var(--red)',
  error:   'var(--yellow)',
}

export default function EventLog({ entries }) {
  if (entries.length === 0) return null

  return (
    <div style={styles.wrap}>
      <div style={styles.title}>Event Log</div>
      <div style={styles.scroll}>
        {entries.map((e, i) => (
          <div key={i} style={styles.row}>
            <span style={styles.ts}>{e.ts}</span>
            <span style={{ ...styles.msg, color: TYPE_COLORS[e.type] || 'var(--text-dim)' }}>
              {e.msg}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}

const styles = {
  wrap: {
    background: 'var(--surface)',
    border: '1px solid var(--border)',
    borderRadius: 10,
    overflow: 'hidden',
  },
  title: {
    padding: '10px 16px', fontSize: 11, fontWeight: 700,
    textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--text-dim)',
    borderBottom: '1px solid var(--border)', background: 'var(--surface2)',
  },
  scroll: {
    maxHeight: 160, overflowY: 'auto',
    padding: '8px 0',
  },
  row: {
    display: 'flex', gap: 12, padding: '3px 16px',
    fontFamily: 'var(--mono)', fontSize: 12,
  },
  ts:  { color: 'var(--text-muted)', flexShrink: 0, width: 80 },
  msg: { flex: 1 },
}
