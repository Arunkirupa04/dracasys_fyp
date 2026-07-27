import { getModuleOutputs } from '../utils/moduleOutputs'

export default function ModuleOutputsDisplay({ moduleKey, data, config, compact = false, viewedSample }) {
  const { rows, table } = getModuleOutputs(moduleKey, data, config)
  const hasData = data.status !== 'idle'

  if (!hasData) {
    return (
      <div style={styles.empty}>
        {viewedSample
          ? `No output for sample #${viewedSample} yet`
          : 'No output yet — start the stream'}
      </div>
    )
  }

  return (
    <div style={compact ? styles.wrapCompact : styles.wrap}>
      <div style={styles.sectionLabel}>
        {viewedSample ? `Outputs · sample #${viewedSample}` : 'Live outputs'}
      </div>
      <div style={styles.grid}>
        {rows.map(({ label, value, highlight, alert }) => (
          <div key={label} style={styles.row}>
            <span style={styles.label}>{label}</span>
            <span style={{
              ...styles.value,
              color: alert ? 'var(--red)' : highlight ? 'var(--accent)' : 'var(--text)',
              fontWeight: alert || highlight ? 700 : 500,
            }}>
              {value}
            </span>
          </div>
        ))}
      </div>

      {table && (
        <div style={styles.tableWrap}>
          <table style={styles.table}>
            <thead>
              <tr>
                {table.headers.map((h) => (
                  <th key={h} style={styles.th}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {table.rows.map((row) => (
                <tr key={row[0]}>
                  {row.map((cell, i) => (
                    <td key={i} style={i === 0 ? styles.tdLabel : styles.td}>{cell}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

const styles = {
  wrap: { flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0 },
  wrapCompact: { flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0 },
  empty: { fontSize: 11, color: 'var(--text-muted)', fontStyle: 'italic', padding: '8px 0' },
  sectionLabel: {
    fontSize: 9, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.08em',
    color: 'var(--text-muted)', marginBottom: 8,
  },
  grid: {
    display: 'flex', flexDirection: 'column', gap: 4,
    maxHeight: 280, overflowY: 'auto', paddingRight: 4,
  },
  row: {
    display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', gap: 8,
    fontSize: 11, lineHeight: 1.4,
  },
  label: { color: 'var(--text-dim)', flexShrink: 0 },
  value: { fontFamily: 'var(--mono)', textAlign: 'right', wordBreak: 'break-word' },
  tableWrap: { marginTop: 10, overflowX: 'auto' },
  table: { width: '100%', borderCollapse: 'collapse', fontSize: 10 },
  th: {
    textAlign: 'right', padding: '4px 6px', color: 'var(--accent)', fontWeight: 700,
    borderBottom: '1px solid var(--border)', textTransform: 'uppercase',
  },
  tdLabel: { textAlign: 'left', padding: '4px 6px', color: 'var(--text-dim)', borderBottom: '1px solid var(--border)' },
  td: {
    textAlign: 'right', padding: '4px 6px', fontFamily: 'var(--mono)', color: 'var(--text)',
    borderBottom: '1px solid var(--border)',
  },
}
