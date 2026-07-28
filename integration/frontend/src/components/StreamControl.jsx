export default function StreamControl({ streamState, onStart, onStop, onReset, hint }) {
  const running = streamState === 'running'
  const done = streamState === 'complete' || streamState === 'error' || streamState === 'stopped'

  return (
    <div style={styles.row}>
      {!running ? (
        <button
          style={{ ...styles.btn, ...styles.btnPrimary }}
          onClick={onStart}
        >
          ▶ Start Stream
        </button>
      ) : (
        <button
          style={{ ...styles.btn, ...styles.btnStop }}
          onClick={onStop}
        >
          ■ Stop Stream
        </button>
      )}
      {done && (
        <button style={{ ...styles.btn, ...styles.btnSecondary }} onClick={onReset}>
          ↺ Reset
        </button>
      )}
      {hint && <span style={styles.hint}>{hint}</span>}
    </div>
  )
}

const styles = {
  row: { display: 'flex', alignItems: 'center', gap: 12 },
  btn: {
    padding: '9px 20px',
    borderRadius: 8,
    border: 'none',
    fontFamily: 'var(--sans)',
    fontSize: 14,
    fontWeight: 600,
    cursor: 'pointer',
    transition: 'opacity 0.15s, transform 0.1s',
    whiteSpace: 'nowrap',
  },
  btnPrimary: {
    background: 'var(--accent)',
    color: '#fff',
  },
  btnStop: {
    background: 'var(--red)',
    color: '#fff',
  },
  btnSecondary: {
    background: 'var(--surface2)',
    color: 'var(--text)',
    border: '1px solid var(--border)',
  },
  hint: { fontSize: 12, color: 'var(--text-dim)' },
}
