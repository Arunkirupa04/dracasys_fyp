export default function StreamControl({ streamState, onStart, onReset }) {
  const running = streamState === 'running'
  const done    = streamState === 'complete' || streamState === 'error'

  return (
    <div style={styles.row}>
      <button
        style={{
          ...styles.btn,
          ...(running ? styles.btnDisabled : styles.btnPrimary),
        }}
        onClick={onStart}
        disabled={running}
      >
        {running ? '⏳ Streaming…' : '▶ Start Stream'}
      </button>
      {done && (
        <button style={{ ...styles.btn, ...styles.btnSecondary }} onClick={onReset}>
          ↺ Reset
        </button>
      )}
      <span style={styles.hint}>
        {streamState === 'idle'    && '4 samples · 5 s interval · 4 modules'}
        {streamState === 'running' && 'Streaming live data through all models…'}
        {streamState === 'complete'&& 'Stream complete. Click Reset to run again.'}
        {streamState === 'error'   && 'Connection error — ensure backend is running on :8000'}
      </span>
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
  btnSecondary: {
    background: 'var(--surface2)',
    color: 'var(--text)',
    border: '1px solid var(--border)',
  },
  btnDisabled: {
    background: 'var(--surface2)',
    color: 'var(--text-dim)',
    cursor: 'not-allowed',
  },
  hint: { fontSize: 12, color: 'var(--text-dim)' },
}
