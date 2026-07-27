const LABELS = ['Normal', 'Normal', 'Anomaly', 'Anomaly']

export default function SampleTimeline({ total, current, streamState }) {
  if (streamState === 'idle') return null

  return (
    <div style={styles.wrap}>
      <span style={styles.title}>Progress</span>
      <div style={styles.track}>
        {Array.from({ length: total }, (_, i) => {
          const num = i + 1
          const done = current >= num
          const active = current === num - 1 && streamState === 'running'
          const isAnom = i >= 2   // samples 3 and 4 are anomalous
          return (
            <div key={i} style={styles.stepWrap}>
              <div
                style={{
                  ...styles.dot,
                  ...(done || active ? (isAnom ? styles.dotAlert : styles.dotDone) : styles.dotPending),
                  ...(active ? styles.dotPulse : {}),
                }}
              >
                {done ? (isAnom ? '!' : '✓') : num}
              </div>
              <span style={{ ...styles.label, color: isAnom ? 'var(--red)' : 'var(--text-dim)' }}>
                {LABELS[i]}
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )
}

const styles = {
  wrap: { display: 'flex', alignItems: 'center', gap: 14, flex: 1 },
  title: { fontSize: 12, fontWeight: 600, color: 'var(--text-dim)', textTransform: 'uppercase', letterSpacing: '0.06em', whiteSpace: 'nowrap' },
  track: { display: 'flex', gap: 24, alignItems: 'center' },
  stepWrap: { display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4 },
  dot: {
    width: 32, height: 32, borderRadius: '50%',
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    fontSize: 13, fontWeight: 700, transition: 'all 0.3s',
  },
  dotPending: { background: 'var(--surface2)', color: 'var(--text-dim)', border: '1px solid var(--border)' },
  dotDone:    { background: 'var(--green)', color: '#fff' },
  dotAlert:   { background: 'var(--red)', color: '#fff' },
  dotPulse:   { boxShadow: '0 0 0 4px rgba(59,130,246,0.3)', border: '2px solid var(--accent)' },
  label:      { fontSize: 10, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em' },
}
