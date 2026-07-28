/**
 * Clickable sample progress track — scrub through completed samples.
 * completedSamples: Set of 1-based sample numbers that have finished.
 */
export default function SampleTimeline({
  total,
  current,
  viewedSample,
  anomalySamples = new Set(),
  streamState,
  onSelectSample,
  followLive = true,
  onFollowLive,
}) {
  if (streamState === 'idle') return null

  const maxNavigable = current
  const canPrev = viewedSample > 1 && maxNavigable >= 1
  const canNext = viewedSample < maxNavigable
  const showLiveBtn = !followLive && streamState === 'running'

  return (
    <div style={styles.wrap}>
      <span style={styles.title}>Samples</span>

      <button
        type="button"
        style={{ ...styles.navBtn, opacity: canPrev ? 1 : 0.35, cursor: canPrev ? 'pointer' : 'default' }}
        onClick={() => canPrev && onSelectSample?.(viewedSample - 1)}
        disabled={!canPrev}
        title="Previous sample"
      >
        ‹
      </button>

      <div style={styles.track}>
        {Array.from({ length: total }, (_, i) => {
          const num = i + 1
          const done = current >= num
          const active = streamState === 'running' && current === num - 1 && !done
          const isAnom = anomalySamples.has(num)
          const selected = done && viewedSample === num
          const clickable = done && !!onSelectSample

          return (
            <button
              key={i}
              type="button"
              disabled={!clickable}
              onClick={() => clickable && onSelectSample(num)}
              title={
                done
                  ? `View sample ${num}${isAnom ? ' (anomaly)' : ''}`
                  : `Sample ${num} — not yet streamed`
              }
              style={{
                ...styles.dot,
                ...(done ? (isAnom ? styles.dotAlert : styles.dotDone) : styles.dotPending),
                ...(active ? styles.dotPulse : {}),
                ...(selected ? styles.dotSelected : {}),
                cursor: clickable ? 'pointer' : 'default',
              }}
            >
              {done ? (isAnom ? '!' : num) : num}
            </button>
          )
        })}
      </div>

      <button
        type="button"
        style={{ ...styles.navBtn, opacity: canNext ? 1 : 0.35, cursor: canNext ? 'pointer' : 'default' }}
        onClick={() => canNext && onSelectSample?.(viewedSample + 1)}
        disabled={!canNext}
        title="Next sample"
      >
        ›
      </button>

      {maxNavigable > 0 && (
        <span style={styles.viewLabel}>
          Viewing <strong style={{ color: 'var(--accent)' }}>#{viewedSample || '—'}</strong>
          {followLive && streamState === 'running' ? ' · live' : ''}
        </span>
      )}

      {showLiveBtn && (
        <button type="button" style={styles.liveBtn} onClick={onFollowLive}>
          Jump to live
        </button>
      )}
    </div>
  )
}

const styles = {
  wrap: { display: 'flex', alignItems: 'center', gap: 10, flex: 1, minWidth: 0 },
  title: {
    fontSize: 12, fontWeight: 600, color: 'var(--text-dim)',
    textTransform: 'uppercase', letterSpacing: '0.06em', whiteSpace: 'nowrap',
  },
  track: { display: 'flex', gap: 6, alignItems: 'center', overflowX: 'auto', paddingBottom: 2 },
  navBtn: {
    width: 28, height: 28, borderRadius: 6, border: '1px solid var(--border)',
    background: 'var(--surface2)', color: 'var(--text)', fontSize: 18, fontWeight: 700,
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    fontFamily: 'inherit', lineHeight: 1, flexShrink: 0,
  },
  dot: {
    width: 28, height: 28, borderRadius: '50%',
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    fontSize: 11, fontWeight: 700, transition: 'all 0.2s',
    border: '2px solid transparent', flexShrink: 0, padding: 0, fontFamily: 'inherit',
  },
  dotPending: { background: 'var(--surface2)', color: 'var(--text-dim)', borderColor: 'var(--border)' },
  dotDone:    { background: 'var(--green)', color: '#fff' },
  dotAlert:   { background: 'var(--red)', color: '#fff' },
  dotPulse:   { boxShadow: '0 0 0 4px rgba(59,130,246,0.3)', borderColor: 'var(--accent)' },
  dotSelected: {
    boxShadow: '0 0 0 3px rgba(59,130,246,0.45)',
    borderColor: 'var(--accent)',
    transform: 'scale(1.12)',
  },
  viewLabel: { fontSize: 12, color: 'var(--text-dim)', whiteSpace: 'nowrap', marginLeft: 4 },
  liveBtn: {
    padding: '4px 10px', borderRadius: 6, border: '1px solid var(--accent)',
    background: 'rgba(59,130,246,0.12)', color: 'var(--accent)',
    fontSize: 11, fontWeight: 600, cursor: 'pointer', fontFamily: 'inherit', whiteSpace: 'nowrap',
  },
}
