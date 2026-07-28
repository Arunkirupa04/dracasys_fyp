import ModuleCard from './ModuleCard'
import { ScoreTrendChart, AnomalyTimelineChart } from './charts'
import { moduleConfig } from '../hooks/useConfig'

export default function Module4Card({ data, history = [], config }) {
  const mod = moduleConfig(config, 'm4')
  const result = data.result
  const isAnom = result?.is_anomaly ?? false
  const score = result?.score
  const threshold = result?.threshold ?? parseFloat(mod?.validation_metrics?.find((m) => m.label.includes('Threshold'))?.value || '-0.3475')
  const attackType = result?.attack_type

  const fillPct = score != null
    ? Math.min(100, Math.max(0, ((score - (threshold - 0.5)) / 1.0) * 100))
    : 0
  const barColor = isAnom ? 'var(--red)' : 'var(--green)'

  const timelinePoints = history.map((r) => ({
    sample: r.sample,
    isAnomaly: r.is_anomaly,
  }))

  return (
    <ModuleCard
      moduleId={mod.id}
      title={mod.title}
      subtitle={mod.subtitle}
      layer={config?.ui?.layers?.anomaly?.label}
      layerColor="var(--red)"
      status={data.status}
      isAnomaly={isAnom}
    >
      {result == null ? (
        <Placeholder mod={mod} />
      ) : (
        <>
          <div style={styles.metricsRow}>
            <Metric label="Status" value={isAnom ? 'THREAT' : 'Normal'} alert={isAnom} />
            <Metric label="Anomaly Score" value={score?.toFixed(5)} highlight={isAnom} />
            <Metric label="Threshold" value={threshold.toFixed(5)} />
            <Metric label="Sample" value={`#${result.sample}`} />
          </div>

          <div style={{ ...styles.alertBadge, background: isAnom ? 'rgba(239,68,68,0.15)' : 'rgba(34,197,94,0.12)', borderColor: isAnom ? 'rgba(239,68,68,0.4)' : 'rgba(34,197,94,0.3)', color: isAnom ? 'var(--red)' : 'var(--green)' }}>
            <span>{isAnom ? '🚨 Security Threat Detected' : '✓ Normal Traffic'}</span>
          </div>

          {isAnom && attackType && (
            <div style={styles.attackBadge}>
              <span style={styles.attackLabel}>Attack type</span>
              <span style={styles.attackName}>{attackType}</span>
            </div>
          )}

          <div style={styles.scoreSection}>
            <div style={styles.scoreHeader}>
              <span style={styles.scoreLabel}>Score vs Threshold</span>
              <span style={{ ...styles.scoreVal, color: barColor }}>{score?.toFixed(5)}</span>
            </div>
            <div style={styles.barTrack}>
              <div style={{ ...styles.barFill, width: `${fillPct}%`, background: barColor }} />
            </div>
            <div style={styles.thresholdLabel}>
              Threshold: {threshold.toFixed(5)} · Higher score = more anomalous
            </div>
          </div>

          <div style={styles.chartGrid}>
            {history.length > 0 && (
              <ScoreTrendChart
                title="Anomaly Score Across Streamed Samples"
                yLabel="Anomaly Score"
                points={history.map((r) => ({ sample: r.sample, value: r.score, isAnomaly: r.is_anomaly }))}
                threshold={threshold}
                scale="linear"
                width={640}
                height={260}
              />
            )}
            {timelinePoints.length > 0 && (
              <AnomalyTimelineChart
                title="Security Alert Timeline"
                points={timelinePoints}
                width={520}
                height={100}
              />
            )}
          </div>
        </>
      )}
    </ModuleCard>
  )
}

function Metric({ label, value, highlight, alert }) {
  const color = alert ? 'var(--red)' : highlight ? 'var(--accent)' : 'var(--text)'
  return (
    <div style={styles.metric}>
      <div style={{ ...styles.metricVal, color }}>{value}</div>
      <div style={styles.metricLabel}>{label}</div>
    </div>
  )
}

function Placeholder({ mod }) {
  return (
    <div style={styles.placeholder}>
      <div style={styles.phIcon}>🔒</div>
      <div style={styles.phText}>Waiting for network flow data</div>
      <div style={styles.phSub}>Window: {mod?.inputs?.window_shape} · {mod?.inputs?.source}</div>
    </div>
  )
}

const styles = {
  metricsRow: { display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12, marginBottom: 16 },
  metric: { background: 'var(--surface2)', borderRadius: 8, padding: '12px 14px', border: '1px solid var(--border)' },
  metricVal: { fontSize: 18, fontWeight: 700, fontFamily: 'var(--mono)' },
  metricLabel: { fontSize: 10, color: 'var(--text-muted)', marginTop: 4, textTransform: 'uppercase' },
  alertBadge: {
    display: 'flex', alignItems: 'center', gap: 8,
    padding: '12px 16px', borderRadius: 8, border: '1px solid',
    fontSize: 14, fontWeight: 700, marginBottom: 12,
  },
  attackBadge: {
    background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.25)',
    borderRadius: 8, padding: '10px 14px', marginBottom: 16,
  },
  attackLabel: { fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--red)', fontWeight: 700 },
  attackName: { fontSize: 14, fontWeight: 600, color: 'var(--text)', marginTop: 4 },
  scoreSection: { marginBottom: 16 },
  scoreHeader: { display: 'flex', justifyContent: 'space-between', marginBottom: 6 },
  scoreLabel: { fontSize: 11, color: 'var(--text-dim)', fontWeight: 600, textTransform: 'uppercase' },
  scoreVal: { fontSize: 14, fontFamily: 'var(--mono)', fontWeight: 700 },
  barTrack: { height: 10, background: 'var(--surface2)', borderRadius: 5, overflow: 'hidden' },
  barFill: { height: '100%', borderRadius: 5, transition: 'width 0.6s ease' },
  thresholdLabel: { marginTop: 6, fontSize: 11, color: 'var(--text-muted)' },
  chartGrid: { display: 'grid', gridTemplateColumns: '1fr', gap: 12 },
  placeholder: {
    display: 'flex', flexDirection: 'column', alignItems: 'center',
    justifyContent: 'center', gap: 8, textAlign: 'center', padding: '40px 8px',
  },
  phIcon: { fontSize: 28, opacity: 0.35 },
  phText: { fontSize: 13, fontWeight: 600, color: 'var(--text-dim)' },
  phSub: { fontSize: 11, color: 'var(--text-muted)' },
}
