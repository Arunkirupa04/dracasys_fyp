import ModuleInfoPanel from './ModuleInfoPanel'
import Module1Card from './Module1Card'
import Module2Card from './Module2Card'
import Module3Card from './Module3Card'
import Module4Card from './Module4Card'
import ModuleOutputsDisplay from './ModuleOutputsDisplay'
import StreamControl from './StreamControl'
import SampleTimeline from './SampleTimeline'
import { layerColor, moduleConfig } from '../hooks/useConfig'

const CARD_MAP = {
  m1: Module1Card,
  m2: Module2Card,
  m3: Module3Card,
  m4: Module4Card,
}

export default function ModuleDetailView({
  moduleKey, config, onBack,
  m1, m2, m3, m4, m2History, m3History, m4History,
  streamState, totalSamples, currentSample, anomalySamples,
  onStart, onStop, onReset, hint,
  viewedSample, onSelectSample, followLive, onFollowLive,
}) {
  const mod = moduleConfig(config, moduleKey)
  const layerKey = mod.layer || 'prediction'
  const Card = CARD_MAP[moduleKey]

  const dataMap = { m1, m2, m3, m4 }
  const historyMap = { m2: m2History, m3: m3History, m4: m4History }
  const moduleData = dataMap[moduleKey]

  return (
    <div style={styles.wrap}>
      <div style={styles.topBar}>
        <button type="button" onClick={onBack} className="back-dashboard-btn" style={styles.backBtn}>
          <span style={styles.backIcon}>←</span>
          <span>Back to Dashboard</span>
        </button>
      </div>

      <div style={styles.streamRow}>
        <StreamControl
          streamState={streamState}
          onStart={onStart}
          onStop={onStop}
          onReset={onReset}
          hint={hint}
        />
        <SampleTimeline
          total={totalSamples}
          current={currentSample}
          viewedSample={viewedSample}
          anomalySamples={anomalySamples}
          streamState={streamState}
          onSelectSample={onSelectSample}
          followLive={followLive}
          onFollowLive={onFollowLive}
        />
      </div>

      <div style={styles.header}>
        <div>
          <span style={{ ...styles.badge, color: layerColor(config, layerKey), borderColor: layerColor(config, layerKey) + '44' }}>
            {mod.id} · {config?.ui?.layers?.[layerKey]?.label}
          </span>
          <h1 style={styles.title}>{mod.title}</h1>
          <p style={styles.subtitle}>
            {mod.subtitle}
            {viewedSample > 0 ? ` · viewing sample #${viewedSample}` : ''}
          </p>
        </div>
        <div style={styles.artifact}>
          <span style={styles.artifactLabel}>Model</span>
          <span style={styles.artifactValue}>{mod.model_artifact}</span>
        </div>
      </div>

      <div style={styles.livePanel}>
        <ModuleOutputsDisplay
          moduleKey={moduleKey}
          data={moduleData}
          config={config}
          viewedSample={viewedSample}
        />
      </div>

      <ModuleInfoPanel mod={mod} />

      <div style={styles.sectionTitle}>Charts & detailed metrics</div>
      <div style={styles.cardWrap}>
        <Card
          data={moduleData}
          history={historyMap[moduleKey]}
          config={config}
        />
      </div>
    </div>
  )
}

const styles = {
  wrap: { maxWidth: 1100, margin: '0 auto' },
  topBar: { marginBottom: 16 },
  backBtn: {
    display: 'inline-flex', alignItems: 'center', gap: 10,
    background: 'var(--surface2)', border: '2px solid var(--accent)',
    color: 'var(--text)', cursor: 'pointer',
    fontSize: 15, fontWeight: 700, padding: '12px 20px',
    borderRadius: 10, fontFamily: 'inherit',
    boxShadow: '0 4px 14px rgba(59, 130, 246, 0.2)',
    transition: 'background 0.15s, transform 0.1s',
  },
  backIcon: { fontSize: 20, color: 'var(--accent)', lineHeight: 1 },
  streamRow: {
    display: 'flex', alignItems: 'center', gap: 20, flexWrap: 'wrap',
    marginBottom: 20, padding: '14px 16px',
    background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 10,
  },
  header: {
    display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start',
    marginBottom: 16, paddingBottom: 16, borderBottom: '1px solid var(--border)',
  },
  badge: {
    fontSize: 10, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.08em',
    border: '1px solid', borderRadius: 4, padding: '3px 8px', display: 'inline-block', marginBottom: 8,
  },
  title: { fontSize: 26, fontWeight: 700, marginBottom: 4 },
  subtitle: { fontSize: 14, color: 'var(--text-dim)' },
  artifact: { textAlign: 'right' },
  artifactLabel: { display: 'block', fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 4 },
  artifactValue: { fontSize: 13, fontFamily: 'var(--mono)', color: 'var(--text)' },
  livePanel: {
    background: 'var(--surface)', border: '1px solid var(--border)',
    borderRadius: 10, padding: '16px 20px', marginBottom: 20,
  },
  sectionTitle: {
    fontSize: 12, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.08em',
    color: 'var(--text-dim)', marginBottom: 12,
  },
  cardWrap: { marginBottom: 24 },
}
