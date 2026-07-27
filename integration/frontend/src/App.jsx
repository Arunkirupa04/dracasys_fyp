import { useState, useCallback, useRef } from 'react'
import { startStream } from './api/sse_client'
import { useConfig, formatStreamHint } from './hooks/useConfig'
import { dataAtSample, historyUpToSample } from './utils/sampleView'
import StreamControl from './components/StreamControl'
import SampleTimeline from './components/SampleTimeline'
import SystemStatus from './components/SystemStatus'
import ModuleSummaryCard from './components/ModuleSummaryCard'
import ModuleDetailView from './components/ModuleDetailView'
import EventLog from './components/EventLog'

const initModuleState = () => ({ status: 'idle', result: null, sampleIndex: null })

const mapCardStatus = (resultStatus) => {
  if (resultStatus === 'ok') return 'ready'
  if (resultStatus === 'error') return 'error'
  if (resultStatus === 'pending') return 'pending'
  return 'ready'
}

export default function App() {
  const { config, loading } = useConfig()
  const [selectedModule, setSelectedModule] = useState(null)
  const [streamState, setStreamState] = useState('idle')
  const [totalSamples, setTotalSamples] = useState(config?.demo?.n_demo_samples ?? 10)
  const [currentSample, setCurrentSample] = useState(0)
  const [viewedSample, setViewedSample] = useState(0)
  const [followLive, setFollowLive] = useState(true)
  const [m1, setM1] = useState(initModuleState())
  const [m2, setM2] = useState(initModuleState())
  const [m3, setM3] = useState(initModuleState())
  const [m4, setM4] = useState(initModuleState())
  const [m2History, setM2History] = useState([])
  const [m3History, setM3History] = useState([])
  const [m4History, setM4History] = useState([])
  const [log, setLog] = useState([])
  const cleanupRef = useRef(null)
  const followLiveRef = useRef(true)

  const appendLog = useCallback((msg, type = 'info') => {
    setLog(prev => [...prev, { msg, type, ts: new Date().toLocaleTimeString() }])
  }, [])

  const handleEvent = useCallback((data) => {
    const { event } = data

    if (event === 'stream_start') {
      const n = data.total_samples || config?.demo?.n_demo_samples || 10
      setTotalSamples(n)
      appendLog(`Stream started — processing ${n} samples`, 'info')
      return
    }

    if (event === 'sample_complete') {
      setCurrentSample(data.sample)
      if (followLiveRef.current) {
        setViewedSample(data.sample)
      }
      appendLog(`Sample ${data.sample} processed`, 'info')
      return
    }

    if (event === 'stream_complete') {
      if (cleanupRef.current) { cleanupRef.current(); cleanupRef.current = null }
      setStreamState('complete')
      setFollowLive(true)
      followLiveRef.current = true
      appendLog('Stream complete — click any sample to review outputs', 'success')
      return
    }

    if (event === 'module_result') {
      const sample = data.sample
      const setter = { m1: setM1, m2: setM2, m3: setM3, m4: setM4 }[data.module]
      if (!setter) return

      setter({ status: mapCardStatus(data.status), result: data, sampleIndex: sample })

      if (data.status === 'ok') {
        if (data.module === 'm2') setM2History(prev => [...prev, data])
        if (data.module === 'm3') setM3History(prev => [...prev, data])
        if (data.module === 'm4') setM4History(prev => [...prev, data])
      }

      if (data.module === 'm3' && data.is_anomaly) {
        appendLog(`[Sample ${sample}] M3 — System anomaly detected! MSE=${data.reconstruction_mse}`, 'alert')
      }
      if (data.module === 'm4' && data.is_anomaly) {
        appendLog(`[Sample ${sample}] M4 — Security alert: ${data.attack_type}`, 'alert')
      }
      if (data.module === 'm1' && data.status === 'ok') {
        appendLog(`M1 — 24h forecast ready for ${data.container_id} (${(data.processing_time_ms / 1000).toFixed(1)}s)`, 'success')
      }
      if (data.module === 'm1' && data.status === 'error') {
        appendLog(`M1 — ${data.error}`, 'error')
      }
    }
  }, [appendLog, config])

  const resetState = useCallback(() => {
    setCurrentSample(0)
    setViewedSample(0)
    setFollowLive(true)
    followLiveRef.current = true
    setM1(initModuleState())
    setM2(initModuleState())
    setM3(initModuleState())
    setM4(initModuleState())
    setM2History([])
    setM3History([])
    setM4History([])
    setLog([])
  }, [])

  const handleStart = useCallback(() => {
    setStreamState('running')
    resetState()

    const cleanup = startStream(handleEvent, () => {
      setStreamState(prev => {
        if (prev === 'complete' || prev === 'stopped' || prev === 'idle') return prev
        appendLog('Stream connection error — is the backend running?', 'error')
        return 'error'
      })
    })
    cleanupRef.current = cleanup
  }, [handleEvent, appendLog, resetState])

  const handleStop = useCallback(() => {
    if (cleanupRef.current) { cleanupRef.current(); cleanupRef.current = null }
    setStreamState('stopped')
    setFollowLive(false)
    followLiveRef.current = false
    appendLog('Stream stopped — click samples to review outputs', 'info')
  }, [appendLog])

  const handleReset = useCallback(() => {
    if (cleanupRef.current) { cleanupRef.current(); cleanupRef.current = null }
    setStreamState('idle')
    resetState()
  }, [resetState])

  const handleSelectSample = useCallback((num) => {
    setViewedSample(num)
    setFollowLive(false)
    followLiveRef.current = false
  }, [])

  const handleFollowLive = useCallback(() => {
    setFollowLive(true)
    followLiveRef.current = true
    setViewedSample(currentSample)
  }, [currentSample])

  const anyAnomaly = (m3.result?.is_anomaly || m4.result?.is_anomaly) && streamState !== 'idle'
  const streamActive = streamState === 'running' || streamState === 'complete' || streamState === 'stopped'

  const anomalySamples = new Set([
    ...m3History.filter(r => r.is_anomaly).map(r => r.sample),
    ...m4History.filter(r => r.is_anomaly).map(r => r.sample),
  ])

  // Display state for the currently viewed sample (scrubbing)
  const displayM1 = m1
  const displayM2 = dataAtSample('m2', m2, m2History, viewedSample, m1)
  const displayM3 = dataAtSample('m3', m3, m3History, viewedSample, m1)
  const displayM4 = dataAtSample('m4', m4, m4History, viewedSample, m1)
  const chartM2History = historyUpToSample(m2History, viewedSample)
  const chartM3History = historyUpToSample(m3History, viewedSample)
  const chartM4History = historyUpToSample(m4History, viewedSample)

  if (loading || !config) {
    return (
      <div style={styles.loading}>
        <div className="spinner" />
        <span>Loading configuration…</span>
      </div>
    )
  }

  const app = config.app || {}
  const streamHint = formatStreamHint(config, streamState, totalSamples)
  const moduleKeys = ['m1', 'm2', 'm3', 'm4']
  const displayData = { m1: displayM1, m2: displayM2, m3: displayM3, m4: displayM4 }

  const timelineProps = {
    total: totalSamples,
    current: currentSample,
    viewedSample,
    anomalySamples,
    streamState,
    onSelectSample: handleSelectSample,
    followLive,
    onFollowLive: handleFollowLive,
  }

  const streamProps = {
    streamState,
    totalSamples,
    currentSample,
    anomalySamples,
    onStart: handleStart,
    onStop: handleStop,
    onReset: handleReset,
    hint: streamHint,
    ...timelineProps,
  }

  if (selectedModule) {
    return (
      <div style={styles.root}>
        <header style={styles.headerCompact}>
          <div style={styles.logo}>
            <span style={styles.logoIcon}>⬡</span>
            <span style={styles.logoText}>{app.name}</span>
          </div>
          <SystemStatus anyAnomaly={anyAnomaly} streamState={streamState} />
        </header>
        <ModuleDetailView
          moduleKey={selectedModule}
          config={config}
          onBack={() => setSelectedModule(null)}
          m1={displayM1}
          m2={displayM2}
          m3={displayM3}
          m4={displayM4}
          m2History={chartM2History}
          m3History={chartM3History}
          m4History={chartM4History}
          viewedSample={viewedSample}
          {...streamProps}
        />
      </div>
    )
  }

  return (
    <div style={styles.root}>
      <header style={styles.header}>
        <div style={styles.headerLeft}>
          <div style={styles.logo}>
            <span style={styles.logoIcon}>⬡</span>
            <span style={styles.logoText}>{app.name}</span>
          </div>
          <div style={styles.subtitle}>{app.subtitle}</div>
          {app.description && <div style={styles.description}>{app.description}</div>}
        </div>
        <SystemStatus anyAnomaly={anyAnomaly} streamState={streamState} />
      </header>

      <div style={styles.controlsRow}>
        <StreamControl {...streamProps} />
        <SampleTimeline {...timelineProps} />
      </div>

      <div style={styles.layerRow}>
        <div style={styles.layerLabel}>
          <span style={{ color: 'var(--accent)' }}>◈</span> {config.ui?.layers?.prediction?.label}
        </div>
        <div style={styles.layerLabel}>
          <span style={{ color: 'var(--red)' }}>◈</span> {config.ui?.layers?.anomaly?.label}
        </div>
      </div>

      <p style={styles.hint}>
        {streamActive
          ? 'Click a completed sample (or use ‹ ›) to review past outputs. Open a module for charts.'
          : 'Start the stream to see all module outputs. Click a module for full charts and metrics.'}
      </p>

      <div className="dashboard-grid" style={styles.grid}>
        {moduleKeys.map((key) => {
          const d = displayData[key]
          const isAnomaly = (key === 'm3' && d.result?.is_anomaly) || (key === 'm4' && d.result?.is_anomaly)
          return (
            <ModuleSummaryCard
              key={key}
              moduleKey={key}
              config={config}
              data={d}
              isAnomaly={isAnomaly}
              streamActive={streamActive}
              viewedSample={viewedSample}
              onClick={() => setSelectedModule(key)}
            />
          )
        })}
      </div>

      <EventLog entries={log} />
    </div>
  )
}

const styles = {
  root: {
    maxWidth: 1400,
    margin: '0 auto',
    padding: '24px 20px 40px',
    minHeight: '100vh',
  },
  loading: {
    minHeight: '100vh', display: 'flex', flexDirection: 'column',
    alignItems: 'center', justifyContent: 'center', gap: 16, color: 'var(--text-dim)',
  },
  header: {
    display: 'flex',
    alignItems: 'flex-start',
    justifyContent: 'space-between',
    marginBottom: 24,
    paddingBottom: 16,
    borderBottom: '1px solid var(--border)',
  },
  headerCompact: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 16,
    paddingBottom: 12,
    borderBottom: '1px solid var(--border)',
  },
  headerLeft: { display: 'flex', flexDirection: 'column', gap: 4 },
  logo: { display: 'flex', alignItems: 'center', gap: 10 },
  logoIcon: { fontSize: 26, color: 'var(--accent)' },
  logoText: { fontSize: 22, fontWeight: 700, letterSpacing: '-0.5px' },
  subtitle: { fontSize: 13, color: 'var(--text-dim)', paddingLeft: 36 },
  description: { fontSize: 12, color: 'var(--text-muted)', paddingLeft: 36, maxWidth: 560, lineHeight: 1.5 },
  controlsRow: {
    display: 'flex',
    alignItems: 'center',
    gap: 20,
    marginBottom: 20,
    flexWrap: 'wrap',
  },
  layerRow: {
    display: 'flex',
    gap: 32,
    marginBottom: 8,
    fontSize: 12,
    fontWeight: 600,
    textTransform: 'uppercase',
    letterSpacing: '0.08em',
    color: 'var(--text-dim)',
  },
  layerLabel: { display: 'flex', alignItems: 'center', gap: 6 },
  hint: { fontSize: 12, color: 'var(--text-muted)', marginBottom: 16 },
  grid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(4, 1fr)',
    gap: 16,
    marginBottom: 24,
  },
}
