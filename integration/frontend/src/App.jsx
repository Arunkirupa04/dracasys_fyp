import { useState, useCallback, useRef } from 'react'
import { startStream } from './api/sse_client'
import StreamControl from './components/StreamControl'
import SampleTimeline from './components/SampleTimeline'
import SystemStatus from './components/SystemStatus'
import Module1Card from './components/Module1Card'
import Module2Card from './components/Module2Card'
import Module3Card from './components/Module3Card'
import Module4Card from './components/Module4Card'
import EventLog from './components/EventLog'

const TOTAL_SAMPLES = 4

const initModuleState = () => ({ status: 'idle', result: null, sampleIndex: null })

export default function App() {
  const [streamState, setStreamState] = useState('idle') // idle | running | complete | error
  const [currentSample, setCurrentSample] = useState(0)
  const [m1, setM1] = useState(initModuleState())
  const [m2, setM2] = useState(initModuleState())
  const [m3, setM3] = useState(initModuleState())
  const [m4, setM4] = useState(initModuleState())
  const [log, setLog] = useState([])
  const cleanupRef = useRef(null)

  const appendLog = useCallback((msg, type = 'info') => {
    setLog(prev => [...prev, { msg, type, ts: new Date().toLocaleTimeString() }])
  }, [])

  const handleEvent = useCallback((data) => {
    const { event } = data

    if (event === 'stream_start') {
      appendLog('Stream started — processing 4 samples', 'info')
      return
    }

    if (event === 'sample_complete') {
      setCurrentSample(data.sample)
      appendLog(`Sample ${data.sample} processed`, 'info')
      return
    }

    if (event === 'stream_complete') {
      setStreamState('complete')
      appendLog('Stream complete — all 4 samples processed', 'success')
      return
    }

    if (event === 'module_result') {
      const sample = data.sample
      const setter = { m1: setM1, m2: setM2, m3: setM3, m4: setM4 }[data.module]
      if (!setter) return

      setter({ status: 'ready', result: data, sampleIndex: sample })

      // Log anomalies
      if (data.module === 'm3' && data.is_anomaly) {
        appendLog(`[Sample ${sample}] M3 — System anomaly detected! MSE=${data.reconstruction_mse}`, 'alert')
      }
      if (data.module === 'm4' && data.is_anomaly) {
        appendLog(`[Sample ${sample}] M4 — Security alert: ${data.attack_type}`, 'alert')
      }
    }
  }, [appendLog])

  const handleStart = useCallback(() => {
    // Reset state
    setStreamState('running')
    setCurrentSample(0)
    setM1(initModuleState())
    setM2(initModuleState())
    setM3(initModuleState())
    setM4(initModuleState())
    setLog([])

    const cleanup = startStream(handleEvent, (err) => {
      setStreamState('error')
      appendLog('Stream connection error — is the backend running?', 'error')
    })
    cleanupRef.current = cleanup
  }, [handleEvent, appendLog])

  const handleReset = useCallback(() => {
    if (cleanupRef.current) { cleanupRef.current(); cleanupRef.current = null }
    setStreamState('idle')
    setCurrentSample(0)
    setM1(initModuleState())
    setM2(initModuleState())
    setM3(initModuleState())
    setM4(initModuleState())
    setLog([])
  }, [])

  // Determine overall anomaly status
  const anyAnomaly = (m3.result?.is_anomaly || m4.result?.is_anomaly) && streamState !== 'idle'

  return (
    <div style={styles.root}>
      {/* Header */}
      <header style={styles.header}>
        <div style={styles.headerLeft}>
          <div style={styles.logo}>
            <span style={styles.logoIcon}>⬡</span>
            <span style={styles.logoText}>DracaSys</span>
          </div>
          <div style={styles.subtitle}>Intelligent Deployment Helper — Live Demo</div>
        </div>
        <SystemStatus anyAnomaly={anyAnomaly} streamState={streamState} />
      </header>

      {/* Controls + Timeline */}
      <div style={styles.controlsRow}>
        <StreamControl
          streamState={streamState}
          onStart={handleStart}
          onReset={handleReset}
        />
        <SampleTimeline
          total={TOTAL_SAMPLES}
          current={currentSample}
          m3Results={m3.result}
          m4Results={m4.result}
          streamState={streamState}
        />
      </div>

      {/* Architecture labels */}
      <div style={styles.layerRow}>
        <div style={styles.layerLabel}>
          <span style={{ color: 'var(--accent)' }}>◈</span> Prediction Layer
        </div>
        <div style={styles.layerLabel}>
          <span style={{ color: 'var(--red)' }}>◈</span> Anomaly Detection Layer
        </div>
      </div>

      {/* 4 module cards */}
      <div style={styles.grid}>
        <Module1Card data={m1} />
        <Module2Card data={m2} />
        <Module3Card data={m3} />
        <Module4Card data={m4} />
      </div>

      {/* Event log */}
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
  header: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 24,
    paddingBottom: 16,
    borderBottom: '1px solid var(--border)',
  },
  headerLeft: { display: 'flex', flexDirection: 'column', gap: 4 },
  logo: { display: 'flex', alignItems: 'center', gap: 10 },
  logoIcon: { fontSize: 26, color: 'var(--accent)' },
  logoText: { fontSize: 22, fontWeight: 700, letterSpacing: '-0.5px' },
  subtitle: { fontSize: 13, color: 'var(--text-dim)', paddingLeft: 36 },
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
    marginBottom: 12,
    fontSize: 12,
    fontWeight: 600,
    textTransform: 'uppercase',
    letterSpacing: '0.08em',
    color: 'var(--text-dim)',
  },
  layerLabel: { display: 'flex', alignItems: 'center', gap: 6 },
  grid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(4, 1fr)',
    gap: 16,
    marginBottom: 24,
  },
}
