import ModuleCard from './ModuleCard'
import { ActualPredictedChart, MultiLineChart } from './charts'
import { moduleConfig } from '../hooks/useConfig'

export default function Module1Card({ data, config }) {
  const mod = moduleConfig(config, 'm1')
  const result = data.result
  const resultStatus = result?.status

  return (
    <ModuleCard
      moduleId={mod.id}
      title={mod.title}
      subtitle={mod.subtitle}
      layer={config?.ui?.layers?.prediction?.label}
      layerColor="var(--accent)"
      status={data.status}
    >
      {!result || resultStatus === 'pending' ? (
        <Pending message={result?.message} config={config} />
      ) : resultStatus === 'error' ? (
        <ErrorState error={result.error} port={config?.services?.module1_port} />
      ) : (
        <Ready result={result} mod={mod} />
      )}
    </ModuleCard>
  )
}

function Pending({ message, config }) {
  const steps = config?.modules?.m1?.inputs?.recommended_history_steps
  return (
    <div style={styles.placeholder}>
      <div className="spinner" />
      <div style={styles.phText}>Forecasting next 24 hours…</div>
      <div style={styles.phSub}>
        {message || `Prophet + GRU refit on ${steps} steps of history (module1 service)`}
      </div>
    </div>
  )
}

function ErrorState({ error, port }) {
  return (
    <div style={styles.placeholder}>
      <div style={styles.phIcon}>⚠</div>
      <div style={{ ...styles.phText, color: 'var(--yellow)' }}>Module 1 unreachable</div>
      <div style={styles.phSub}>{error || `Start the service on port ${port}`}</div>
    </div>
  )
}

function Ready({ result, mod }) {
  const predicted = result.predicted_cpu_percent || []
  const actual = result.history_tail_cpu_percent || []
  const prophet = result.prophet_component_percent || []
  const gru = result.gru_residual_component_percent || []
  const horizon = mod?.inputs?.forecast_horizon_steps ?? result.horizon_steps
  const last = predicted[predicted.length - 1]
  const first = predicted[0]

  return (
    <>
      <div style={styles.metricsRow}>
        <Metric label="24h forecast end" value={`${last?.toFixed(1)}%`} highlight />
        <Metric label="Forecast start" value={`${first?.toFixed(1)}%`} />
        <Metric label="Container" value={result.container_id} />
        <Metric label="Latency" value={`${(result.processing_time_ms / 1000).toFixed(1)}s`} />
      </div>

      <div className="detail-chart-grid" style={styles.chartGrid}>
        <ActualPredictedChart
          title="CPU Utilization — Actual vs Predicted"
          yLabel="CPU %"
          unit="%"
          actual={actual}
          predicted={predicted}
          actualLabel="Actual (recent history)"
          predictedLabel={`Predicted (${horizon} steps)`}
          width={640}
          height={260}
        />
        <MultiLineChart
          title="Forecast Decomposition — Prophet + GRU Residual"
          yLabel="Component %"
          unit="%"
          series={[
            { label: 'Prophet trend/seasonality', data: prophet, color: 'var(--purple)' },
            { label: 'GRU residual correction', data: gru, color: 'var(--yellow)', dashed: true },
            { label: 'Combined forecast', data: predicted, color: 'var(--accent)' },
          ]}
          width={640}
          height={260}
        />
      </div>

      <div style={styles.metaGrid}>
        <MetaItem label="Horizon" value={`${result.horizon_steps} × 15 min`} />
        <MetaItem label="History used" value={`${result.history_steps_used} steps`} />
        <MetaItem label="Scaler mode" value={result.scaler_mode} />
        <MetaItem label="Model" value={result.model_version} />
      </div>
    </>
  )
}

function Metric({ label, value, highlight }) {
  return (
    <div style={styles.metric}>
      <div style={{ ...styles.metricVal, color: highlight ? 'var(--accent)' : 'var(--text)' }}>{value}</div>
      <div style={styles.metricLabel}>{label}</div>
    </div>
  )
}

function MetaItem({ label, value }) {
  return (
    <div style={styles.metaItem}>
      <span style={styles.metaLabel}>{label}</span>
      <span style={styles.metaValue}>{value}</span>
    </div>
  )
}

const styles = {
  placeholder: {
    display: 'flex', flexDirection: 'column', alignItems: 'center',
    justifyContent: 'center', gap: 10, padding: '40px 12px', textAlign: 'center',
  },
  phIcon: { fontSize: 28, color: 'var(--yellow)' },
  phText: { fontSize: 13, fontWeight: 600, color: 'var(--text-dim)' },
  phSub:  { fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.5 },
  metricsRow: { display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12, marginBottom: 16 },
  metric: { background: 'var(--surface2)', borderRadius: 8, padding: '12px 14px', border: '1px solid var(--border)' },
  metricVal: { fontSize: 20, fontWeight: 700, fontFamily: 'var(--mono)' },
  metricLabel: { fontSize: 10, color: 'var(--text-muted)', marginTop: 4, textTransform: 'uppercase', letterSpacing: '0.05em' },
  chartGrid: { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 },
  metaGrid: { display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 8, marginTop: 16 },
  metaItem: { display: 'flex', flexDirection: 'column', gap: 2 },
  metaLabel: { fontSize: 9, textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--text-muted)' },
  metaValue: { fontSize: 11, fontFamily: 'var(--mono)', color: 'var(--text)' },
}
