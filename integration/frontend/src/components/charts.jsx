/**
 * Professional SVG chart primitives — axes, grid, labels, legends.
 * Dependency-free; tuned for the dark dashboard theme.
 */

const CHART = {
  padLeft: 52,
  padRight: 16,
  padTop: 28,
  padBottom: 32,
}

/** Default viewBox — wider/taller so series is not edge-cropped. */
const DEFAULT_W = 640
const DEFAULT_H = 260

function clean(arr) {
  return (arr || []).filter((v) => typeof v === 'number' && !Number.isNaN(v))
}

/** Expand data min/max so the plot looks zoomed-out (breathing room). */
function padDomain(min, max, padFrac = 0.28) {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [0, 1]
  if (min === max) {
    const d = Math.abs(min) * 0.15 || 1
    return [min - d, max + d]
  }
  const range = max - min
  const pad = Math.max(range * padFrac, range * 0.1)
  return [min - pad, max + pad]
}

function niceTicks(min, max, count = 5) {
  if (min === max) return [min]
  const range = max - min
  const rough = range / count
  const pow = Math.pow(10, Math.floor(Math.log10(Math.abs(rough) || 1)))
  const candidates = [1, 2, 2.5, 5, 10].map((m) => m * pow)
  const step = candidates.find((s) => rough <= s) || candidates[candidates.length - 1]
  const start = Math.floor(min / step) * step
  const end = Math.ceil(max / step) * step
  const ticks = []
  for (let v = start; v <= end + step * 0.001; v += step) {
    ticks.push(Number(v.toPrecision(12)))
  }
  return ticks.length ? ticks : [min, max]
}

function ChartFrame({ title, yLabel, xLabel, width = DEFAULT_W, height = DEFAULT_H, children, legend }) {
  return (
    <div style={frameStyles.wrap}>
      {title && <div style={frameStyles.title}>{title}</div>}
      <svg
        width="100%"
        viewBox={`0 0 ${width} ${height}`}
        style={{ display: 'block', maxHeight: 300 }}
        preserveAspectRatio="xMidYMid meet"
      >
        {children}
      </svg>
      {legend && <div style={frameStyles.legendRow}>{legend}</div>}
      {(yLabel || xLabel) && (
        <div style={frameStyles.axisLabels}>
          {yLabel && <span>{yLabel}</span>}
          {xLabel && <span style={{ marginLeft: 'auto' }}>{xLabel}</span>}
        </div>
      )}
    </div>
  )
}

function GridAndAxes({ w, h, pad, yTicks, xLabels, formatY = (v) => String(v) }) {
  const plotW = w - pad.left - pad.right
  const plotH = h - pad.top - pad.bottom
  const yMin = yTicks[0]
  const yMax = yTicks[yTicks.length - 1]
  const yRange = (yMax - yMin) || 1
  const yAt = (v) => pad.top + plotH - ((v - yMin) / yRange) * plotH

  return (
    <>
      {/* Plot background */}
      <rect x={pad.left} y={pad.top} width={plotW} height={plotH} fill="var(--surface2)" rx="4" opacity="0.5" />
      {/* Horizontal grid + Y labels */}
      {yTicks.map((tick) => (
        <g key={tick}>
          <line
            x1={pad.left} y1={yAt(tick)} x2={pad.left + plotW} y2={yAt(tick)}
            stroke="var(--border)" strokeWidth="1"
          />
          <text x={pad.left - 6} y={yAt(tick) + 3} textAnchor="end" fill="var(--text-muted)" fontSize="9" fontFamily="var(--mono)">
            {formatY(tick)}
          </text>
        </g>
      ))}
      {/* X axis labels */}
      {xLabels?.map(({ i, label, n }) => {
        const x = pad.left + (n <= 1 ? plotW / 2 : (i / (n - 1)) * plotW)
        return (
          <text key={i} x={x} y={h - 6} textAnchor="middle" fill="var(--text-muted)" fontSize="9" fontFamily="var(--mono)">
            {label}
          </text>
        )
      })}
    </>
  )
}

function LegendItem({ label, color, dashed }) {
  return (
    <span style={legendStyles.item}>
      <span style={{
        ...legendStyles.swatch,
        background: dashed ? 'transparent' : color,
        borderBottom: dashed ? `2px dashed ${color}` : 'none',
      }} />
      {label}
    </span>
  )
}

export function ChartLegend({ items }) {
  return (
    <div style={legendStyles.row}>
      {items.map(({ label, color, dashed }) => (
        <LegendItem key={label} label={label} color={color} dashed={dashed} />
      ))}
    </div>
  )
}

/**
 * Actual (solid) → Predicted (dashed) time series with full axes.
 */
export function ActualPredictedChart({
  actual = [], predicted = [], title, yLabel, unit = '',
  actualLabel = 'Actual', predictedLabel = 'Predicted',
  width = DEFAULT_W, height = DEFAULT_H,
}) {
  const a = clean(actual)
  const p = clean(predicted)
  if (a.length === 0 && p.length === 0) return null

  const pad = CHART
  const w = width, h = height
  const combined = [...a, ...p]
  const [dMin, dMax] = padDomain(Math.min(...combined), Math.max(...combined))
  const yTicks = niceTicks(dMin, dMax)
  const yMin = yTicks[0], yMax = yTicks[yTicks.length - 1]
  const yRange = (yMax - yMin) || 1
  const plotW = w - pad.padLeft - pad.padRight
  const plotH = h - pad.padTop - pad.padBottom
  const total = a.length + p.length

  const xAt = (i) => pad.padLeft + (total <= 1 ? plotW / 2 : (i / (total - 1)) * plotW)
  const yAt = (v) => pad.padTop + plotH - ((v - yMin) / yRange) * plotH

  const actualPath = a.map((v, i) => `${xAt(i).toFixed(1)},${yAt(v).toFixed(1)}`).join(' ')
  const predPoints = []
  if (a.length > 0) predPoints.push(`${xAt(a.length - 1).toFixed(1)},${yAt(a[a.length - 1]).toFixed(1)}`)
  p.forEach((v, i) => predPoints.push(`${xAt(a.length + i).toFixed(1)},${yAt(v).toFixed(1)}`))
  const predictedPath = predPoints.join(' ')

  const nowX = a.length > 0 ? xAt(a.length - 1) : pad.padLeft
  const fmt = (v) => `${v.toFixed(Math.abs(v) < 10 ? 1 : 0)}${unit}`

  const xLabels = [
    { i: 0, label: 'Start', n: 2 },
    ...(a.length > 0 ? [{ i: a.length - 1, label: 'Now', n: total }] : []),
    ...(p.length > 0 ? [{ i: total - 1, label: `+${p.length}`, n: total }] : []),
  ]

  return (
    <ChartFrame
      title={title}
      yLabel={yLabel}
      width={w} height={h}
      legend={
        <>
          <LegendItem label={actualLabel} color="var(--text)" />
          <LegendItem label={predictedLabel} color="var(--accent)" dashed />
        </>
      }
    >
      <GridAndAxes w={w} h={h} pad={{ left: pad.padLeft, right: pad.padRight, top: pad.padTop, bottom: pad.padBottom }} yTicks={yTicks} xLabels={xLabels} formatY={fmt} />
      {a.length > 0 && p.length > 0 && (
        <line x1={nowX} y1={pad.padTop} x2={nowX} y2={h - pad.padBottom} stroke="var(--yellow)" strokeWidth="1.5" strokeDasharray="4,3" opacity="0.8" />
      )}
      {a.length > 0 && (
        <polyline points={actualPath} fill="none" stroke="var(--text)" strokeWidth="2" />
      )}
      {p.length > 0 && (
        <polyline points={predictedPath} fill="none" stroke="var(--accent)" strokeWidth="2" strokeDasharray="6,4" />
      )}
      {a.length > 0 && (
        <circle cx={xAt(a.length - 1)} cy={yAt(a[a.length - 1])} r="3.5" fill="var(--text)" stroke="var(--bg)" strokeWidth="1.5" />
      )}
      {p.length > 0 && (
        <circle cx={xAt(total - 1)} cy={yAt(p[p.length - 1])} r="4" fill="var(--accent)" stroke="var(--bg)" strokeWidth="1.5" />
      )}
    </ChartFrame>
  )
}

/**
 * Stacked/grouped bar chart: actual vs predicted per category.
 */
export function GroupedBarChart({
  categories = [], actual = [], predicted = [],
  title, yLabel, unit = '',
  actualLabel = 'Actual', predictedLabel = 'Predicted',
  width = DEFAULT_W, height = DEFAULT_H,
}) {
  if (!categories.length) return null
  const pad = CHART
  const w = width, h = height
  const allVals = [...clean(actual), ...clean(predicted)]
  const rawMax = Math.max(...allVals, 1)
  const [, dMax] = padDomain(0, rawMax, 0.2)
  const yTicks = niceTicks(0, dMax)
  const yMax = yTicks[yTicks.length - 1]
  const plotW = w - pad.padLeft - pad.padRight
  const plotH = h - pad.padTop - pad.padBottom
  const n = categories.length
  const groupW = plotW / n
  const barW = groupW * 0.22
  const yAt = (v) => pad.padTop + plotH - (v / yMax) * plotH
  const fmt = (v) => `${v.toFixed(Math.abs(v) < 10 ? 1 : 0)}${unit}`

  const xLabels = categories.map((label, i) => ({ i, label, n }))

  return (
    <ChartFrame
      title={title}
      yLabel={yLabel}
      width={w} height={h}
      legend={
        <>
          <LegendItem label={actualLabel} color="var(--text)" />
          <LegendItem label={predictedLabel} color="var(--accent)" />
        </>
      }
    >
      <GridAndAxes w={w} h={h} pad={{ left: pad.padLeft, right: pad.padRight, top: pad.padTop, bottom: pad.padBottom }} yTicks={yTicks} xLabels={xLabels} formatY={fmt} />
      {categories.map((_, i) => {
        const cx = pad.padLeft + i * groupW + groupW / 2
        const aVal = actual[i]
        const pVal = predicted[i]
        return (
          <g key={i}>
            {typeof aVal === 'number' && (
              <rect x={cx - barW - 2} y={yAt(aVal)} width={barW} height={pad.padTop + plotH - yAt(aVal)} fill="var(--text)" rx="2" opacity="0.85" />
            )}
            {typeof pVal === 'number' && (
              <rect x={cx + 2} y={yAt(pVal)} width={barW} height={pad.padTop + plotH - yAt(pVal)} fill="var(--accent)" rx="2" opacity="0.85" />
            )}
          </g>
        )
      })}
    </ChartFrame>
  )
}

/**
 * Multi-series line chart (e.g. M1 Prophet + GRU components).
 */
export function MultiLineChart({
  series = [], title, yLabel, unit = '', width = DEFAULT_W, height = DEFAULT_H,
}) {
  const valid = series.filter((s) => clean(s.data).length > 0)
  if (!valid.length) return null

  const pad = CHART
  const w = width, h = height
  const allVals = valid.flatMap((s) => clean(s.data))
  const [dMin, dMax] = padDomain(Math.min(...allVals), Math.max(...allVals))
  const yTicks = niceTicks(dMin, dMax)
  const yMin = yTicks[0], yMax = yTicks[yTicks.length - 1]
  const yRange = (yMax - yMin) || 1
  const plotW = w - pad.padLeft - pad.padRight
  const plotH = h - pad.padTop - pad.padBottom
  const n = Math.max(...valid.map((s) => clean(s.data).length))
  const fmt = (v) => `${v.toFixed(1)}${unit}`

  const xAt = (i) => pad.padLeft + (n <= 1 ? plotW / 2 : (i / (n - 1)) * plotW)
  const yAt = (v) => pad.padTop + plotH - ((v - yMin) / yRange) * plotH

  const xLabels = [{ i: 0, label: '0', n }, { i: n - 1, label: String(n - 1), n }]

  return (
    <ChartFrame
      title={title}
      yLabel={yLabel}
      width={w} height={h}
      legend={valid.map((s) => <LegendItem key={s.label} label={s.label} color={s.color} dashed={s.dashed} />)}
    >
      <GridAndAxes w={w} h={h} pad={{ left: pad.padLeft, right: pad.padRight, top: pad.padTop, bottom: pad.padBottom }} yTicks={yTicks} xLabels={xLabels} formatY={fmt} />
      {valid.map((s) => {
        const d = clean(s.data)
        const path = d.map((v, i) => `${xAt(i).toFixed(1)},${yAt(v).toFixed(1)}`).join(' ')
        return (
          <polyline
            key={s.label}
            points={path}
            fill="none"
            stroke={s.color}
            strokeWidth="2"
            strokeDasharray={s.dashed ? '5,4' : undefined}
          />
        )
      })}
    </ChartFrame>
  )
}

/**
 * Score/error trend across samples with threshold line.
 */
export function ScoreTrendChart({
  points, threshold, scale = 'linear', title, yLabel,
  width = DEFAULT_W, height = DEFAULT_H,
}) {
  if (!points?.length) return null

  const pad = CHART
  const w = width, h = height
  const t = scale === 'log' ? (v) => Math.log10(Math.max(v, 1e-6)) : (v) => v
  const invT = scale === 'log' ? (v) => Math.pow(10, v) : (v) => v

  const values = points.map((p) => t(p.value))
  const thresholdT = t(threshold)
  const allValues = [...values, thresholdT]
  const [dMin, dMax] = padDomain(Math.min(...allValues), Math.max(...allValues), 0.25)
  const yTicks = niceTicks(dMin, dMax)
  const yMin = yTicks[0], yMax = yTicks[yTicks.length - 1]
  const yRange = (yMax - yMin) || 1
  const plotW = w - pad.padLeft - pad.padRight
  const plotH = h - pad.padTop - pad.padBottom
  const n = points.length

  const xAt = (i) => pad.padLeft + (n <= 1 ? plotW / 2 : (i / (n - 1)) * plotW)
  const yAt = (v) => pad.padTop + plotH - ((v - yMin) / yRange) * plotH
  const fmt = (v) => {
    const raw = invT(v)
    return scale === 'log' ? raw.toExponential(1) : raw.toFixed(3)
  }

  const linePath = points.map((p, i) => `${xAt(i).toFixed(1)},${yAt(t(p.value)).toFixed(1)}`).join(' ')
  const thresholdY = yAt(thresholdT)
  const xLabels = points.length <= 8
    ? points.map((p, i) => ({ i, label: `#${p.sample}`, n }))
    : [{ i: 0, label: `#${points[0].sample}`, n }, { i: n - 1, label: `#${points[n - 1].sample}`, n }]

  return (
    <ChartFrame
      title={title}
      yLabel={yLabel || (scale === 'log' ? 'MSE (log)' : 'Score')}
      width={w} height={h}
      legend={
        <>
          <LegendItem label="Score" color="var(--accent)" />
          <LegendItem label="Threshold" color="var(--yellow)" dashed />
          <LegendItem label="Anomaly" color="var(--red)" />
          <LegendItem label="Normal" color="var(--green)" />
        </>
      }
    >
      <GridAndAxes w={w} h={h} pad={{ left: pad.padLeft, right: pad.padRight, top: pad.padTop, bottom: pad.padBottom }} yTicks={yTicks} xLabels={xLabels} formatY={fmt} />
      <line x1={pad.padLeft} y1={thresholdY} x2={w - pad.padRight} y2={thresholdY} stroke="var(--yellow)" strokeWidth="1.5" strokeDasharray="5,4" />
      {n > 1 && (
        <polyline points={linePath} fill="none" stroke="var(--accent)" strokeWidth="2" opacity="0.6" />
      )}
      {points.map((p, i) => (
        <circle
          key={p.sample ?? i}
          cx={xAt(i)}
          cy={yAt(t(p.value))}
          r={p.isAnomaly ? 4.5 : 3}
          fill={p.isAnomaly ? 'var(--red)' : 'var(--green)'}
          stroke="var(--bg)"
          strokeWidth="1.5"
        />
      ))}
    </ChartFrame>
  )
}

/**
 * Anomaly status timeline — binary normal/alert per sample.
 */
export function AnomalyTimelineChart({ points = [], title, width = 480, height = 80 }) {
  if (!points.length) return null
  const pad = { left: 44, right: 12, top: 16, bottom: 24 }
  const w = width, h = height
  const plotW = w - pad.left - pad.right
  const n = points.length
  const cellW = plotW / n

  return (
    <ChartFrame title={title} width={w} height={h}>
      {points.map((p, i) => (
        <g key={p.sample ?? i}>
          <rect
            x={pad.left + i * cellW + 1}
            y={pad.top}
            width={cellW - 2}
            height={h - pad.top - pad.bottom}
            fill={p.isAnomaly ? 'rgba(239,68,68,0.35)' : 'rgba(34,197,94,0.2)'}
            stroke={p.isAnomaly ? 'var(--red)' : 'var(--green)'}
            strokeWidth="1"
            rx="3"
          />
          <text
            x={pad.left + i * cellW + cellW / 2}
            y={h - 6}
            textAnchor="middle"
            fill="var(--text-muted)"
            fontSize="8"
            fontFamily="var(--mono)"
          >
            {p.sample}
          </text>
        </g>
      ))}
    </ChartFrame>
  )
}

const frameStyles = {
  wrap: {
    background: 'var(--surface2)',
    border: '1px solid var(--border)',
    borderRadius: 8,
    padding: '14px 14px 10px',
    marginBottom: 12,
    maxWidth: '100%',
  },
  title: {
    fontSize: 11,
    fontWeight: 700,
    textTransform: 'uppercase',
    letterSpacing: '0.06em',
    color: 'var(--text-dim)',
    marginBottom: 8,
  },
  legendRow: { display: 'flex', flexWrap: 'wrap', gap: 14, marginTop: 8, fontSize: 10, color: 'var(--text-muted)' },
  axisLabels: { display: 'flex', fontSize: 9, color: 'var(--text-muted)', marginTop: 4, paddingLeft: 52 },
}

const legendStyles = {
  row: { display: 'flex', flexWrap: 'wrap', gap: 14, fontSize: 10, color: 'var(--text-muted)' },
  item: { display: 'flex', alignItems: 'center', gap: 5 },
  swatch: { width: 14, height: 3, borderRadius: 1, display: 'inline-block' },
}
