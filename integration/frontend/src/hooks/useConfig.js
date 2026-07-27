import { useEffect, useState } from 'react'

let cached = null
let pending = null

export function useConfig() {
  const [config, setConfig] = useState(cached)
  const [loading, setLoading] = useState(!cached)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (cached) return
    if (!pending) {
      pending = fetch('/api/config')
        .then((r) => {
          if (!r.ok) throw new Error(`Config fetch failed (${r.status})`)
          return r.json()
        })
        .then((data) => {
          cached = data
          return data
        })
    }
    pending
      .then((data) => { setConfig(data); setLoading(false) })
      .catch((e) => { setError(e.message); setLoading(false) })
  }, [])

  return { config, loading, error }
}

export function layerColor(config, layerKey) {
  const key = config?.ui?.layers?.[layerKey]?.color || 'accent'
  return key === 'red' ? 'var(--red)' : key === 'accent' ? 'var(--accent)' : `var(--${key})`
}

export function moduleConfig(config, moduleKey) {
  return config?.modules?.[moduleKey] || {}
}

export function formatStreamHint(config, streamState, totalSamples) {
  const hints = {
    stopped: 'Stream stopped. Click Reset to clear, or Start to run again.',
    ...(config?.ui?.stream_hints || {}),
  }
  const port = config?.services?.integration_port ?? 5000
  const template = hints[streamState] || ''
  return template
    .replace('{samples}', String(totalSamples))
    .replace('{port}', String(port))
}
