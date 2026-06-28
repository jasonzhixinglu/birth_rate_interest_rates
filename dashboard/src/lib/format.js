// Small formatting helpers shared across panels/charts.

export const pct = (x, dp = 1) =>
  x == null || Number.isNaN(x) ? '—' : `${(x).toFixed(dp)}%`

export const num = (x, dp = 2) =>
  x == null || Number.isNaN(x) ? '—' : x.toFixed(dp)

// "Nice" axis ticks from 0 up to a rounded ceiling above maxVal.
// Avoids Recharts auto-generating ugly/out-of-range float ticks on an
// explicit non-round domain. Returns { ticks, niceMax }.
export function niceTicks(maxVal, count = 5) {
  if (!maxVal || maxVal <= 0 || !Number.isFinite(maxVal)) return { ticks: [0, 1], niceMax: 1 }
  const rawStep = maxVal / count
  const mag = Math.pow(10, Math.floor(Math.log10(rawStep)))
  const norm = rawStep / mag
  const step = (norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 2.5 ? 2.5 : norm <= 5 ? 5 : 10) * mag
  const niceMax = Math.ceil(maxVal / step) * step
  const ticks = []
  for (let v = 0; v <= niceMax + step * 1e-6; v += step) ticks.push(Number(v.toFixed(6)))
  return { ticks, niceMax }
}

// Trigger a client-side CSV download from an array of row objects.
export function downloadCSV(filename, rows, columns) {
  const header = columns.map(c => c.label ?? c.key).join(',')
  const body = rows.map(r =>
    columns.map(c => {
      const v = r[c.key]
      return v == null ? '' : (typeof v === 'number' ? v : `${v}`)
    }).join(',')
  )
  const csv = [header, ...body].join('\n')
  const blob = new Blob([csv], { type: 'text/csv' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}
