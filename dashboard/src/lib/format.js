// Small formatting helpers shared across panels/charts.

export const pct = (x, dp = 1) =>
  x == null || Number.isNaN(x) ? '—' : `${(x).toFixed(dp)}%`

export const num = (x, dp = 2) =>
  x == null || Number.isNaN(x) ? '—' : x.toFixed(dp)

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
