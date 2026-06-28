import { useState, useEffect } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine, Legend,
} from 'recharts'
import { getTheme, getTooltipStyle } from '../lib/chartTheme.js'
import { useDarkMode } from '../lib/useDarkMode.jsx'
import { pct, downloadCSV } from '../lib/format.js'

const BASE = import.meta.env.BASE_URL

// Overlay every country's interest-rate transition on one axis.
export default function CrossCountryPanel() {
  const [countries, setCountries] = useState(null)
  const { isDark } = useDarkMode()
  const theme = getTheme(isDark)
  const ui = theme.ui

  useEffect(() => {
    fetch(`${BASE}data/manifest.json`)
      .then(r => r.json())
      .then(m => Promise.all(m.countries.map(c =>
        fetch(`${BASE}data/countries/${c.slug}.json`).then(r => r.json()))))
      .then(setCountries)
      .catch(() => {})
  }, [])

  if (!countries) return <div className="flex items-center justify-center h-64 text-sm text-slate-500">Loading…</div>

  // Merge all rPaths into rows keyed by year (1950–2050).
  const byYear = {}
  for (const d of countries) {
    for (const p of d.rPath) {
      byYear[p.year] = byYear[p.year] || { year: p.year }
      byYear[p.year][d.meta.slug] = p.r
    }
  }
  const merged = Object.values(byYear).sort((a, b) => a.year - b.year)

  const download = () => {
    const cols = [{ key: 'year', label: 'year' },
      ...countries.map(d => ({ key: d.meta.slug, label: `${d.meta.slug}_pct` }))]
    downloadCSV('cross_country_rate_paths.csv', merged, cols)
  }

  return (
    <div className="max-w-screen-xl mx-auto flex flex-col gap-4 py-1">

      <div className="panel p-4">
        <div className="flex items-baseline justify-between mb-2">
          <div>
            <h2 className="text-sm font-semibold text-slate-800 dark:text-slate-100">Interest-rate transitions across countries</h2>
            <p className="text-xs text-slate-500">Same structural model; only the fertility shock’s size and timing differ.</p>
          </div>
          <button onClick={download}
            className="flex items-center gap-1 px-2 py-0.5 rounded bg-slate-100 hover:bg-slate-200 dark:bg-slate-800 dark:hover:bg-slate-700 text-slate-500 dark:text-slate-400 text-xs">
            <svg width="10" height="10" viewBox="0 0 12 12" fill="none"><path d="M6 1v7M3.5 5.5 6 8l2.5-2.5M2 10h8" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" /></svg>
            CSV
          </button>
        </div>
        <div className="h-[360px]">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={merged} margin={{ top: 8, right: 18, bottom: 4, left: 0 }}>
              <CartesianGrid stroke={ui.grid} strokeDasharray="2 4" vertical={false} />
              <ReferenceLine x={1970} stroke={theme.colors.muted} strokeDasharray="3 3" strokeOpacity={0.5} />
              <XAxis dataKey="year" tick={{ fill: ui.tickLabel, fontSize: ui.tickFontSize }} stroke={ui.axis} tickLine={false}
                ticks={[1950, 1970, 1990, 2010, 2030, 2050]} />
              <YAxis tick={{ fill: ui.tickLabel, fontSize: ui.tickFontSize }} stroke={ui.axis} tickLine={false}
                width={44} tickFormatter={v => `${v}%`} />
              <Tooltip contentStyle={getTooltipStyle(isDark)}
                formatter={(v, n) => [`${v.toFixed(2)}%`, countries.find(d => d.meta.slug === n)?.meta.name ?? n]} />
              <Legend wrapperStyle={{ fontSize: 11, color: ui.tickLabel }}
                formatter={(v) => countries.find(d => d.meta.slug === v)?.meta.name ?? v} />
              {countries.map(d => (
                <Line key={d.meta.slug} type="monotone" dataKey={d.meta.slug} name={d.meta.slug}
                  stroke={d.meta.color} strokeWidth={1.8} dot={false} connectNulls />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* summary table */}
      <div className="panel p-4 overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left label">
              <th className="py-1.5 pr-4 font-medium">Country</th>
              <th className="py-1.5 pr-4 font-medium">TFR pre → post</th>
              <th className="py-1.5 pr-4 font-medium">Shock</th>
              <th className="py-1.5 pr-4 font-medium">r init</th>
              <th className="py-1.5 pr-4 font-medium">r terminal</th>
              <th className="py-1.5 pr-4 font-medium">Trough</th>
            </tr>
          </thead>
          <tbody>
            {countries.map(d => {
              const m = d.meta
              return (
                <tr key={m.slug} className="border-t border-slate-200 dark:border-slate-800">
                  <td className="py-1.5 pr-4">
                    <span className="inline-flex items-center gap-2">
                      <span className="w-2.5 h-2.5 rounded-full" style={{ background: m.color }} />
                      <span className="font-medium text-slate-800 dark:text-slate-200">{m.name}</span>
                    </span>
                  </td>
                  <td className="py-1.5 pr-4 tabular-nums text-slate-600 dark:text-slate-400">{m.tfrHigh} → {m.tfrLow}</td>
                  <td className="py-1.5 pr-4 tabular-nums text-slate-600 dark:text-slate-400">{m.shockYear}</td>
                  <td className="py-1.5 pr-4 tabular-nums val-neutral">{pct(m.rInit)}</td>
                  <td className="py-1.5 pr-4 tabular-nums val-neutral">{pct(m.rTerminal)}</td>
                  <td className="py-1.5 pr-4 tabular-nums val-negative">{pct(m.trough.value)} <span className="text-slate-400">· {m.trough.year}</span></td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}
