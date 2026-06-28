import { useState } from 'react'
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceArea, ReferenceLine,
} from 'recharts'
import { getTheme, getTooltipStyle } from '../../lib/chartTheme.js'
import { useDarkMode } from '../../lib/useDarkMode.jsx'
import { niceTicks } from '../../lib/format.js'

const MODES = {
  aggregate: {
    label: 'Aggregate', key: 'wealth', colorKey: 'accent',
    desc: 'Total assets held by each age = cohort size × per-capita assets. The hump is the savings glut.',
  },
  percapita: {
    label: 'Per capita', key: 'assetPc', colorKey: 'positive',
    desc: 'One cohort’s asset holdings over its life: zero until working age, peak at retirement, drawn down after.',
  },
}

// Asset distribution by age at the selected year. Toggle between aggregate
// wealth (distribution across cohorts) and per-capita (lifecycle accumulation).
export default function AssetByAgeChart({ ages, pop, assetPc, chi, psi, assetPcMax, wealthMax }) {
  const [mode, setMode] = useState('aggregate')
  const { isDark } = useDarkMode()
  const theme = getTheme(isDark)
  const ui = theme.ui

  if (!assetPc) return <div className="flex items-center justify-center h-full text-xs text-slate-500">—</div>

  const cfg = MODES[mode]
  const color = theme.colors[cfg.colorKey]
  const data = ages.map(a => ({ age: a, assetPc: assetPc[a], wealth: pop[a] * assetPc[a] }))
  const { ticks: yt, niceMax } = niceTicks(mode === 'aggregate' ? wealthMax : assetPcMax)

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center gap-1.5 mb-1">
        {Object.entries(MODES).map(([k, m]) => (
          <button
            key={k}
            onClick={() => setMode(k)}
            className={`text-xs px-2.5 py-0.5 rounded-md font-medium transition-all ${
              mode === k
                ? 'bg-indigo-600 text-white'
                : 'bg-slate-100 dark:bg-slate-800 text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-200'
            }`}
          >{m.label}</button>
        ))}
      </div>
      <div className="flex-1 min-h-0">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 6, right: 10, bottom: 2, left: 0 }}>
            <defs>
              <linearGradient id={`asset-fill-${mode}`} x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={color} stopOpacity={0.45} />
                <stop offset="100%" stopColor={color} stopOpacity={0.04} />
              </linearGradient>
            </defs>
            <CartesianGrid stroke={ui.grid} strokeDasharray="2 4" vertical={false} />
            <ReferenceArea x1={chi} x2={psi - 1} fill={theme.colors.primary} fillOpacity={0.05} />
            <ReferenceLine x={psi} stroke={theme.colors.warning} strokeDasharray="3 3" strokeOpacity={0.6}
              label={{ value: 'retire', position: 'insideTopRight', fill: theme.colors.warning, fontSize: 9 }} />
            <XAxis
              dataKey="age" type="number" domain={[0, ages.length - 1]}
              ticks={[0, 20, 40, 60, 79]}
              tick={{ fill: ui.tickLabel, fontSize: ui.tickFontSize }}
              stroke={ui.axis} tickLine={false}
              label={{ value: 'age', position: 'insideBottomRight', offset: -2, fill: ui.tickLabel, fontSize: 10 }}
            />
            <YAxis
              domain={[0, niceMax]} ticks={yt}
              tick={{ fill: ui.tickLabel, fontSize: ui.tickFontSize }}
              stroke={ui.axis} tickLine={false} width={40}
            />
            <Tooltip
              contentStyle={getTooltipStyle(isDark)}
              cursor={{ stroke: ui.axis }}
              formatter={(v) => [v.toFixed(3), cfg.label === 'Aggregate' ? 'total assets' : 'assets / capita']}
              labelFormatter={(a) => `age ${a}`}
            />
            <Area
              type="monotone" dataKey={cfg.key} stroke={color} strokeWidth={1.8}
              fill={`url(#asset-fill-${mode})`} isAnimationActive={false} dot={false}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
      <p className="text-[11px] text-slate-500 dark:text-slate-500 mt-1 leading-snug">{cfg.desc}</p>
    </div>
  )
}
