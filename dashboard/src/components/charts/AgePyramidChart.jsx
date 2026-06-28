import {
  BarChart, Bar, Cell, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceArea,
} from 'recharts'
import { getTheme, getTooltipStyle } from '../../lib/chartTheme.js'
import { useDarkMode } from '../../lib/useDarkMode.jsx'
import { niceTicks } from '../../lib/format.js'

// Cohort sizes by age at the selected year. Bars are coloured by life stage
// so the baby-boom hump and its march toward retirement are visible as you
// scrub the year. y-axis is fixed (popMax) so cross-year comparison is honest.
export default function AgePyramidChart({ ages, pop, chi, psi, popMax }) {
  const { isDark } = useDarkMode()
  const theme = getTheme(isDark)
  const ui = theme.ui

  if (!pop) return <div className="flex items-center justify-center h-full text-xs text-slate-500">—</div>

  const data = ages.map(a => ({ age: a, pop: pop[a] }))
  const { ticks: yt, niceMax } = niceTicks(popMax)
  const colorFor = (age) =>
    age < chi ? theme.colors.muted
      : age < psi ? theme.colors.primary
        : theme.colors.warning

  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={data} margin={{ top: 6, right: 10, bottom: 2, left: 0 }} barCategoryGap={0}>
        <CartesianGrid stroke={ui.grid} strokeDasharray="2 4" vertical={false} />
        <ReferenceArea x1={chi} x2={psi - 1} fill={theme.colors.primary} fillOpacity={0.05} />
        <XAxis
          dataKey="age" type="category"
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
          cursor={{ fill: isDark ? 'rgba(148,163,184,0.08)' : 'rgba(71,85,105,0.08)' }}
          formatter={(v) => [v.toFixed(3), 'cohort size']}
          labelFormatter={(a) => `age ${a}`}
        />
        <Bar dataKey="pop" isAnimationActive={false}>
          {data.map(d => <Cell key={d.age} fill={colorFor(d.age)} />)}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}
