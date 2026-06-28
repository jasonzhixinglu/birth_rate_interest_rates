import { useRef, useEffect, useState, useCallback } from 'react'
import { getTheme } from '../../lib/chartTheme.js'
import { useDarkMode } from '../../lib/useDarkMode.jsx'

const MARGIN = { top: 14, right: 18, bottom: 28, left: 40 }

// Interactive interest-rate path. Click or drag anywhere to select a year;
// the selected year drives the linked age-distribution / asset panels.
export default function RateNavigatorChart({
  rPath, shockYear, trough, rInit, rTerminal,
  selectedYear, onSelectYear, color = '#6366f1',
}) {
  const svgRef = useRef(null)
  const wrapRef = useRef(null)
  const dragging = useRef(false)
  const [dims, setDims] = useState({ width: 720, height: 240 })
  const [hover, setHover] = useState(null)
  const { isDark } = useDarkMode()
  const theme = getTheme(isDark)

  useEffect(() => {
    if (!wrapRef.current) return
    const ro = new ResizeObserver(([e]) => {
      const w = e.contentRect.width
      const h = e.contentRect.height
      if (w > 0) setDims({ width: w, height: h > 80 ? h : Math.round(w * 0.34) })
    })
    ro.observe(wrapRef.current)
    return () => ro.disconnect()
  }, [])

  const { width, height } = dims
  const innerW = width - MARGIN.left - MARGIN.right
  const innerH = height - MARGIN.top - MARGIN.bottom

  if (!rPath || rPath.length === 0) {
    return <div ref={wrapRef} className="flex items-center justify-center h-full text-xs text-slate-500">Loading…</div>
  }

  const years = rPath.map(p => p.year)
  const vals = rPath.map(p => p.r)
  const xMin = years[0], xMax = years[years.length - 1]
  const rawMin = Math.min(...vals, rTerminal, trough.value)
  const rawMax = Math.max(...vals, rInit)
  const padV = (rawMax - rawMin) * 0.12 || 0.5
  const yMin = Math.max(0, rawMin - padV), yMax = rawMax + padV

  const X = (yr) => ((yr - xMin) / (xMax - xMin)) * innerW
  const Y = (v) => innerH - ((v - yMin) / (yMax - yMin)) * innerH

  const linePath = rPath
    .map((p, i) => `${i === 0 ? 'M' : 'L'}${X(p.year).toFixed(1)},${Y(p.r).toFixed(1)}`)
    .join(' ')
  const areaPath = `${linePath} L${X(xMax).toFixed(1)},${innerH} L${X(xMin).toFixed(1)},${innerH} Z`

  // Axis ticks
  const xTicks = []
  for (let yr = Math.ceil(xMin / 20) * 20; yr <= xMax; yr += 20) xTicks.push(yr)
  const yTicks = Array.from({ length: 5 }, (_, i) => yMin + (i / 4) * (yMax - yMin))

  const pxToYear = useCallback((e) => {
    const rect = svgRef.current.getBoundingClientRect()
    const px = e.clientX - rect.left - MARGIN.left
    const frac = Math.max(0, Math.min(1, px / innerW))
    return Math.round(xMin + frac * (xMax - xMin))
  }, [innerW, xMin, xMax])

  const down = useCallback((e) => { dragging.current = true; onSelectYear(pxToYear(e)) }, [onSelectYear, pxToYear])
  const move = useCallback((e) => {
    const yr = pxToYear(e)
    const pt = rPath.find(p => p.year === yr)
    setHover(pt ?? null)
    if (dragging.current) onSelectYear(yr)
  }, [pxToYear, rPath, onSelectYear])
  const up = useCallback(() => { dragging.current = false }, [])
  const leave = useCallback(() => { dragging.current = false; setHover(null) }, [])

  const selPt = rPath.find(p => p.year === selectedYear)
  const selX = selPt ? X(selPt.year) : null
  const selY = selPt ? Y(selPt.r) : null

  return (
    <div ref={wrapRef} className="w-full h-full">
      <svg
        ref={svgRef} width={width} height={height}
        className="w-full select-none cursor-crosshair"
        onMouseDown={down} onMouseMove={move} onMouseUp={up} onMouseLeave={leave}
      >
        <defs>
          <linearGradient id="rate-fill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity={isDark ? 0.28 : 0.18} />
            <stop offset="100%" stopColor={color} stopOpacity={0} />
          </linearGradient>
        </defs>
        <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>

          {/* horizontal grid */}
          {yTicks.map((v, i) => (
            <g key={i}>
              <line x1={0} x2={innerW} y1={Y(v)} y2={Y(v)} stroke={theme.ui.grid} strokeWidth={0.5} />
              <text x={-6} y={Y(v) + 3} fontSize={theme.ui.tickFontSize} fill={theme.ui.tickLabel} textAnchor="end">
                {v.toFixed(1)}%
              </text>
            </g>
          ))}

          {/* BGP reference lines */}
          {[{ v: rInit, label: 'r init' }, { v: rTerminal, label: 'r terminal' }].map((ref, i) => (
            <g key={i}>
              <line x1={0} x2={innerW} y1={Y(ref.v)} y2={Y(ref.v)}
                stroke={theme.colors.muted} strokeWidth={1} strokeDasharray="4 4" opacity={0.6} />
              <text x={innerW - 2} y={Y(ref.v) - 3} fontSize={9} fill={theme.colors.muted} textAnchor="end" opacity={0.85}>
                {ref.label} {ref.v.toFixed(2)}%
              </text>
            </g>
          ))}

          {/* fertility shock marker */}
          {shockYear >= xMin && shockYear <= xMax && (
            <g>
              <line x1={X(shockYear)} x2={X(shockYear)} y1={0} y2={innerH}
                stroke={theme.colors.warning} strokeWidth={1} strokeDasharray="2 3" opacity={0.7} />
              <text x={X(shockYear) + 3} y={11} fontSize={9} fill={theme.colors.warning}>fertility shock {shockYear}</text>
            </g>
          )}

          {/* area + line */}
          <path d={areaPath} fill="url(#rate-fill)" stroke="none" />
          <path d={linePath} fill="none" stroke={color} strokeWidth={theme.strokeWidths.line} />

          {/* trough dot */}
          {trough.year >= xMin && trough.year <= xMax && (
            <circle cx={X(trough.year)} cy={Y(trough.value)} r={3} fill={color} stroke={isDark ? '#0f172a' : '#fff'} strokeWidth={1} />
          )}

          {/* selected marker */}
          {selX !== null && (
            <g>
              <line x1={selX} x2={selX} y1={0} y2={innerH} stroke={color} strokeWidth={1.5} strokeDasharray="5 3" />
              <circle cx={selX} cy={selY} r={4.5} fill={color} stroke={isDark ? '#0f172a' : '#fff'} strokeWidth={1.5} />
            </g>
          )}

          {/* x ticks */}
          {xTicks.map(yr => (
            <text key={yr} x={X(yr)} y={innerH + 16} fontSize={theme.ui.tickFontSize} fill={theme.ui.tickLabel} textAnchor="middle">
              {yr}
            </text>
          ))}
          <line x1={0} x2={innerW} y1={innerH} y2={innerH} stroke={theme.ui.axis} />

          {/* hover readout */}
          {hover && (() => {
            const hx = X(hover.year), hy = Y(hover.r)
            const label = `${hover.year}  ·  ${hover.r.toFixed(2)}%`
            const LW = 92
            const lx = hx > innerW - LW ? hx - LW - 6 : hx + 6
            return (
              <g pointerEvents="none">
                <circle cx={hx} cy={hy} r={2.5} fill={theme.colors.muted} />
                <rect x={lx} y={2} width={LW} height={15} rx={3} fill={theme.ui.tooltipBg} stroke={theme.ui.tooltipBorder} />
                <text x={lx + 6} y={13} fontSize={10} fill={theme.ui.tooltipText}>{label}</text>
              </g>
            )
          })()}
        </g>
      </svg>
    </div>
  )
}
