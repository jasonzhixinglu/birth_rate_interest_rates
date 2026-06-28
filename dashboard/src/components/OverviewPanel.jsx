import { useState, useEffect, useRef, useMemo, useCallback } from 'react'
import RateNavigatorChart from './charts/RateNavigatorChart.jsx'
import AgePyramidChart from './charts/AgePyramidChart.jsx'
import AssetByAgeChart from './charts/AssetByAgeChart.jsx'
import { useSessionState } from '../lib/sessionState.js'
import { pct, num, downloadCSV } from '../lib/format.js'

const BASE = import.meta.env.BASE_URL

function PlayIcon() { return <svg width="11" height="11" viewBox="0 0 12 12" fill="currentColor"><path d="M3 2l7 4-7 4z" /></svg> }
function PauseIcon() { return <svg width="11" height="11" viewBox="0 0 12 12" fill="currentColor"><rect x="3" y="2" width="2.5" height="8" /><rect x="7" y="2" width="2.5" height="8" /></svg> }
function DownloadIcon() {
  return <svg width="11" height="11" viewBox="0 0 12 12" fill="none" className="shrink-0"><path d="M6 1v7M3.5 5.5 6 8l2.5-2.5M2 10h8" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" /></svg>
}

export default function OverviewPanel() {
  const [manifest, setManifest] = useState(null)
  const [slug, setSlug] = useSessionState('brir-country', 'germany')
  const [year, setYear] = useSessionState('brir-year', 2028)
  const [playing, setPlaying] = useState(false)
  const cache = useRef({})
  const [data, setData] = useState(null)
  const [sidebarOpen, setSidebarOpen] = useState(false)

  // Load manifest once.
  useEffect(() => {
    fetch(`${BASE}data/manifest.json`).then(r => r.json()).then(setManifest).catch(() => {})
  }, [])

  // Load the selected country (cached).
  useEffect(() => {
    let alive = true
    if (cache.current[slug]) { setData(cache.current[slug]); return }
    setData(null)
    fetch(`${BASE}data/countries/${slug}.json`)
      .then(r => r.json())
      .then(d => { if (alive) { cache.current[slug] = d; setData(d) } })
      .catch(() => {})
    return () => { alive = false }
  }, [slug])

  const meta = data?.meta
  const snapStart = meta?.snapStart ?? 1950
  const snapEnd = meta?.snapEnd ?? 2050

  // Clamp year into range whenever the country changes.
  useEffect(() => {
    if (!meta) return
    setYear(y => Math.min(snapEnd, Math.max(snapStart, y)))
  }, [meta, snapStart, snapEnd, setYear])

  // Play/pause animation across years.
  useEffect(() => {
    if (!playing || !meta) return
    const id = setInterval(() => {
      setYear(y => (y >= snapEnd ? snapStart : y + 1))
    }, 120)
    return () => clearInterval(id)
  }, [playing, meta, snapStart, snapEnd, setYear])

  const snap = data?.snapshots?.[String(year)]
  const rAtYear = useMemo(() => data?.rPath?.find(p => p.year === year)?.r ?? null, [data, year])

  // Derived demographics for the selected snapshot.
  const stats = useMemo(() => {
    if (!snap || !meta) return null
    const { chi, psi, J } = meta
    let working = 0, retired = 0, young = 0, capital = 0
    for (let a = 0; a < J; a++) {
      const n = snap.pop[a]
      if (a < chi) young += n
      else if (a < psi) working += n
      else retired += n
      capital += n * snap.assetPc[a]
    }
    return {
      dependency: retired / working * 100,
      capPerWorker: capital / working,
      workingShare: working / (young + working + retired) * 100,
    }
  }, [snap, meta])

  const onSelectYear = useCallback((y) => { setPlaying(false); setYear(y) }, [setYear])

  if (!manifest || !data || !meta) {
    return <div className="flex items-center justify-center h-64 text-sm text-slate-500">Loading model output…</div>
  }

  const downloadRate = () => downloadCSV(`${slug}_rate_path.csv`, data.rPath,
    [{ key: 'year', label: 'year' }, { key: 'r', label: 'real_rate_pct' }])
  const downloadSnapshot = () => {
    const rows = data.ages.map(a => ({ age: a, pop: snap.pop[a], assetPc: snap.assetPc[a], wealth: snap.pop[a] * snap.assetPc[a] }))
    downloadCSV(`${slug}_${year}_age_profile.csv`, rows,
      [{ key: 'age', label: 'age' }, { key: 'pop', label: 'cohort_size' }, { key: 'assetPc', label: 'assets_per_capita' }, { key: 'wealth', label: 'aggregate_assets' }])
  }

  const statCards = [
    { label: `Real rate · ${year}`, value: pct(rAtYear), tone: 'val-neutral' },
    { label: 'Old-age dependency', value: stats ? `${stats.dependency.toFixed(0)}%` : '—', tone: 'val-positive' },
    { label: 'Capital / worker', value: stats ? num(stats.capPerWorker, 2) : '—', tone: 'val-neutral' },
    { label: `Trough`, value: `${pct(meta.trough.value)}`, sub: meta.trough.year, tone: 'val-negative' },
  ]

  return (
    <div className="flex flex-col lg:flex-row gap-4 md:h-full md:overflow-hidden">

      {/* ── Sidebar: config + legend ── */}
      <div className="panel lg:w-[210px] xl:w-[230px] lg:shrink-0">
        <button
          className="lg:hidden w-full flex items-center justify-between px-4 py-3 text-xs font-medium text-slate-700 dark:text-slate-300"
          onClick={() => setSidebarOpen(o => !o)}
        >
          <span>Configuration</span>
          <svg width="12" height="12" viewBox="0 0 12 12" fill="none" className={`transition-transform ${sidebarOpen ? 'rotate-180' : ''}`}>
            <path d="M2 4l4 4 4-4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </button>

        <div className={`flex-col gap-4 p-4 ${sidebarOpen ? 'flex' : 'hidden'} lg:flex`}>

          {/* Country */}
          <div>
            <div className="label mb-2">Country</div>
            <div className="grid grid-cols-2 gap-1.5">
              {manifest.countries.map(c => (
                <button
                  key={c.slug}
                  onClick={() => setSlug(c.slug)}
                  className={`flex items-center gap-1.5 text-xs py-1.5 px-2 rounded-md font-medium transition-all ${
                    slug === c.slug
                      ? 'bg-indigo-600 text-white'
                      : 'bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400 hover:bg-slate-200 dark:hover:bg-slate-700'
                  }`}
                >
                  <span className="w-2 h-2 rounded-full shrink-0" style={{ background: c.color }} />
                  {c.name}
                </button>
              ))}
            </div>
          </div>

          <div className="border-t border-slate-200 dark:border-slate-800" />

          {/* Year */}
          <div>
            <div className="flex items-baseline justify-between mb-2">
              <span className="label">Year</span>
              <span className="text-lg font-semibold tabular-nums text-indigo-600 dark:text-indigo-300">{year}</span>
            </div>
            <input
              type="range" min={snapStart} max={snapEnd} value={year}
              onChange={e => onSelectYear(Number(e.target.value))}
              className="w-full accent-indigo-600"
            />
            <div className="flex items-center gap-1.5 mt-2">
              <button onClick={() => onSelectYear(Math.max(snapStart, year - 1))}
                className="flex-1 py-1 rounded bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400 hover:bg-slate-200 dark:hover:bg-slate-700 text-xs">−</button>
              <button onClick={() => setPlaying(p => !p)}
                className="flex-1 py-1 rounded bg-indigo-600 text-white hover:bg-indigo-500 flex items-center justify-center gap-1 text-xs">
                {playing ? <PauseIcon /> : <PlayIcon />}{playing ? 'Pause' : 'Play'}
              </button>
              <button onClick={() => onSelectYear(Math.min(snapEnd, year + 1))}
                className="flex-1 py-1 rounded bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400 hover:bg-slate-200 dark:hover:bg-slate-700 text-xs">+</button>
            </div>
          </div>

          <div className="border-t border-slate-200 dark:border-slate-800" />

          {/* Country facts */}
          <div className="space-y-1.5 text-xs">
            <div className="label mb-1">Calibration</div>
            <Row k="TFR pre → post" v={`${meta.tfrHigh} → ${meta.tfrLow}`} />
            <Row k="Fertility shock" v={meta.shockYear} />
            <Row k="r init / terminal" v={`${meta.rInit}% / ${meta.rTerminal}%`} />
            <Row k="Pop. growth post" v={`${meta.gLow}%`} />
          </div>

          <div className="border-t border-slate-200 dark:border-slate-800" />

          {/* Legend */}
          <div className="space-y-1.5 text-xs">
            <div className="label mb-1">Life stage</div>
            <LegendRow swatch="#94a3b8" label={`Pre-work (0–${meta.chi})`} />
            <LegendRow swatch="#6366f1" label={`Working (${meta.chi}–${meta.psi})`} />
            <LegendRow swatch="#fbbf24" label={`Retired (${meta.psi}+)`} />
          </div>

          <div className="border-t border-slate-200 dark:border-slate-800" />

          {/* Downloads */}
          <div className="space-y-1.5">
            <div className="label mb-1">Download</div>
            <button onClick={downloadRate}
              className="w-full text-xs py-1.5 px-2 rounded-md font-medium text-left bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400 hover:bg-slate-200 dark:hover:bg-slate-700 flex items-center gap-1.5">
              <DownloadIcon /> Rate path (CSV)
            </button>
            <button onClick={downloadSnapshot}
              className="w-full text-xs py-1.5 px-2 rounded-md font-medium text-left bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400 hover:bg-slate-200 dark:hover:bg-slate-700 flex items-center gap-1.5">
              <DownloadIcon /> Age profile {year} (CSV)
            </button>
          </div>
        </div>
      </div>

      {/* ── Main column ── */}
      <div className="flex-1 min-w-0 flex flex-col gap-3 md:overflow-hidden md:min-h-0">

        {/* stat cards */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-2.5">
          {statCards.map(s => (
            <div key={s.label} className="card px-3 py-2">
              <div className="label">{s.label}</div>
              <div className={`text-xl font-semibold mt-0.5 ${s.tone}`}>
                {s.value}{s.sub && <span className="text-xs font-normal text-slate-400 ml-1">{s.sub}</span>}
              </div>
            </div>
          ))}
        </div>

        {/* navigator */}
        <div className="panel p-3 flex flex-col md:flex-2 md:min-h-[170px]">
          <div className="flex items-baseline justify-between mb-1">
            <span className="label">Real interest rate transition · {meta.name}</span>
            <span className="hidden sm:block text-xs text-slate-400 dark:text-slate-600">Click or drag to select a year</span>
          </div>
          <div className="flex-1 min-h-[150px]">
            <RateNavigatorChart
              rPath={data.rPath} shockYear={meta.shockYear} trough={meta.trough}
              rInit={meta.rInit} rTerminal={meta.rTerminal}
              selectedYear={year} onSelectYear={onSelectYear} color={meta.color}
            />
          </div>
        </div>

        {/* linked snapshot charts */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 md:flex-3 md:min-h-[220px]">
          <div className="panel p-3 flex flex-col min-h-[240px]">
            <span className="label mb-1">Population by age · {year}</span>
            <div className="flex-1 min-h-0">
              <AgePyramidChart ages={data.ages} pop={snap?.pop} chi={meta.chi} psi={meta.psi} popMax={meta.popMax} />
            </div>
          </div>
          <div className="panel p-3 flex flex-col min-h-[240px]">
            <span className="label mb-1">Assets by age · {year}</span>
            <div className="flex-1 min-h-0">
              <AssetByAgeChart
                ages={data.ages} pop={snap?.pop} assetPc={snap?.assetPc}
                chi={meta.chi} psi={meta.psi} assetPcMax={meta.assetPcMax} wealthMax={meta.wealthMax}
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

function Row({ k, v }) {
  return (
    <div className="flex justify-between gap-2">
      <span className="text-slate-500 dark:text-slate-500">{k}</span>
      <span className="text-slate-700 dark:text-slate-300 font-medium tabular-nums text-right">{v}</span>
    </div>
  )
}

function LegendRow({ swatch, label }) {
  return (
    <div className="flex items-center gap-2 text-slate-500 dark:text-slate-400">
      <span className="inline-block w-3 h-3 rounded-sm" style={{ background: swatch }} />
      {label}
    </div>
  )
}
