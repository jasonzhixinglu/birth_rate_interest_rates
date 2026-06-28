import { useEffect } from 'react'
import OverviewPanel from './components/OverviewPanel.jsx'
import CrossCountryPanel from './components/CrossCountryPanel.jsx'
import AboutPanel from './components/AboutPanel.jsx'
import { useSessionState } from './lib/sessionState.js'
import { useDarkMode } from './lib/useDarkMode.jsx'

// ─── Dashboard config ────────────────────────────────────────────────────────
// Add a tab by adding one entry here; the shell renders nav + panel generically.
const TITLE = 'Birth Rate Collapse & the Decline in Interest Rates'
const SUBTITLE = 'Lu & Teulings · OLG replication · interactive companion'

const TABS = [
  { id: 'overview', label: 'Explorer',      sub: 'Rates · cohorts · assets', fullHeight: true,  render: () => <OverviewPanel /> },
  { id: 'cross',    label: 'Cross-country', sub: 'Compare transitions',      fullHeight: false, render: () => <CrossCountryPanel /> },
  { id: 'about',    label: 'About',         sub: 'Paper · Model',            fullHeight: false, render: () => <AboutPanel /> },
]
// ─────────────────────────────────────────────────────────────────────────────

function SunIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="5"/>
      <line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/>
      <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/>
      <line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/>
      <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>
    </svg>
  )
}

function MoonIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
    </svg>
  )
}

export default function App() {
  const [activeTab, setActiveTab] = useSessionState('brir-tab', TABS[0].id)
  const { isDark, toggle } = useDarkMode()

  const activeTabCfg = TABS.find(t => t.id === activeTab) ?? TABS[0]
  const fullHeight = activeTabCfg.fullHeight

  useEffect(() => { document.title = TITLE }, [])

  return (
    <div className={`flex flex-col ${fullHeight ? 'min-h-screen md:h-screen md:overflow-hidden' : 'min-h-screen'}`}>

      <header className="border-b border-slate-200 dark:border-slate-800 px-4 sm:px-6 py-3 sm:py-4">
        <div className="max-w-screen-2xl mx-auto flex flex-col sm:flex-row sm:items-start sm:justify-between gap-2 sm:gap-6">
          <div className="min-w-0">
            <h1 className="text-base font-semibold text-slate-900 dark:text-white leading-tight">{TITLE}</h1>
            <p className="text-xs text-slate-500 mt-0.5">{SUBTITLE}</p>
          </div>
          <div className="flex items-start gap-2 flex-shrink-0">
            <nav className="grid gap-1.5 sm:flex sm:flex-wrap" style={{ gridTemplateColumns: `repeat(${TABS.length}, minmax(0, 1fr))` }}>
              {TABS.map(tab => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`tab-btn text-center lg:text-left px-2 py-1.5 text-xs sm:px-3 lg:px-5 lg:py-2.5 lg:text-sm leading-tight ${
                    activeTab === tab.id ? 'tab-btn-active' : 'tab-btn-inactive'
                  }`}
                >
                  <div>{tab.label}</div>
                  <div className={`text-xs mt-0.5 font-normal hidden lg:block ${
                    activeTab === tab.id ? 'text-indigo-300' : 'text-slate-400 dark:text-slate-600'
                  }`}>{tab.sub}</div>
                </button>
              ))}
            </nav>
            <button
              onClick={toggle}
              className="shrink-0 p-2 rounded-lg text-slate-500 hover:text-slate-700 hover:bg-slate-100 dark:text-slate-400 dark:hover:text-slate-200 dark:hover:bg-slate-800 transition-all"
              aria-label="Toggle dark/light mode"
            >
              {isDark ? <SunIcon /> : <MoonIcon />}
            </button>
          </div>
        </div>
      </header>

      <main className={`flex-1 px-4 sm:px-6 py-4 sm:py-5${fullHeight ? ' md:overflow-hidden md:flex md:flex-col' : ''}`}>
        <div className={`max-w-screen-2xl mx-auto${fullHeight ? ' w-full md:flex-1 md:overflow-hidden md:flex md:flex-col' : ''}`}>
          {activeTabCfg.render()}
        </div>
      </main>

    </div>
  )
}
