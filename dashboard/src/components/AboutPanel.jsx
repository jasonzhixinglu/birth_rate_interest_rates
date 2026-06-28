const BASE = import.meta.env.BASE_URL
const PAPER_PDF = `${BASE}paper/index.pdf`

export default function AboutPanel() {
  return (
    <div className="max-w-3xl mx-auto py-6 space-y-6">

      <div className="panel p-6 space-y-4">
        <div>
          <h2 className="text-xl font-semibold text-slate-900 dark:text-white leading-snug">
            Birth Rate Collapse and Its Effect on Interest Rates
          </h2>
          <p className="text-sm text-slate-500 mt-1">
            A Python replication of Lu &amp; Teulings (2015), <em>Fertility Rates and the Age Distribution</em>
          </p>
        </div>

        <div className="border-t border-slate-200 dark:border-slate-800 pt-4 space-y-3">
          <div className="flex flex-col sm:flex-row sm:gap-8 gap-2">
            <div>
              <div className="text-sm font-medium text-slate-800 dark:text-slate-100">Jason Lu</div>
              <div className="text-xs text-slate-500 mt-0.5">University of Cambridge</div>
            </div>
            <div>
              <div className="text-sm font-medium text-slate-800 dark:text-slate-100">Coen Teulings</div>
              <div className="text-xs text-slate-500 mt-0.5">University of Cambridge</div>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-indigo-50 border border-indigo-200/80 text-xs text-indigo-700 font-medium dark:bg-indigo-950/60 dark:border-indigo-700/40 dark:text-indigo-300">
              Working Paper · first version Sep 2015, last edited Oct 2017
            </span>
            <a
              href={PAPER_PDF}
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-md bg-indigo-600 text-white hover:bg-indigo-500 font-medium"
            >
              <svg width="11" height="11" viewBox="0 0 12 12" fill="none"><path d="M6 1v7M3.5 5.5 6 8l2.5-2.5M2 10h8" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" /></svg>
              Full paper (PDF)
            </a>
          </div>
        </div>
      </div>

      <div className="panel p-6 space-y-3">
        <div className="label">Abstract</div>
        <p className="text-sm text-slate-900 dark:text-white leading-relaxed">
          Real interest rates have fallen roughly four percentage points since the mid-1980s. We show
          that the collapse in birth rates around 1970 accounts for a significant share of this decline:
          the baby boomers — the last large birth cohort — accumulate savings ahead of retirement,
          driving down interest rates to a trough below one percent around 2031, overshooting the new
          long-run equilibrium, itself lower due to negative population growth. We use a large OLG model
          with CES production calibrated to Germany (working age 20–65, pre-shock rate 3.0%), and show
          that the same demographic mechanism, fed with observed fertility profiles, generates
          qualitatively similar transitions for the US, Japan, and China. Relative to the original, this
          replication omits the land extension and the bequest motive.
        </p>
      </div>

      <div className="card p-4">
        <p className="text-xs text-slate-500 leading-relaxed">
          This dashboard is a companion to the replication and presents the model’s output interactively.
          Source code and methodology:{' '}
          <a href="https://github.com/jasonzhixinglu/birth_rate_interest_rates" className="text-indigo-600 dark:text-indigo-400 underline" target="_blank" rel="noreferrer">github.com/jasonzhixinglu/birth_rate_interest_rates</a>.
        </p>
      </div>
    </div>
  )
}
