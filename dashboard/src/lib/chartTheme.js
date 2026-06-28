// Central Recharts palette for dark/light. Reference colours by semantic key
// from charts so a single edit re-skins every figure.
export const DARK_THEME = {
  colors: {
    primary:   '#6366f1', // indigo  — main model series
    accent:    '#22d3ee', // cyan    — secondary series
    positive:  '#34d399', // emerald
    warning:   '#fbbf24', // amber
    muted:     '#cbd5e1', // slate   — data / reference line
    band:      'rgba(99,102,241,0.18)',
  },
  strokeWidths: {
    line:      1.8,
    reference: 1.25,
  },
  ui: {
    grid:          'rgba(51,65,85,0.4)',
    axis:          'rgba(51,65,85,0.6)',
    tickLabel:     '#cbd5e1',
    tickFontSize:  10,
    tooltipBg:     '#0f172a',
    tooltipBorder: 'rgba(51,65,85,0.6)',
    tooltipText:   '#cbd5e1',
  },
}

export const LIGHT_THEME = {
  colors: {
    primary:   '#4f46e5',
    accent:    '#0891b2',
    positive:  '#059669',
    warning:   '#d97706',
    muted:     '#334155',
    band:      'rgba(79,70,229,0.12)',
  },
  strokeWidths: DARK_THEME.strokeWidths,
  ui: {
    grid:          'rgba(203,213,225,0.6)',
    axis:          'rgba(203,213,225,0.8)',
    tickLabel:     '#1e293b',
    tickFontSize:  10,
    tooltipBg:     '#ffffff',
    tooltipBorder: 'rgba(203,213,225,0.8)',
    tooltipText:   '#1e293b',
  },
}

export function getTheme(isDark) {
  return isDark ? DARK_THEME : LIGHT_THEME
}

export function getTooltipStyle(isDark) {
  const ui = isDark ? DARK_THEME.ui : LIGHT_THEME.ui
  return {
    backgroundColor: ui.tooltipBg,
    border:          `1px solid ${ui.tooltipBorder}`,
    borderRadius:    '6px',
    fontSize:        '11px',
    color:           ui.tooltipText,
  }
}
