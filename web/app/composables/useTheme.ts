/**
 * Theme and direction-palette preference.
 *
 * Both persist, because a viewer who needs the colour-vision-safe palette
 * needs it on every visit, not once.
 */

type Theme = 'light' | 'dark'
type Palette = 'default' | 'cvd'

const THEME_KEY = 'intraday.theme'
const PALETTE_KEY = 'intraday.palette'

export function useTheme() {
  const theme = useState<Theme>('theme', () => 'dark')

  onMounted(() => {
    const stored = localStorage.getItem(THEME_KEY) as Theme | null
    theme.value =
      stored ??
      (window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark')
    document.documentElement.dataset.theme = theme.value
  })

  return theme
}

export function toggleTheme() {
  const theme = useState<Theme>('theme')
  theme.value = theme.value === 'dark' ? 'light' : 'dark'
  document.documentElement.dataset.theme = theme.value
  localStorage.setItem(THEME_KEY, theme.value)
}

export function usePalette() {
  const palette = useState<Palette>('palette', () => 'default')

  onMounted(() => {
    const stored = localStorage.getItem(PALETTE_KEY) as Palette | null
    if (stored) {
      palette.value = stored
      document.documentElement.dataset.palette = stored
    }
  })

  return palette
}

export function togglePalette() {
  const palette = useState<Palette>('palette')
  palette.value = palette.value === 'cvd' ? 'default' : 'cvd'
  document.documentElement.dataset.palette = palette.value
  localStorage.setItem(PALETTE_KEY, palette.value)
}
