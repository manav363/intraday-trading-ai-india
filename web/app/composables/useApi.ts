/**
 * Gateway client.
 *
 * The whole point of this module is that **loading, empty and failed are three
 * different states** and nothing here lets them collapse into one.
 *
 * The bug being designed out: `const ok = data?.model.trained` renders `false`
 * while the request is still in flight, so a page states "untrained" as a fact
 * for the first seconds of every load. In a project whose entire thesis is
 * honesty about uncertainty, that is the UI lying for free. An unknown must
 * never borrow the weight of a finding.
 */

export type ApiState<T> =
  | { status: 'loading' }
  | { status: 'ok'; data: T }
  | { status: 'empty'; reason: string }
  | { status: 'failed'; code: string; message: string }

export interface ErrorDetail {
  code: string
  message: string
  detail?: Record<string, unknown>
}

export interface Significance {
  p_value: number
  n_permutations: number
  is_significant: boolean
  label: string
  verdict: string
  detail: string
}

export interface SystemStatus {
  lake_present: boolean
  model_trained: boolean
  symbols_available: number
  significance: Significance | null
  trained_on_synthetic: boolean | null
}

export interface Bar {
  t: string
  o: number
  h: number
  l: number
  c: number
  v: number
}

export interface Prediction {
  symbol: string
  as_of: string
  horizon_bars: number
  p_up: number
  certainty: number
  meta_probability: number | null
  side: 'long' | 'short' | 'flat'
  size_fraction: number
  is_calibrated: boolean
}

export interface Narrative {
  headline: string
  failure_framing: string
  meta_note: string | null
  calibration_caveat: string | null
  reasons: string[]
}

export interface ScreenRow {
  symbol: string
  side: 'long' | 'short' | 'flat'
  p_up: number
  certainty: number
  meta_probability: number | null
  size_fraction: number
  headline: string
}

export interface ModelCard {
  model_id: string
  trained_at: string
  n_samples: number
  n_symbols: number
  n_features: number
  oos_accuracy: number
  oos_auc: number
  brier_score: number
  permutation_p_value: number
  permutation_n: number
  baseline_oos_auc: number | null
  deflated_sharpe: number | null
  trained_on_synthetic: boolean
}

export interface PanelCube {
  feature: string
  available_features: string[]
  symbols: string[]
  timestamps: string[]
  /** null means "no value for this cell" — never 0, which is a real rank. */
  values: (number | null)[][]
}

function apiBase(): string {
  return useRuntimeConfig().public.apiBase as string
}

/**
 * Fetch a gateway endpoint into an explicit three-state result.
 *
 * A 503 is surfaced as `failed` with its code, not swallowed into an empty
 * render — the difference between "no model has been trained" and "there is
 * nothing to show" is the most important thing this UI communicates.
 *
 * **Deliberately client-side `$fetch` rather than `useAsyncData`.** The Nuxt
 * idiom for first-paint data is `useAsyncData`, and that would give SSR. It is
 * not used here because the gateway is a separate service that is routinely
 * down during development and on a cold stack: server-rendering against it
 * either blocks the page or bakes a stale error into the HTML. Fetching on the
 * client means the shell always renders and the gateway's real state — pending,
 * untrained, unreachable — is shown as a first-class state instead of a
 * server-side 500.
 *
 * The trade is no SSR data on these routes. For a dashboard behind a private
 * API that is the right side of it; if this ever needs indexable content, the
 * swap is `useAsyncData` with its `status` mapped onto `ApiState`.
 */
export function useGateway<T>(path: () => string, options: { immediate?: boolean } = {}) {
  const state = ref<ApiState<T>>({ status: 'loading' })

  async function load() {
    state.value = { status: 'loading' }
    try {
      const response = await $fetch<T>(`${apiBase()}${path()}`)
      state.value = { status: 'ok', data: response }
    } catch (error: unknown) {
      const err = error as { data?: { detail?: ErrorDetail }; message?: string }
      const detail = err?.data?.detail
      state.value = {
        status: 'failed',
        code: detail?.code ?? 'unreachable',
        message:
          detail?.message ??
          err?.message ??
          'The gateway could not be reached. Is the stack running?',
      }
    }
  }

  if (options.immediate !== false && import.meta.client) {
    void load()
  }

  return { state, load }
}

export function formatPercent(value: number, digits = 1): string {
  return `${(value * 100).toFixed(digits)}%`
}

/**
 * Render a missing number as an em-dash.
 *
 * A missing value is not zero. Rendering `null` as `0.00%` states a fact that
 * was never measured, and a confident green +0.00% is worse than a blank.
 */
export function orDash(value: number | null | undefined, format: (v: number) => string): string {
  return value === null || value === undefined || Number.isNaN(value) ? '—' : format(value)
}
