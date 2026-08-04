<script setup lang="ts">
/**
 * The terminal.
 *
 * The product flow is: type a ticker, see its chart and its call. Everything
 * else on this page is subordinate to that, and nothing is allowed to bury it.
 */
import type { Bar, Narrative, Prediction } from '~/composables/useApi'

const symbol = ref('')
const active = ref('')

const universe = useGateway<{ symbols: string[] }>(() => '/api/universe')

const bars = useGateway<{ bars: Bar[] }>(
  () => `/api/bars/${active.value}?limit=180`,
  { immediate: false },
)
const call = useGateway<{ prediction: Prediction; narrative: Narrative }>(
  () => `/api/predict/${active.value}`,
  { immediate: false },
)

function submit() {
  const next = symbol.value.trim().toUpperCase()
  if (!next) return
  active.value = next
  void bars.load()
  void call.load()
}

const suggestions = computed(() =>
  universe.state.value.status === 'ok' ? universe.state.value.data.symbols : [],
)

onMounted(() => {
  // Pick the first eligible symbol so the terminal is never a blank screen.
  const stop = watch(
    suggestions,
    (list) => {
      if (list.length && !active.value) {
        symbol.value = list[0]!
        submit()
        stop()
      }
    },
    { immediate: true },
  )
})
</script>

<template>
  <div class="terminal">
    <form class="search" @submit.prevent="submit">
      <label for="ticker" class="visually-hidden">Stock symbol</label>
      <span class="search__prompt mono" aria-hidden="true">&gt;</span>
      <input
        id="ticker"
        v-model="symbol"
        class="search__input mono"
        list="universe"
        autocomplete="off"
        spellcheck="false"
        placeholder="RELIANCE"
      />
      <datalist id="universe">
        <option v-for="s in suggestions" :key="s" :value="s" />
      </datalist>
      <button type="submit" class="search__go">analyse</button>
    </form>

    <p class="search__note">
      Only symbols with enough clean history are offered. A ticker the system
      cannot answer honestly is absent rather than answered with a guess.
    </p>

    <section v-if="active" class="panel" aria-labelledby="call-heading">
      <h2 id="call-heading" class="panel__title">
        <span class="mono">{{ active }}</span>
      </h2>

      <StateBlock :state="call.state.value" what="the call">
        <template v-if="call.state.value.status === 'ok'">
          <div class="verdict">
            <DirectionBar
              :probability="call.state.value.data.prediction.p_up"
              :side="call.state.value.data.prediction.side"
            />

            <p class="verdict__headline">
              {{ call.state.value.data.narrative.headline }}
            </p>

            <!-- Every probability is immediately followed by what it means in
                 failure terms. Shown alone, a probability reads as a promise. -->
            <p class="verdict__framing">
              {{ call.state.value.data.narrative.failure_framing }}
            </p>

            <p
              v-if="call.state.value.data.narrative.meta_note"
              class="verdict__meta"
            >
              {{ call.state.value.data.narrative.meta_note }}
            </p>

            <p
              v-if="call.state.value.data.narrative.calibration_caveat"
              class="verdict__caveat"
            >
              ⚠ {{ call.state.value.data.narrative.calibration_caveat }}
            </p>

            <ul
              v-if="call.state.value.data.narrative.reasons.length"
              class="verdict__reasons"
            >
              <li v-for="r in call.state.value.data.narrative.reasons" :key="r">
                {{ r }}
              </li>
            </ul>

            <dl class="stats">
              <div>
                <dt>size</dt>
                <dd class="mono">
                  {{ formatPercent(call.state.value.data.prediction.size_fraction) }}
                </dd>
              </div>
              <div>
                <dt>certainty</dt>
                <dd class="mono">
                  {{ formatPercent(call.state.value.data.prediction.certainty) }}
                </dd>
              </div>
              <div>
                <dt>P(correct)</dt>
                <dd class="mono">
                  {{ orDash(call.state.value.data.prediction.meta_probability, (v) => formatPercent(v)) }}
                </dd>
              </div>
              <div>
                <dt>horizon</dt>
                <dd class="mono">
                  {{ call.state.value.data.prediction.horizon_bars }} bars
                </dd>
              </div>
            </dl>
          </div>
        </template>
      </StateBlock>
    </section>

    <section v-if="active" class="panel" aria-labelledby="chart-heading">
      <h2 id="chart-heading" class="panel__title">price</h2>
      <StateBlock :state="bars.state.value" what="bars">
        <template v-if="bars.state.value.status === 'ok'">
          <p v-if="!bars.state.value.data.bars.length" class="muted">
            No bars in this window.
          </p>
          <CandleChart v-else :bars="bars.state.value.data.bars" />
        </template>
      </StateBlock>
    </section>
  </div>
</template>

<style scoped>
.terminal {
  display: flex;
  flex-direction: column;
  gap: var(--space-6);
}

.search {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  background: var(--surface-1);
  border: 1px solid var(--border-strong);
  border-radius: var(--radius);
  padding: var(--space-2) var(--space-3);
}
.search:focus-within {
  border-color: var(--accent);
}
.search__prompt {
  color: var(--accent);
}
.search__input {
  flex: 1 1 auto;
  background: transparent;
  border: 0;
  outline: none;
  font-size: var(--text-lg);
  letter-spacing: 0.04em;
  text-transform: uppercase;
  min-width: 0;
}
.search__go {
  background: var(--text-primary);
  color: var(--surface-0);
  border: 0;
  border-radius: var(--radius-sm);
  padding: var(--space-1) var(--space-3);
  font-size: var(--text-xs);
  text-transform: uppercase;
  letter-spacing: 0.06em;
  cursor: pointer;
}
.search__note {
  font-size: var(--text-xs);
  color: var(--text-muted);
  margin-top: calc(var(--space-4) * -1 + 2px);
  max-width: 68ch;
}

.panel__title {
  font-size: var(--text-xs);
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--text-muted);
  margin-bottom: var(--space-3);
}

.verdict {
  display: flex;
  flex-direction: column;
  gap: var(--space-3);
  padding: var(--space-4);
  background: var(--surface-1);
  border: 1px solid var(--border);
  border-radius: var(--radius);
}
.verdict__headline {
  font-size: var(--text-xl);
  font-weight: 600;
  letter-spacing: -0.015em;
  max-width: 40ch;
}
.verdict__framing {
  color: var(--text-secondary);
  max-width: 62ch;
}
.verdict__meta,
.verdict__caveat {
  font-size: var(--text-sm);
  color: var(--text-secondary);
  max-width: 62ch;
}
.verdict__caveat {
  color: var(--warn);
}
.verdict__reasons {
  margin: 0;
  padding-left: var(--space-4);
  color: var(--text-secondary);
  font-size: var(--text-sm);
  display: flex;
  flex-direction: column;
  gap: var(--space-1);
}

.stats {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(110px, 1fr));
  gap: var(--space-3);
  margin: var(--space-2) 0 0;
  padding-top: var(--space-3);
  border-top: 1px solid var(--border);
}
.stats dt {
  font-size: var(--text-xs);
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--text-muted);
}
.stats dd {
  margin: 2px 0 0;
  font-size: var(--text-lg);
}

.muted {
  color: var(--text-muted);
  font-size: var(--text-sm);
}
</style>
