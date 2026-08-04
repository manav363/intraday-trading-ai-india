<script setup lang="ts">
import type { ScreenRow } from '~/composables/useApi'

const { state } = useGateway<{
  rows: ScreenRow[]
  scored: number
  universe_size: number
}>(() => '/api/screen?limit=40')
</script>

<template>
  <div>
    <h1 class="title">Screener</h1>
    <p class="lede">
      Ranked by conviction across the eligible universe. Symbols the model
      cannot score are absent rather than listed with a placeholder — a
      placeholder in a ranked table reads as a real ranking.
    </p>

    <StateBlock :state="state" what="the screen">
      <template v-if="state.status === 'ok'">
        <p class="count">
          {{ state.data.scored }} of {{ state.data.universe_size }} symbols scored
        </p>

        <div class="scroll-x">
          <table>
            <caption class="visually-hidden">Ranked intraday calls</caption>
            <thead>
              <tr>
                <th scope="col">symbol</th>
                <th scope="col">direction</th>
                <th scope="col">certainty</th>
                <th scope="col">P(correct)</th>
                <th scope="col">size</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="row in state.data.rows" :key="row.symbol">
                <th scope="row" class="mono sym">{{ row.symbol }}</th>
                <td class="dir-cell">
                  <DirectionBar
                    :probability="row.p_up"
                    :side="row.side"
                    compact
                    :show-value="false"
                  />
                </td>
                <td class="mono num">{{ formatPercent(row.certainty) }}</td>
                <td class="mono num">
                  {{ orDash(row.meta_probability, (v) => formatPercent(v)) }}
                </td>
                <td class="mono num">{{ formatPercent(row.size_fraction) }}</td>
              </tr>
            </tbody>
          </table>
        </div>

        <p v-if="!state.data.rows.length" class="muted">
          No symbol currently clears the conviction threshold. That is a result,
          not an error.
        </p>
      </template>
    </StateBlock>
  </div>
</template>

<style scoped>
.title {
  font-size: var(--text-xl);
  margin-bottom: var(--space-2);
}
.lede {
  color: var(--text-secondary);
  font-size: var(--text-sm);
  max-width: 68ch;
  margin-bottom: var(--space-6);
}
.count {
  font-size: var(--text-xs);
  color: var(--text-muted);
  margin-bottom: var(--space-2);
}
table {
  border-collapse: collapse;
  width: 100%;
  min-width: 620px;
}
th, td {
  padding: var(--space-2) var(--space-3);
  border-bottom: 1px solid var(--border);
  text-align: left;
}
thead th {
  font-size: var(--text-xs);
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--text-muted);
  font-weight: 500;
}
tbody tr:hover {
  background: var(--surface-1);
}
.sym { font-weight: 600; }
.num { text-align: right; }
.dir-cell { min-width: 180px; }
.muted { color: var(--text-muted); font-size: var(--text-sm); margin-top: var(--space-4); }
</style>
