<script setup lang="ts">
import type { PanelCube } from '~/composables/useApi'

const { state } = useGateway<PanelCube>(
  () => '/api/panel?limit_symbols=16&limit_times=48',
)
</script>

<template>
  <div>
    <h1 class="title">Cross-sectional panel</h1>
    <p class="lede">
      Every symbol's feature rank, at every bar. This is the change that made the
      model worth training: v1 fitted one model per ticker on about 1,200 rows,
      which forecloses a sub-1% effect before any modelling happens. Pooling the
      cross-section turns the question from "what does this stock do?" into
      "what do stocks do relative to each other right now?".
    </p>
    <p class="lede">
      Height and colour both encode rank, so the surface still reads with all hue
      removed. Drag to rotate.
    </p>

    <StateBlock :state="state" what="the panel">
      <template v-if="state.status === 'ok'">
        <p class="meta mono">
          {{ state.data.feature }} · {{ state.data.symbols.length }} symbols ×
          {{ state.data.timestamps.length }} bars
        </p>
        <PanelCube :cube="state.data" />
      </template>
    </StateBlock>
  </div>
</template>

<style scoped>
.title { font-size: var(--text-xl); margin-bottom: var(--space-2); }
.lede {
  color: var(--text-secondary);
  font-size: var(--text-sm);
  max-width: 68ch;
  margin-bottom: var(--space-3);
}
.meta {
  font-size: var(--text-xs);
  color: var(--text-muted);
  margin-bottom: var(--space-2);
}
</style>
