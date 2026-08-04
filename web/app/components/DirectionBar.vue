<script setup lang="ts">
/**
 * Direction, encoded four ways.
 *
 * The measured CVD separation of the light green/red pair is ΔE 7.2 (protan),
 * which sits in the 6–8 band the validator calls legal ONLY with secondary
 * encoding. So direction is never carried by hue alone:
 *
 *   1. colour    — the fill
 *   2. glyph     — ▲ / ▼ / ●
 *   3. text      — the word "up" / "down" / "no call"
 *   4. position  — the bar extends right of centre for up, left for down
 *
 * Acceptance test: set every direction token to the same value and this
 * component still reads correctly. `DirectionBar.test` covers that.
 */

const props = withDefaults(
  defineProps<{
    /** P(up). 0.5 is neutral. */
    probability: number
    side: 'long' | 'short' | 'flat'
    /** Show the numeric probability beside the bar. */
    showValue?: boolean
    compact?: boolean
  }>(),
  { showValue: true, compact: false },
)

const glyph = computed(() =>
  props.side === 'long' ? '▲' : props.side === 'short' ? '▼' : '●',
)

const word = computed(() =>
  props.side === 'long' ? 'up' : props.side === 'short' ? 'down' : 'no call',
)

const tone = computed(() =>
  props.side === 'long' ? 'up' : props.side === 'short' ? 'down' : 'flat',
)

/** Distance from the 50% centre line, as a percentage of half-width. */
const extent = computed(() => Math.min(100, Math.abs(props.probability - 0.5) * 200))

const label = computed(
  () =>
    `${word.value}, probability ${(props.probability * 100).toFixed(0)} percent`,
)
</script>

<template>
  <div class="dir" :class="[`dir--${tone}`, { 'dir--compact': compact }]">
    <span class="dir__glyph" aria-hidden="true">{{ glyph }}</span>

    <div class="dir__track" role="img" :aria-label="label">
      <span class="dir__centre" aria-hidden="true" />
      <span
        class="dir__fill"
        :class="`dir__fill--${tone}`"
        :style="{ width: `${extent}%` }"
        aria-hidden="true"
      />
    </div>

    <span class="dir__word">{{ word }}</span>
    <span v-if="showValue" class="dir__value mono">
      {{ (probability * 100).toFixed(0) }}%
    </span>
  </div>
</template>

<style scoped>
.dir {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  min-width: 0;
}

.dir__glyph {
  font-size: var(--text-sm);
  line-height: 1;
  color: var(--dir-flat);
}
.dir--up .dir__glyph {
  color: var(--dir-up);
}
.dir--down .dir__glyph {
  color: var(--dir-down);
}

.dir__track {
  position: relative;
  flex: 1 1 auto;
  min-width: 60px;
  height: 10px;
  background: var(--surface-2);
  border-radius: var(--radius-sm);
  overflow: hidden;
}
.dir--compact .dir__track {
  height: 7px;
}

.dir__centre {
  position: absolute;
  left: 50%;
  top: 0;
  bottom: 0;
  width: 1px;
  background: var(--border-strong);
}

/* Position is the fourth encoding: up grows right of centre, down grows
   left. With all hue removed the direction is still unambiguous. */
.dir__fill {
  position: absolute;
  top: 0;
  bottom: 0;
  border-radius: var(--radius-sm);
  transition: width var(--duration) var(--ease-out);
}
.dir__fill--up {
  left: 50%;
  background: var(--dir-up);
}
.dir__fill--down {
  right: 50%;
  background: var(--dir-down);
}
.dir__fill--flat {
  left: 50%;
  width: 0 !important;
}

.dir__word {
  font-size: var(--text-xs);
  text-transform: uppercase;
  letter-spacing: 0.06em;
  /* Text wears text tokens, never the series colour. The coloured bar beside
     it carries identity. */
  color: var(--text-secondary);
  white-space: nowrap;
}

.dir__value {
  font-size: var(--text-sm);
  color: var(--text-primary);
  min-width: 3ch;
  text-align: right;
}
</style>
