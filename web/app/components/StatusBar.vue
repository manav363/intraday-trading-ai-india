<script setup lang="ts">
/**
 * The status bar.
 *
 * The permutation p-value and its verdict sit here on every screen, not on a
 * `/model` page. A `/model` page is a page nobody opens, and a system that
 * shows predictions without showing whether they are distinguishable from
 * chance is hiding the single most important number about itself. Beside the
 * ticker, it cannot be read past.
 *
 * Note the pending state is visually *weaker* than either finding. "We don't
 * know yet" must not borrow the weight of "not significant".
 */
import type { ApiState, SystemStatus } from '~/composables/useApi'

defineProps<{ state: ApiState<SystemStatus> }>()

const theme = useTheme()
const palette = usePalette()
</script>

<template>
  <div class="bar">
    <div class="bar__left">
      <NuxtLink to="/" class="bar__brand">
        <span class="bar__mark" aria-hidden="true">◫</span>
        <span>intraday<span class="bar__dim">/nse</span></span>
      </NuxtLink>

      <nav class="bar__nav" aria-label="Sections">
        <NuxtLink to="/">terminal</NuxtLink>
        <NuxtLink to="/screen">screen</NuxtLink>
        <NuxtLink to="/panel">panel</NuxtLink>
        <NuxtLink to="/method">method</NuxtLink>
      </nav>
    </div>

    <div class="bar__right">
      <!-- Three states, never collapsed. -->
      <span v-if="state.status === 'loading'" class="pill pill--pending">
        <span class="pill__dot" aria-hidden="true" />
        checking model…
      </span>

      <span v-else-if="state.status === 'failed'" class="pill pill--failed">
        <span aria-hidden="true">⚠</span>
        gateway unreachable
      </span>

      <template v-else-if="state.status === 'ok'">
        <span v-if="state.data.trained_on_synthetic" class="pill pill--warn">
          <span aria-hidden="true">⚗</span>
          synthetic data
        </span>

        <span v-if="!state.data.model_trained" class="pill pill--failed">
          <span aria-hidden="true">⚠</span>
          no model trained
        </span>

        <span
          v-else-if="state.data.significance"
          class="pill"
          :class="
            state.data.significance.is_significant ? 'pill--sig' : 'pill--notsig'
          "
          :title="state.data.significance.detail"
        >
          <span class="mono">{{ state.data.significance.label }}</span>
          <span>{{ state.data.significance.verdict }}</span>
        </span>
      </template>

      <button
        class="toggle"
        type="button"
        :aria-pressed="palette === 'cvd'"
        title="Switch the up/down colours to a blue/orange pair with far higher colour-vision separation"
        @click="togglePalette()"
      >
        {{ palette === 'cvd' ? 'blue/orange' : 'green/red' }}
      </button>

      <button
        class="toggle"
        type="button"
        :title="`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`"
        @click="toggleTheme()"
      >
        {{ theme === 'dark' ? '☾' : '☀' }}
      </button>
    </div>
  </div>
</template>

<style scoped>
.bar {
  position: sticky;
  top: 0;
  z-index: 20;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--space-4);
  height: var(--status-bar-height);
  padding: 0 var(--space-4);
  background: var(--surface-1);
  border-bottom: 1px solid var(--border);
  font-size: var(--text-xs);
}

.bar__left,
.bar__right {
  display: flex;
  align-items: center;
  gap: var(--space-4);
  min-width: 0;
}

.bar__brand {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  color: var(--text-primary);
  text-decoration: none;
  font-weight: 600;
  letter-spacing: -0.01em;
}
.bar__mark {
  color: var(--accent);
}
.bar__dim {
  color: var(--text-muted);
  font-weight: 400;
}

.bar__nav {
  display: flex;
  gap: var(--space-3);
}
.bar__nav a {
  color: var(--text-secondary);
  text-decoration: none;
  padding: 2px 0;
  border-bottom: 1px solid transparent;
}
.bar__nav a:hover {
  color: var(--text-primary);
}
.bar__nav a.router-link-exact-active {
  color: var(--text-primary);
  border-bottom-color: var(--accent);
}

.pill {
  display: inline-flex;
  align-items: center;
  gap: var(--space-2);
  padding: 2px var(--space-2);
  border-radius: var(--radius-sm);
  border: 1px solid var(--border);
  white-space: nowrap;
}

/* Pending is deliberately the quietest thing on the bar. */
.pill--pending {
  color: var(--text-muted);
  border-style: dashed;
}
.pill__dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: currentColor;
  animation: pulse 1.4s ease-in-out infinite;
}
@keyframes pulse {
  0%,
  100% {
    opacity: 0.3;
  }
  50% {
    opacity: 1;
  }
}

.pill--sig {
  color: var(--dir-up);
  border-color: var(--dir-up);
}
.pill--notsig {
  color: var(--text-secondary);
  border-color: var(--border-strong);
}
.pill--failed {
  color: var(--dir-down);
  border-color: var(--dir-down);
}
.pill--warn {
  color: var(--warn);
  border-color: var(--warn);
}

.toggle {
  background: transparent;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  color: var(--text-secondary);
  padding: 2px var(--space-2);
  cursor: pointer;
  font-size: var(--text-xs);
}
.toggle:hover {
  color: var(--text-primary);
  border-color: var(--border-strong);
}

@media (max-width: 720px) {
  .bar {
    height: auto;
    flex-wrap: wrap;
    padding: var(--space-2) var(--space-3);
  }
}
</style>
