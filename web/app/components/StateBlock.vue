<script setup lang="ts">
/**
 * Renders the three states explicitly.
 *
 * Exists so no page can accidentally collapse them. A bare heading over empty
 * space reads as "nothing to show" when the truth is "still working", and
 * optional chaining with a falsy fallback turns "loading" into "failed".
 */
import type { ApiState } from '~/composables/useApi'

defineProps<{ state: ApiState<unknown>; what?: string }>()
</script>

<template>
  <div v-if="state.status === 'loading'" class="block block--loading">
    <span class="block__dot" aria-hidden="true" />
    <span>Loading{{ what ? ` ${what}` : '' }}…</span>
  </div>

  <div v-else-if="state.status === 'failed'" class="block block--failed" role="alert">
    <strong>{{ state.code === 'no_model' ? 'No model trained' : 'Could not load' }}</strong>
    <p>{{ state.message }}</p>
  </div>

  <div v-else-if="state.status === 'empty'" class="block block--empty">
    <p>{{ state.reason }}</p>
  </div>

  <slot v-else />
</template>

<style scoped>
.block {
  padding: var(--space-6);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--surface-1);
  font-size: var(--text-sm);
}
.block--loading {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  color: var(--text-muted);
  border-style: dashed;
}
.block__dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: currentColor;
  animation: pulse 1.4s ease-in-out infinite;
}
@keyframes pulse {
  0%, 100% { opacity: 0.3; }
  50% { opacity: 1; }
}
.block--failed {
  border-color: var(--dir-down);
  color: var(--text-primary);
}
.block--failed p {
  margin-top: var(--space-2);
  color: var(--text-secondary);
}
.block--empty {
  color: var(--text-secondary);
}
</style>
