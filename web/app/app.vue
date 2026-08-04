<script setup lang="ts">
import type { SystemStatus } from '~/composables/useApi'

const { state } = useGateway<SystemStatus>(() => '/api/status')
</script>

<template>
  <div class="shell">
    <StatusBar :state="state" />
    <main class="shell__main">
      <NuxtPage />
    </main>
    <footer class="shell__footer">
      <p>
        Research and education only. Not investment advice, and no order is ever
        placed. Signals are generated from public NSE data for personal,
        non-commercial use.
      </p>
    </footer>
  </div>
</template>

<style scoped>
.shell {
  min-height: 100dvh;
  display: flex;
  flex-direction: column;
}
.shell__main {
  flex: 1 1 auto;
  width: 100%;
  max-width: 1180px;
  margin: 0 auto;
  padding: var(--space-6) var(--space-4) var(--space-12);
}
.shell__footer {
  border-top: 1px solid var(--border);
  padding: var(--space-4);
  font-size: var(--text-xs);
  color: var(--text-muted);
  text-align: center;
}
.shell__footer p {
  max-width: 62ch;
  margin: 0 auto;
}
</style>
