<script setup lang="ts">
/**
 * The credibility page.
 *
 * Its job is to make the project's weaknesses easy to find. A system that
 * reports a number without reporting how much to believe it is worth less than
 * one that reports a smaller number honestly.
 */
import type { ModelCard } from '~/composables/useApi'

const { state } = useGateway<ModelCard>(() => '/api/model')
</script>

<template>
  <div class="method">
    <h1 class="title">Method &amp; evidence</h1>
    <p class="lede">
      Everything here is measured out of sample under purged cross-validation.
      Nothing on this page is a number the model produced about data it had
      already seen.
    </p>

    <StateBlock :state="state" what="the model card">
      <template v-if="state.status === 'ok'">
        <!-- The headline is the p-value, not the accuracy. -->
        <section
          class="hero"
          :class="state.data.permutation_p_value < 0.05 ? 'hero--sig' : 'hero--notsig'"
        >
          <p class="hero__label">permutation test</p>
          <p class="hero__value mono">
            p = {{ state.data.permutation_p_value.toFixed(4) }}
          </p>
          <p class="hero__verdict">
            {{
              state.data.permutation_p_value < 0.05
                ? 'The measured edge is distinguishable from chance.'
                : 'The measured edge is NOT distinguishable from chance.'
            }}
          </p>
          <p class="hero__note">
            Labels were shuffled within folds and the whole pipeline re-run
            {{ state.data.permutation_n }} times. With
            {{ state.data.permutation_n }} permutations the smallest reportable
            p-value is {{ (1 / (state.data.permutation_n + 1)).toFixed(4) }} —
            this system cannot report p = 0.
          </p>
        </section>

        <dl class="grid">
          <div>
            <dt>out-of-sample AUC</dt>
            <dd class="mono">{{ state.data.oos_auc.toFixed(4) }}</dd>
            <p>0.5 is a coin flip. A realistic equity edge is a little above it.</p>
          </div>
          <div>
            <dt>out-of-sample accuracy</dt>
            <dd class="mono">{{ formatPercent(state.data.oos_accuracy, 2) }}</dd>
            <p>On a near-balanced binary target, so 50% is the baseline.</p>
          </div>
          <div>
            <dt>Brier score</dt>
            <dd class="mono">{{ state.data.brier_score.toFixed(4) }}</dd>
            <p>Calibration error. 0.25 is a coin flip; lower is better.</p>
          </div>
          <div>
            <dt>ElasticNet baseline AUC</dt>
            <dd class="mono">
              {{ orDash(state.data.baseline_oos_auc, (v) => v.toFixed(4)) }}
            </dd>
            <p>
              A regularised linear model on the same features. If the stack
              cannot beat it, that is the finding.
            </p>
          </div>
          <div>
            <dt>deflated Sharpe</dt>
            <dd class="mono">
              {{ orDash(state.data.deflated_sharpe, (v) => v.toFixed(4)) }}
            </dd>
            <p>
              P(true Sharpe &gt; 0) after discounting for how many
              configurations were tried. Below 0.95 it does not survive its own
              multiple testing.
            </p>
            <p class="footnote">
              This can read high while the permutation test says "not
              significant", and the two are not in conflict: this asks whether
              the Sharpe's <em>sign</em> is reliable, which many samples can
              settle even for a tiny effect. The permutation test asks the
              harder question — whether the features relate to the labels at
              all. When they disagree, believe the permutation test.
            </p>
          </div>
          <div>
            <dt>samples</dt>
            <dd class="mono">{{ state.data.n_samples.toLocaleString() }}</dd>
            <p>
              {{ state.data.n_symbols }} symbols pooled,
              {{ state.data.n_features }} features.
            </p>
          </div>
        </dl>

        <section v-if="state.data.trained_on_synthetic" class="warning">
          <h2>This model was trained on synthetic data</h2>
          <p>
            It is quarantined and can never be the served model. Every figure
            above describes generated bars, not the market — a good-looking
            number here would indicate a leak, not an edge.
          </p>
        </section>

        <section class="prose">
          <h2>What this gets wrong</h2>
          <ul>
            <li>
              <strong>History is short.</strong> The public NSE endpoint serves a
              rolling 60–90 day window, which spans one or two market regimes.
              The model cannot learn regime dependence from it. The lake is
              append-only and grows past that window, but only with calendar
              time.
            </li>
            <li>
              <strong>Microstructure is estimated, not measured.</strong> Real
              order-flow imbalance needs the trade-and-quote tape and a real
              spread needs the book. Both features here are inferred from OHLCV
              and carry the error of that inference.
            </li>
            <li>
              <strong>Costs are material.</strong> A round trip costs roughly
              0.08% before slippage. A profit target below that is not a
              strategy, and the barrier width is chosen with this in mind.
            </li>
            <li>
              <strong>No execution.</strong> This is a research system. It places
              no orders and models no queue position.
            </li>
          </ul>

          <h2>How the numbers are produced</h2>
          <ul>
            <li>
              Labels are triple-barrier: profit-take, stop-loss and time, each
              scaled to that symbol's own past-only volatility. Barriers are
              checked against the bar's high and low, never its close.
            </li>
            <li>
              Folds split on <em>timestamp</em>, so every symbol on a bar lands in
              the same fold. Splitting rows would leak across the cross-section.
            </li>
            <li>
              Training rows whose label resolves inside a test window are purged,
              and an embargo removes rows immediately after it.
            </li>
            <li>
              Overlapping labels are down-weighted by average uniqueness.
              Without that correction, every p-value on this page would be
              inflated.
            </li>
          </ul>
        </section>
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
  margin-bottom: var(--space-6);
}

.hero {
  padding: var(--space-6);
  border: 1px solid var(--border);
  border-left-width: 3px;
  border-radius: var(--radius);
  background: var(--surface-1);
  margin-bottom: var(--space-6);
}
.hero--sig { border-left-color: var(--dir-up); }
.hero--notsig { border-left-color: var(--text-muted); }

.hero__label {
  font-size: var(--text-xs);
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--text-muted);
}
.hero__value {
  font-size: var(--text-hero);
  font-weight: 600;
  letter-spacing: -0.03em;
  line-height: 1.1;
  margin: var(--space-2) 0;
}
.hero__verdict {
  font-size: var(--text-lg);
  max-width: 46ch;
}
.hero__note {
  margin-top: var(--space-3);
  font-size: var(--text-sm);
  color: var(--text-secondary);
  max-width: 64ch;
}

.grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: var(--space-4);
  margin: 0 0 var(--space-8);
}
.grid > div {
  padding: var(--space-4);
  border: 1px solid var(--border);
  border-radius: var(--radius);
}
.grid dt {
  font-size: var(--text-xs);
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--text-muted);
}
.grid dd {
  margin: var(--space-1) 0 var(--space-2);
  font-size: var(--text-xl);
  font-weight: 600;
}
.grid p {
  font-size: var(--text-xs);
  color: var(--text-secondary);
}
.grid p.footnote {
  margin-top: var(--space-2);
  padding-top: var(--space-2);
  border-top: 1px dashed var(--border);
  color: var(--text-muted);
}

.warning {
  padding: var(--space-4);
  border: 1px solid var(--warn);
  border-radius: var(--radius);
  margin-bottom: var(--space-8);
}
.warning h2 { font-size: var(--text-base); color: var(--warn); }
.warning p { margin-top: var(--space-2); font-size: var(--text-sm); color: var(--text-secondary); max-width: 64ch; }

.prose h2 {
  font-size: var(--text-base);
  margin: var(--space-6) 0 var(--space-3);
}
.prose ul {
  margin: 0;
  padding-left: var(--space-4);
  display: flex;
  flex-direction: column;
  gap: var(--space-2);
  color: var(--text-secondary);
  font-size: var(--text-sm);
  max-width: 72ch;
}
.prose strong { color: var(--text-primary); }
</style>
