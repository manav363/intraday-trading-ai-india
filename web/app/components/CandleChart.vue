<script setup lang="ts">
/**
 * Candlestick + volume, hand-built in SVG.
 *
 * No charting library. The mark specs matter here — 2px surface gap between
 * adjacent bodies, thin wicks, recessive grid, tabular numerics, a crosshair
 * with a tooltip — and controlling them directly is less code than bending a
 * library into shape.
 *
 * 2D on purpose. A price series is two-dimensional data; rendering it in 3D
 * would be decoration. The 3D view in this app is reserved for the panel cube,
 * where the data genuinely has three axes.
 */
import type { Bar } from '~/composables/useApi'

const props = withDefaults(
  defineProps<{
    bars: Bar[]
    height?: number
    /** Session VWAP, drawn as a reference line when supplied. */
    showVolume?: boolean
  }>(),
  { height: 320, showVolume: true },
)

const WIDTH = 900
const PAD = { top: 12, right: 56, bottom: 22, left: 8 }

const priceHeight = computed(() =>
  props.showVolume ? props.height * 0.74 : props.height,
)
const volumeTop = computed(() => priceHeight.value + 8)
const volumeHeight = computed(() => props.height - volumeTop.value - PAD.bottom)

const plotWidth = computed(() => WIDTH - PAD.left - PAD.right)

const bounds = computed(() => {
  if (!props.bars.length) return { lo: 0, hi: 1, vmax: 1 }
  const lo = Math.min(...props.bars.map((b) => b.l))
  const hi = Math.max(...props.bars.map((b) => b.h))
  const pad = (hi - lo) * 0.06 || 1
  return {
    lo: lo - pad,
    hi: hi + pad,
    vmax: Math.max(...props.bars.map((b) => b.v)) || 1,
  }
})

const step = computed(() => plotWidth.value / Math.max(props.bars.length, 1))
/** 2px surface gap between adjacent bodies, per the mark spec. */
const bodyWidth = computed(() => Math.max(1, step.value - 2))

function x(i: number): number {
  return PAD.left + i * step.value + step.value / 2
}

function y(price: number): number {
  const { lo, hi } = bounds.value
  return PAD.top + (1 - (price - lo) / (hi - lo)) * (priceHeight.value - PAD.top)
}

function volY(v: number): number {
  return volumeHeight.value * (1 - v / bounds.value.vmax)
}

const gridLines = computed(() => {
  const { lo, hi } = bounds.value
  return Array.from({ length: 5 }, (_, i) => {
    const price = lo + ((hi - lo) * i) / 4
    return { price, y: y(price) }
  })
})

const hovered = ref<number | null>(null)

function onMove(event: MouseEvent) {
  const svg = event.currentTarget as SVGSVGElement
  const rect = svg.getBoundingClientRect()
  const local = ((event.clientX - rect.left) / rect.width) * WIDTH
  const index = Math.floor((local - PAD.left) / step.value)
  hovered.value = index >= 0 && index < props.bars.length ? index : null
}

const hoveredBar = computed(() =>
  hovered.value === null ? null : props.bars[hovered.value],
)

function timeLabel(iso: string): string {
  const d = new Date(iso)
  return d.toLocaleString('en-IN', {
    day: '2-digit',
    month: 'short',
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
  })
}

function priceLabel(v: number): string {
  return v.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })
}

function volumeLabel(v: number): string {
  if (v >= 1e7) return `${(v / 1e7).toFixed(2)}Cr`
  if (v >= 1e5) return `${(v / 1e5).toFixed(2)}L`
  if (v >= 1e3) return `${(v / 1e3).toFixed(1)}k`
  return v.toFixed(0)
}
</script>

<template>
  <figure class="chart">
    <div class="scroll-x">
      <svg
        :viewBox="`0 0 ${WIDTH} ${height}`"
        :style="{ height: `${height}px` }"
        class="chart__svg"
        role="img"
        :aria-label="`Candlestick chart, ${bars.length} bars`"
        @mousemove="onMove"
        @mouseleave="hovered = null"
      >
        <!-- Recessive grid. -->
        <g class="grid">
          <line
            v-for="line in gridLines"
            :key="line.price"
            :x1="PAD.left"
            :x2="WIDTH - PAD.right"
            :y1="line.y"
            :y2="line.y"
          />
        </g>

        <g class="axis">
          <text
            v-for="line in gridLines"
            :key="`l-${line.price}`"
            :x="WIDTH - PAD.right + 6"
            :y="line.y + 3"
          >
            {{ priceLabel(line.price) }}
          </text>
        </g>

        <!-- Candles. Direction is colour + body fill + position of close
             relative to open; the tooltip adds the words. -->
        <g>
          <template v-for="(bar, i) in bars" :key="bar.t">
            <line
              class="wick"
              :class="bar.c >= bar.o ? 'wick--up' : 'wick--down'"
              :x1="x(i)"
              :x2="x(i)"
              :y1="y(bar.h)"
              :y2="y(bar.l)"
            />
            <rect
              class="body"
              :class="bar.c >= bar.o ? 'body--up' : 'body--down'"
              :x="x(i) - bodyWidth / 2"
              :y="y(Math.max(bar.o, bar.c))"
              :width="bodyWidth"
              :height="Math.max(1, Math.abs(y(bar.o) - y(bar.c)))"
            />
          </template>
        </g>

        <!-- Volume. -->
        <g v-if="showVolume" :transform="`translate(0, ${volumeTop})`">
          <rect
            v-for="(bar, i) in bars"
            :key="`v-${bar.t}`"
            class="vol"
            :class="bar.c >= bar.o ? 'vol--up' : 'vol--down'"
            :x="x(i) - bodyWidth / 2"
            :y="volY(bar.v)"
            :width="bodyWidth"
            :height="volumeHeight - volY(bar.v)"
          />
        </g>

        <!-- Crosshair. -->
        <g v-if="hovered !== null" class="crosshair">
          <line :x1="x(hovered)" :x2="x(hovered)" :y1="PAD.top" :y2="height - PAD.bottom" />
        </g>
      </svg>
    </div>

    <figcaption class="chart__caption">
      <template v-if="hoveredBar">
        <span class="mono">{{ timeLabel(hoveredBar.t) }}</span>
        <span class="mono">O {{ priceLabel(hoveredBar.o) }}</span>
        <span class="mono">H {{ priceLabel(hoveredBar.h) }}</span>
        <span class="mono">L {{ priceLabel(hoveredBar.l) }}</span>
        <span class="mono">C {{ priceLabel(hoveredBar.c) }}</span>
        <span class="mono">V {{ volumeLabel(hoveredBar.v) }}</span>
        <span :class="hoveredBar.c >= hoveredBar.o ? 'up' : 'down'">
          {{ hoveredBar.c >= hoveredBar.o ? '▲ up' : '▼ down' }}
        </span>
      </template>
      <template v-else>
        <span>{{ bars.length }} bars · hover for detail</span>
      </template>
    </figcaption>
  </figure>
</template>

<style scoped>
.chart {
  margin: 0;
}

.chart__svg {
  width: 100%;
  min-width: 640px;
  display: block;
  cursor: crosshair;
}

.grid line {
  stroke: var(--border);
  stroke-width: 1;
  opacity: 0.55;
}

.axis text {
  fill: var(--text-muted);
  font-size: 10px;
  font-family: var(--font-mono);
}

.wick {
  stroke-width: 1;
}
.wick--up {
  stroke: var(--dir-up);
}
.wick--down {
  stroke: var(--dir-down);
}

.body--up {
  fill: var(--dir-up);
}
.body--down {
  fill: var(--dir-down);
}

.vol--up {
  fill: var(--dir-up);
  opacity: 0.32;
}
.vol--down {
  fill: var(--dir-down);
  opacity: 0.32;
}

.crosshair line {
  stroke: var(--text-muted);
  stroke-width: 1;
  stroke-dasharray: 3 3;
}

.chart__caption {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-3);
  padding: var(--space-2) 0 0;
  font-size: var(--text-xs);
  color: var(--text-secondary);
  border-top: 1px solid var(--border);
  margin-top: var(--space-2);
}

.up {
  color: var(--dir-up);
}
.down {
  color: var(--dir-down);
}
</style>
