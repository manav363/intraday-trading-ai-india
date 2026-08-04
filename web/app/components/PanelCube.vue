<script setup lang="ts">
/**
 * The cross-sectional panel, in 3D.
 *
 * **Why this one view earns 3D.** On a price series 3D is decoration — the data
 * has two axes and a third would be invented. The panel is different: it is
 * genuinely `(time × symbol × feature-rank)`, three real axes, and the thing
 * worth seeing is how the cross-section *reorders itself* through the session.
 * A heatmap flattens that; a rotatable surface does not.
 *
 * Height and colour both encode rank, so the surface still reads when hue is
 * removed — the same redundancy rule the 2D charts follow.
 *
 * three.js is imported dynamically so it never enters the initial bundle. It is
 * ~600kb and only this route needs it.
 */
import type { PanelCube } from '~/composables/useApi'

const props = defineProps<{ cube: PanelCube }>()

const host = ref<HTMLDivElement | null>(null)
let cleanup: (() => void) | null = null

const reducedMotion = ref(false)

onMounted(async () => {
  reducedMotion.value = window.matchMedia('(prefers-reduced-motion: reduce)').matches
  await render()
})

onBeforeUnmount(() => cleanup?.())

watch(() => props.cube, () => void render())

function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim()
}

async function render() {
  cleanup?.()
  if (!host.value) return

  const THREE = await import('three')
  const el = host.value
  const width = el.clientWidth
  const height = el.clientHeight

  const scene = new THREE.Scene()
  const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000)
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
  renderer.setSize(width, height)
  el.appendChild(renderer.domElement)

  const rows = props.cube.timestamps.length
  const cols = props.cube.symbols.length
  const spacing = 1
  const depth = rows * spacing
  const span = cols * spacing

  const up = new THREE.Color(cssVar('--dir-up') || '#12a012')
  const down = new THREE.Color(cssVar('--dir-down') || '#e66767')
  const flat = new THREE.Color(cssVar('--dir-flat') || '#8e8d84')

  const group = new THREE.Group()

  // One instanced mesh for the whole grid: rows x cols can reach a few
  // thousand cells, and that many individual meshes would tank the frame rate.
  const geometry = new THREE.BoxGeometry(spacing * 0.82, 1, spacing * 0.82)
  // NOT `vertexColors: true`. That flag makes the shader look for a `color`
  // attribute on the *geometry*, which BoxGeometry does not have, so every
  // instance renders black. Per-instance colour goes through InstancedMesh's
  // separate `instanceColor` path, which three wires up on its own.
  const material = new THREE.MeshLambertMaterial()
  const mesh = new THREE.InstancedMesh(geometry, material, rows * cols)

  const dummy = new THREE.Object3D()
  const colour = new THREE.Color()
  let i = 0

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const value = props.cube.values[r]?.[c]

      // A missing cell is drawn as a flat sliver, not as rank 0. Rank 0 is the
      // bottom of the scale and would render as a real, extreme reading.
      const known = value !== null && value !== undefined
      const rank = known ? value : 0.5
      const barHeight = known ? Math.max(0.06, rank * 4) : 0.04

      dummy.position.set(
        c * spacing - span / 2,
        barHeight / 2,
        r * spacing - depth / 2,
      )
      dummy.scale.set(1, barHeight, 1)
      dummy.updateMatrix()
      mesh.setMatrixAt(i, dummy.matrix)

      if (!known) {
        colour.copy(flat).multiplyScalar(0.4)
      } else {
        // Diverging around the 0.5 midpoint: the cross-sectional median.
        colour.copy(flat).lerp(rank >= 0.5 ? up : down, Math.abs(rank - 0.5) * 2)
      }
      mesh.setColorAt(i, colour)
      i++
    }
  }
  mesh.instanceMatrix.needsUpdate = true
  if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true
  group.add(mesh)

  const grid = new THREE.GridHelper(Math.max(span, depth), Math.max(cols, rows))
  grid.material.opacity = 0.12
  grid.material.transparent = true
  group.add(grid)

  scene.add(group)
  scene.add(new THREE.AmbientLight(0xffffff, 0.75))
  const key = new THREE.DirectionalLight(0xffffff, 0.85)
  key.position.set(6, 12, 8)
  scene.add(key)

  camera.position.set(span * 0.9, Math.max(span, depth) * 0.8, depth * 1.05)
  camera.lookAt(0, 0, 0)

  // Drag to rotate. Deliberately hand-rolled rather than pulling in
  // OrbitControls — this needs two axes and a clamp, which is 15 lines.
  let dragging = false
  let lastX = 0
  let lastY = 0
  let yaw = 0
  let pitch = 0

  const onDown = (e: PointerEvent) => {
    dragging = true
    lastX = e.clientX
    lastY = e.clientY
    el.setPointerCapture(e.pointerId)
  }
  const onMove = (e: PointerEvent) => {
    if (!dragging) return
    yaw += (e.clientX - lastX) * 0.005
    pitch = Math.max(-0.4, Math.min(0.9, pitch + (e.clientY - lastY) * 0.004))
    lastX = e.clientX
    lastY = e.clientY
  }
  const onUp = () => {
    dragging = false
  }

  el.addEventListener('pointerdown', onDown)
  el.addEventListener('pointermove', onMove)
  el.addEventListener('pointerup', onUp)
  el.addEventListener('pointerleave', onUp)

  let frame = 0
  const animate = () => {
    frame = requestAnimationFrame(animate)
    // Idle drift, unless the viewer asked for reduced motion.
    if (!dragging && !reducedMotion.value) yaw += 0.0015
    group.rotation.y = yaw
    group.rotation.x = pitch
    renderer.render(scene, camera)
  }
  animate()

  const onResize = () => {
    if (!host.value) return
    const w = host.value.clientWidth
    const h = host.value.clientHeight
    camera.aspect = w / h
    camera.updateProjectionMatrix()
    renderer.setSize(w, h)
  }
  window.addEventListener('resize', onResize)

  cleanup = () => {
    cancelAnimationFrame(frame)
    window.removeEventListener('resize', onResize)
    el.removeEventListener('pointerdown', onDown)
    el.removeEventListener('pointermove', onMove)
    el.removeEventListener('pointerup', onUp)
    el.removeEventListener('pointerleave', onUp)
    geometry.dispose()
    material.dispose()
    renderer.dispose()
    el.innerHTML = ''
  }
}
</script>

<template>
  <div class="cube">
    <div ref="host" class="cube__canvas" />

    <!-- The 3D view is not the only way to read this. Screen-reader users and
         anyone who would rather have numbers get the same data as a table. -->
    <details class="cube__table">
      <summary>View as a table</summary>
      <div class="scroll-x">
        <table>
          <caption class="visually-hidden">
            Cross-sectional {{ cube.feature }} rank by symbol and time
          </caption>
          <thead>
            <tr>
              <th scope="col">time</th>
              <th v-for="s in cube.symbols" :key="s" scope="col">{{ s }}</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(row, r) in cube.values.slice(-12)" :key="r">
              <th scope="row" class="mono">
                {{ new Date(cube.timestamps.slice(-12)[r]!).toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit', hour12: false }) }}
              </th>
              <td v-for="(v, c) in row" :key="c" class="mono">
                {{ v === null ? '—' : v.toFixed(2) }}
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </details>
  </div>
</template>

<style scoped>
.cube__canvas {
  width: 100%;
  height: clamp(320px, 52vh, 560px);
  background: var(--surface-1);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  cursor: grab;
  touch-action: none;
}
.cube__canvas:active {
  cursor: grabbing;
}

.cube__table {
  margin-top: var(--space-4);
  font-size: var(--text-xs);
}
.cube__table summary {
  cursor: pointer;
  color: var(--text-secondary);
  padding: var(--space-2) 0;
}

table {
  border-collapse: collapse;
  width: 100%;
  min-width: 520px;
}
th,
td {
  padding: 3px var(--space-2);
  text-align: right;
  border-bottom: 1px solid var(--border);
  white-space: nowrap;
}
thead th {
  color: var(--text-muted);
  font-weight: 500;
  text-align: right;
}
tbody th {
  text-align: left;
  color: var(--text-secondary);
  font-weight: 400;
}
</style>
