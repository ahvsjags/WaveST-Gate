import { copyFile, mkdir, readFile, writeFile } from 'node:fs/promises'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const appRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const repoRoot = resolve(appRoot, '..')

const source = {
  coords: join(repoRoot, 'data/processed/cytassist_xenium_rep2_common297/spot_coords.csv'),
  proportions: join(repoRoot, 'results/nature_main/cytassist_rep2_radius55/predicted_proportions.csv'),
  uncertainty: join(repoRoot, 'results/nature_main/cytassist_rep2_radius55/spot_uncertainty.csv'),
  gates: join(repoRoot, 'results/nature_main/cytassist_rep2_radius55/gate_weights.csv'),
  niches: join(repoRoot, 'results/nature_main/cytassist_rep2_radius55/nature_analysis/niche_assignments.csv'),
  tissue: join(repoRoot, 'data/raw/wavestgate_breast_core/10x/visium/extracted/spatial/tissue_lowres_image.png'),
  boundary: join(repoRoot, 'results/nature_main/cytassist_rep2_radius55/nature_analysis/boundary_he_overlay.png'),
  nicheMap: join(repoRoot, 'results/nature_main/cytassist_rep2_radius55/nature_analysis/niche_map.png'),
  uncertaintyMap: join(repoRoot, 'results/nature_main/cytassist_rep2_radius55/nature_analysis/spot_uncertainty_map.png'),
}

function parseCsv(text) {
  const lines = text.trim().split(/\r?\n/)
  const headers = lines[0].split(',')
  return {
    headers,
    rows: lines.slice(1).map((line) => {
      const values = line.split(',')
      return Object.fromEntries(headers.map((header, index) => [header, values[index] ?? '']))
    }),
  }
}

const [coordText, proportionText, uncertaintyText, gateText, nicheText] = await Promise.all([
  readFile(source.coords, 'utf8'),
  readFile(source.proportions, 'utf8'),
  readFile(source.uncertainty, 'utf8'),
  readFile(source.gates, 'utf8'),
  readFile(source.niches, 'utf8'),
])

const coords = parseCsv(coordText)
const proportions = parseCsv(proportionText)
const uncertainty = parseCsv(uncertaintyText)
const gates = parseCsv(gateText)
const niches = parseCsv(nicheText)

const byId = (rows) => new Map(rows.map((row) => [row.spot_id, row]))
const propById = byId(proportions.rows)
const uncertaintyById = byId(uncertainty.rows)
const gateById = byId(gates.rows)
const nicheById = byId(niches.rows)
const cellTypes = proportions.headers.slice(1)

const xs = coords.rows.map((row) => Number(row.x))
const ys = coords.rows.map((row) => Number(row.y))
const bounds = {
  minX: Math.min(...xs), maxX: Math.max(...xs), minY: Math.min(...ys), maxY: Math.max(...ys),
}

const means = new Array(cellTypes.length).fill(0)
const spots = coords.rows.map((coord) => {
  const prop = propById.get(coord.spot_id)
  const unc = uncertaintyById.get(coord.spot_id)
  const gate = gateById.get(coord.spot_id)
  const niche = nicheById.get(coord.spot_id)
  const values = cellTypes.map((cellType, index) => {
    const value = Number(prop?.[cellType] ?? 0)
    means[index] += value
    return Number(value.toFixed(5))
  })
  let dominant = 0
  for (let index = 1; index < values.length; index += 1) {
    if (values[index] > values[dominant]) dominant = index
  }
  return {
    id: coord.spot_id,
    x: Number((((Number(coord.x) - bounds.minX) / (bounds.maxX - bounds.minX)) * 2 - 1).toFixed(5)),
    y: Number((((Number(coord.y) - bounds.minY) / (bounds.maxY - bounds.minY)) * 2 - 1).toFixed(5)),
    values,
    dominant,
    uncertainty: Number(Number(unc?.spot_uncertainty ?? 0).toFixed(5)),
    gates: [Number(gate?.image ?? 0), Number(gate?.expression ?? 0), Number(gate?.reference ?? 0)].map((v) => Number(v.toFixed(5))),
    niche: Number(niche?.niche ?? 0),
    nicheLabel: niche?.niche_label ?? 'unassigned',
  }
})

const output = {
  case: {
    id: 'DEMO-ST-001',
    title: 'Multi-modal tumour spatial atlas',
    titleZh: '多模态肿瘤空间图谱',
    specimen: 'FFPE tumour tissue',
    specimenZh: 'FFPE肿瘤组织',
    validationContext: 'Breast carcinoma benchmark',
    validationContextZh: '乳腺癌首个验证数据集',
    platform: 'Visium/CytAssist + Xenium',
    model: 'WaveST-Gate 0.1.0',
  },
  cellTypes,
  aggregate: {
    spots: spots.length,
    cellStates: cellTypes.length,
    supervisedSpots: 485,
    typedCells: 115275,
    meanProportions: means.map((value) => Number((value / spots.length).toFixed(5))),
    uncertaintyCorrelation: 0.5263,
    boundaryJumpRatio: 2.362,
    calibrationBinCorrelation: 0.9619,
  },
  spots,
}

const publicData = join(appRoot, 'public/data')
const publicAssets = join(appRoot, 'public/assets')
await Promise.all([mkdir(publicData, { recursive: true }), mkdir(publicAssets, { recursive: true })])
await writeFile(join(publicData, 'spatial-demo.json'), JSON.stringify(output), 'utf8')
await Promise.all([
  copyFile(source.tissue, join(publicAssets, 'tissue-lowres.png')),
  copyFile(source.boundary, join(publicAssets, 'boundary-evidence.png')),
  copyFile(source.nicheMap, join(publicAssets, 'niche-map.png')),
  copyFile(source.uncertaintyMap, join(publicAssets, 'uncertainty-map.png')),
])

console.log(`Prepared ${spots.length} de-identified spots and ${cellTypes.length} cell states.`)
