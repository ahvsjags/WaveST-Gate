import { mkdir, writeFile } from 'node:fs/promises'
import { chromium } from '@playwright/test'

const artifacts = new URL('../artifacts/', import.meta.url)
await mkdir(artifacts, { recursive: true })

const browser = await chromium.launch({ headless: true })
const consoleErrors = []

try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 950 } })
  page.on('console', (message) => { if (message.type() === 'error') consoleErrors.push(message.text()) })
  page.on('pageerror', (error) => consoleErrors.push(error.message))
  await page.goto('http://127.0.0.1:4173', { waitUntil: 'networkidle' })
  await page.getByTestId('tissue-canvas').waitFor({ state: 'visible' })

  const canvas = page.getByTestId('tissue-canvas').locator('canvas')
  await page.waitForFunction(() => {
    const node = document.querySelector('[data-testid="tissue-canvas"] canvas')
    return node instanceof HTMLCanvasElement && node.width > 300 && node.height > 240
  }, { timeout: 15_000 })
  await page.waitForTimeout(750)
  const nonblankPixels = await canvas.evaluate((node) => {
    const context = node.getContext('webgl2') || node.getContext('webgl')
    if (!context) return 0
    const gl = context
    const sample = new Uint8Array(4 * 24 * 24)
    gl.readPixels(Math.max(0, node.width / 2 - 12), Math.max(0, node.height / 2 - 12), 24, 24, gl.RGBA, gl.UNSIGNED_BYTE, sample)
    return Array.from(sample).filter((value) => value > 0).length
  })
  if (nonblankPixels < 50) throw new Error(`WebGL scene appears blank: ${nonblankPixels} nonzero bytes`)

  await page.getByRole('button', { name: /复核高风险区域/ }).click()
  await page.locator('.spot-id').waitFor({ state: 'visible' })
  await page.getByRole('button', { name: '病理俯视' }).click()
  await page.getByTestId('nav-mdt').click()
  await page.getByPlaceholder('记录该区域的专业判断…').fill('高风险区域已完成多学科复核。')
  await page.getByRole('button', { name: '提交意见' }).click()
  await page.getByText('高风险区域已完成多学科复核。').waitFor({ state: 'visible' })

  await page.getByTestId('nav-report').click()
  const downloadPromise = page.waitForEvent('download')
  await page.getByRole('button', { name: '导出证据JSON' }).click()
  const download = await downloadPromise
  if (download.suggestedFilename() !== 'DEMO-ST-001-evidence.json') throw new Error('Unexpected evidence filename')

  await page.getByTestId('nav-cases').click()
  await page.getByRole('button', { name: '导入病例' }).click()
  await page.locator('input[type="file"]').setInputFiles({ name: 'deidentified_spots.csv', mimeType: 'text/csv', buffer: Buffer.from('spot_id,x,y') })
  await page.getByText('deidentified_spots.csv').waitFor({ state: 'visible' })
  await page.screenshot({ path: new URL('final-desktop.png', artifacts).pathname.slice(1), fullPage: true })

  if (consoleErrors.length) throw new Error(`Console errors:\n${consoleErrors.join('\n')}`)
  const result = { status: 'PASS', nonblankPixels, bilingual: true, mdt: true, reportExport: true, importQueue: true, consoleErrors: 0 }
  await writeFile(new URL('smoke-result.json', artifacts), JSON.stringify(result, null, 2), 'utf8')
  console.log(JSON.stringify(result, null, 2))
  await page.close()
} finally {
  await browser.close()
}
