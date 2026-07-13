import { mkdir, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { chromium } from '@playwright/test'

const appRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const workspaceRoot = path.resolve(appRoot, '..')
const smokeRoot = path.join(appRoot, 'runtime', 'smoke-input')
const outputRoot = path.join(appRoot, 'artifacts', 'screenshots')
await mkdir(outputRoot, { recursive: true })
const mark = async (stage) => writeFile(path.join(outputRoot, 'capture-progress.txt'), `${new Date().toISOString()} ${stage}\n`)

const browser = await chromium.launch({ headless: true })
const page = await browser.newPage({ viewport: { width: 1600, height: 1000 }, deviceScaleFactor: 2 })
const consoleErrors = []
page.on('console', (message) => {
  if (message.type() === 'error') consoleErrors.push(message.text())
})
page.on('pageerror', (error) => consoleErrors.push(error.message))

await page.goto('http://127.0.0.1:4173/', { waitUntil: 'networkidle' })
await mark('loaded-overview')
await page.getByTestId('tissue-canvas').waitFor({ state: 'visible' })
const canvas = page.getByTestId('tissue-canvas').locator('canvas')
await page.waitForFunction(() => {
  const node = document.querySelector('[data-testid="tissue-canvas"] canvas')
  return Boolean(node && node.width > 300 && node.height > 240)
}, undefined, { timeout: 20_000 })
await page.waitForTimeout(800)
const scene = await canvas.evaluate((node) => {
  const gl = node.getContext('webgl2') || node.getContext('webgl')
  if (!gl) return { width: node.width, height: node.height, nonblankPixels: 0 }
  const sample = new Uint8Array(4 * 16 * 16)
  gl.readPixels(Math.max(0, node.width / 2 - 8), Math.max(0, node.height / 2 - 8), 16, 16, gl.RGBA, gl.UNSIGNED_BYTE, sample)
  return { width: node.width, height: node.height, nonblankPixels: Array.from(sample).filter((value) => value > 0).length }
})
if (scene.nonblankPixels < 20) throw new Error(`Rendered tissue canvas was blank: ${JSON.stringify(scene)}`)
await page.screenshot({ path: path.join(outputRoot, 'wavest-mdt-workstation-overview-desktop.png'), fullPage: true })
await mark('saved-overview')

await page.getByTestId('nav-cases').click()
await page.locator('.primary-button.full').click()
await page.getByRole('dialog').waitFor({ state: 'visible' })
await page.screenshot({ path: path.join(outputRoot, 'wavest-mdt-local-inference-dialog.png'), fullPage: true })
await mark('saved-dialog')

const inputs = page.locator('.file-contract-grid input')
await inputs.nth(0).setInputFiles(path.join(smokeRoot, 'he_image.png'))
await mark('selected-he-image')
await inputs.nth(1).setInputFiles(path.join(smokeRoot, 'expression.csv'))
await mark('selected-expression')
await inputs.nth(2).setInputFiles(path.join(smokeRoot, 'coordinates.csv'))
await mark('selected-coordinates')
await inputs.nth(3).setInputFiles(path.join(smokeRoot, 'reference.csv'))
await mark('selected-reference')
await page.getByRole('button', { name: /提交本地推理|Run local inference/ }).click()
await mark('submitted-local-job')
await page.getByRole('dialog').waitFor({ state: 'hidden' })
await mark('dialog-hidden')
await page.waitForFunction(() => {
  const label = document.querySelector('.scene-legend')?.textContent ?? ''
  return label.includes('Local inference result') || label.includes('本地推理结果')
}, undefined, { timeout: 90_000 })
await mark('live-result-loaded')
await page.screenshot({ path: path.join(outputRoot, 'wavest-mdt-live-local-inference-desktop.png'), fullPage: true })

const mobile = await browser.newPage({ viewport: { width: 412, height: 915 }, deviceScaleFactor: 2 })
await mobile.goto('http://127.0.0.1:4173/', { waitUntil: 'networkidle' })
await mobile.getByTestId('tissue-canvas').waitFor({ state: 'visible' })
await mobile.waitForTimeout(700)
const overflow = await mobile.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth)
if (overflow > 1) throw new Error(`Mobile horizontal overflow: ${overflow}`)
await mobile.screenshot({ path: path.join(outputRoot, 'wavest-mdt-workstation-mobile.png'), fullPage: true })

await browser.close()
console.log(JSON.stringify({ scene, mobileHorizontalOverflow: overflow, consoleErrors, outputRoot }, null, 2))
if (consoleErrors.length) process.exitCode = 1
