import { mkdir, writeFile } from 'node:fs/promises'
import { chromium } from '@playwright/test'

const url = process.env.WAVEST_PUBLIC_URL
if (!url) throw new Error('WAVEST_PUBLIC_URL is required')

const artifacts = new URL('../artifacts/', import.meta.url)
await mkdir(artifacts, { recursive: true })

const browser = await chromium.launch({ headless: true })
const failures = []
const browserErrors = []
let result

try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 950 } })
  page.on('requestfailed', (request) => failures.push(`${request.method()} ${request.url()} ${request.failure()?.errorText ?? ''}`))
  page.on('console', (message) => {
    if (message.type() === 'error') {
      browserErrors.push(message.text())
      console.error(`[browser] ${message.text()}`)
    }
  })
  page.on('pageerror', (error) => {
    browserErrors.push(error.message)
    console.error(`[pageerror] ${error.message}`)
  })

  for (let attempt = 1; attempt <= 5; attempt += 1) {
    try {
      await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 30_000 })
      await page.getByTestId('tissue-canvas').waitFor({ state: 'visible', timeout: 25_000 })
      await page.getByTestId('tissue-canvas').locator('canvas').waitFor({ state: 'visible', timeout: 10_000 })
      break
    } catch (error) {
      if (attempt === 5) throw error
      await page.waitForTimeout(2_000)
    }
  }

  const canvas = page.getByTestId('tissue-canvas').locator('canvas')
  await page.waitForFunction(() => {
    const node = document.querySelector('[data-testid="tissue-canvas"] canvas')
    return node instanceof HTMLCanvasElement && node.width > 300 && node.height > 240
  }, { timeout: 15_000 })
  await page.waitForTimeout(750)
  const dimensions = await canvas.evaluate((node) => ({ width: node.width, height: node.height }))
  const nonblankPixels = await canvas.evaluate((node) => {
    const gl = node.getContext('webgl2') || node.getContext('webgl')
    if (!gl) return 0
    const sample = new Uint8Array(4 * 20 * 20)
    gl.readPixels(Math.max(0, node.width / 2 - 10), Math.max(0, node.height / 2 - 10), 20, 20, gl.RGBA, gl.UNSIGNED_BYTE, sample)
    return Array.from(sample).filter((value) => value > 0).length
  })

  if (dimensions.width < 300 || dimensions.height < 240 || nonblankPixels < 50) {
    throw new Error(`Public WebGL validation failed: ${JSON.stringify({ dimensions, nonblankPixels })}`)
  }

  const title = await page.title()
  const spots = await page.locator('body').getByText('4,992', { exact: true }).count()
  await page.screenshot({ path: new URL('public-desktop.png', artifacts).pathname.slice(1), fullPage: true })
  await page.setViewportSize({ width: 393, height: 851 })
  await page.waitForTimeout(750)
  const mobileHorizontalOverflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth)
  if (mobileHorizontalOverflow > 1) throw new Error(`Mobile horizontal overflow: ${mobileHorizontalOverflow}px`)
  await page.getByTestId('evidence-inspector').waitFor({ state: 'visible', timeout: 10_000 })
  await page.screenshot({ path: new URL('public-mobile.png', artifacts).pathname.slice(1), fullPage: true })

  result = {
    status: 'PASS',
    url,
    title,
    dimensions,
    nonblankPixels,
    spotCountVisible: spots > 0,
    mobileHorizontalOverflow,
    transientRequestFailures: failures,
    browserErrors,
  }
  await writeFile(new URL('public-smoke-result.json', artifacts), JSON.stringify(result, null, 2), 'utf8')
  console.log(JSON.stringify(result, null, 2))
} finally {
  await browser.close()
}
