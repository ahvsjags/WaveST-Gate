import { expect, test } from '@playwright/test'

test.beforeEach(async ({ page }) => {
  await page.goto('/')
  await page.waitForLoadState('networkidle')
  await expect(page.getByTestId('tissue-canvas')).toBeVisible()
})

test('renders a nonblank spatial scene and bilingual shell', async ({ page }, testInfo) => {
  const canvas = page.getByTestId('tissue-canvas').locator('canvas')
  await expect(canvas).toBeVisible()
  await expect.poll(() => canvas.evaluate((node) => node.width), { timeout: 15_000 }).toBeGreaterThan(300)
  await expect.poll(() => canvas.evaluate((node) => node.height), { timeout: 15_000 }).toBeGreaterThan(240)
  await page.waitForTimeout(750)
  const dimensions = await canvas.evaluate((node) => ({ width: node.width, height: node.height }))
  expect(dimensions.width).toBeGreaterThan(300)
  expect(dimensions.height).toBeGreaterThan(240)

  const pixels = await canvas.evaluate((node) => {
    const context = node.getContext('webgl2') || node.getContext('webgl')
    if (!context) return 0
    const gl = context as WebGLRenderingContext
    const sample = new Uint8Array(4 * 16 * 16)
    gl.readPixels(Math.max(0, node.width / 2 - 8), Math.max(0, node.height / 2 - 8), 16, 16, gl.RGBA, gl.UNSIGNED_BYTE, sample)
    return Array.from(sample).filter((value) => value > 0).length
  })
  expect(pixels).toBeGreaterThan(20)

  await page.getByTestId('language-toggle').click()
  await expect(page.getByText('Spatial pathology workstation for multidisciplinary oncology')).toBeVisible()
  await page.screenshot({ path: testInfo.outputPath('workstation-desktop.png'), fullPage: true })
})

test('switches evidence layers and completes an MDT opinion', async ({ page }) => {
  test.skip(test.info().project.name === 'mobile-chromium', 'Desktop clinical workflow test')
  await page.getByTestId('layer-uncertainty').click()
  await expect(page.getByTestId('layer-uncertainty')).toHaveClass(/active/)
  await page.getByTestId('nav-mdt').click()
  const textarea = page.getByPlaceholder('记录该区域的专业判断…')
  await textarea.fill('建议在高不确定性边界区域进行病理复核。')
  await page.getByRole('button', { name: '提交意见' }).click()
  await expect(page.getByText('建议在高不确定性边界区域进行病理复核。')).toBeVisible()
})

test('signs the structured report and opens the local import flow', async ({ page }) => {
  test.skip(test.info().project.name === 'mobile-chromium', 'Desktop clinical workflow test')
  await page.getByTestId('nav-report').click()
  await page.getByRole('button', { name: '完成签阅' }).click()
  await expect(page.getByText('签阅完成')).toBeVisible()

  await page.getByTestId('nav-cases').click()
  await page.getByRole('button', { name: '导入病例' }).click()
  await expect(page.getByRole('dialog')).toBeVisible()
  await expect(page.getByText('选择去标识化病例文件')).toBeVisible()
})

test('guides high-risk review, switches view, exports evidence and validates an import queue', async ({ page }) => {
  test.skip(test.info().project.name === 'mobile-chromium', 'Desktop clinical workflow test')
  await page.getByRole('button', { name: /复核高风险区域/ }).click()
  await expect(page.locator('.spot-id')).toBeVisible()
  await expect(page.getByTestId('layer-uncertainty')).toHaveClass(/active/)
  await page.getByRole('button', { name: '病理俯视' }).click()

  await page.getByTestId('nav-report').click()
  const downloadPromise = page.waitForEvent('download')
  await page.getByRole('button', { name: '导出证据JSON' }).click()
  const download = await downloadPromise
  expect(download.suggestedFilename()).toBe('DEMO-ST-001-evidence.json')

  await page.getByTestId('nav-cases').click()
  await page.getByRole('button', { name: '导入病例' }).click()
  await page.locator('input[type="file"]').setInputFiles({ name: 'deidentified_spots.csv', mimeType: 'text/csv', buffer: Buffer.from('spot_id,x,y') })
  await expect(page.getByText('deidentified_spots.csv')).toBeVisible()
  await expect(page.getByRole('button', { name: '进入质量检查' })).toBeEnabled()
})

test('has no incoherent text overlap on mobile', async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== 'mobile-chromium', 'Mobile layout test')
  await page.screenshot({ path: testInfo.outputPath('workstation-mobile.png'), fullPage: true })
  const horizontalOverflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth)
  expect(horizontalOverflow).toBeLessThanOrEqual(1)
  await expect(page.getByTestId('nav-atlas')).toBeVisible()
  await expect(page.getByTestId('evidence-inspector')).toBeVisible()
})
