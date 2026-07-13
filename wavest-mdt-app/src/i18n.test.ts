import { describe, expect, it } from 'vitest'
import { useCopy } from './i18n'

describe('bilingual clinical copy', () => {
  it('provides matching review-path lengths', () => {
    expect(useCopy('zh').guideSteps).toHaveLength(5)
    expect(useCopy('en').guideSteps).toHaveLength(5)
  })

  it('keeps the product identity stable across languages', () => {
    expect(useCopy('zh').product).toBe('WaveST-MDT')
    expect(useCopy('en').product).toBe('WaveST-MDT')
  })
})
