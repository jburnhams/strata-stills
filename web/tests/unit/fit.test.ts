import { describe, expect, it } from 'vitest'
import { MIN_FRAME, frameFit } from '../../src/fit'

describe('frameFit: the tallest a picture frame may be so the whole arena fits the window', () => {
  it('gives the frames what the rest of the page leaves', () => {
    // 900 tall window; the page is 1200 tall with two 600 frames side by side (one row): 600 of other content, 300 left for the frames.
    expect(frameFit(900, 1200, [{ top: 150, height: 600 }, { top: 150, height: 600 }])).toBe(300)
    // Content shorter than the window grows the frames.
    expect(frameFit(900, 500, [{ top: 150, height: 200 }])).toBe(600)
  })

  it('shares the space between frames stacked in rows (side by side on a phone)', () => {
    expect(frameFit(800, 1000, [{ top: 100, height: 300 }, { top: 430.4, height: 300 }])).toBe(200)
  })

  it('counts a row once, by its tallest frame', () => {
    expect(frameFit(900, 1000, [{ top: 100, height: 400 }, { top: 100.3, height: 380 }])).toBe(300)
  })

  it('stops at a floor in a tiny window, where the page scrolls instead', () => {
    expect(frameFit(200, 900, [{ top: 100, height: 400 }])).toBe(MIN_FRAME)
    expect(frameFit(200, 900, [{ top: 100, height: 400 }], 50)).toBe(50)
  })

  it('has nothing to say with no frames (still loading)', () => {
    expect(frameFit(900, 400, [])).toBeNull()
  })
})
