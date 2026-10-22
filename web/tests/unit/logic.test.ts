import { describe, expect, it } from 'vitest'
import { parse, href } from '../../src/route'
import { imageUrl } from '../../src/api'
import { describe as say } from '../../src/components/ItemPage'

describe('routes', () => {
  it('parses every page', () => {
    expect(parse('')).toEqual({ page: 'projects' })
    expect(parse('#/p/2025%2Fdrone%2Ftwo')).toEqual({ page: 'project', project: '2025/drone/two' })
    expect(parse('#/p/demo/s/abcd1234')).toEqual({ page: 'source', project: 'demo', source: 'abcd1234' })
    expect(parse('#/p/demo/i/abcd1234~A_1')).toEqual({ page: 'item', project: 'demo', item: 'abcd1234~A_1', source: 'abcd1234' })
  })
  it('round-trips ids that contain slashes', () => {
    const h = href.item('2025/drone/two', 'abcd1234~A_1'); expect(parse(h)).toMatchObject({ page: 'item', project: '2025/drone/two', item: 'abcd1234~A_1' })
  })
})

describe('imageUrl', () => {
  it('encodes the ids and leaves out an unset variant', () => {
    expect(imageUrl('a/b', 'x~y', null, 'level', 360)).toBe('/api/project/image?id=a%2Fb&item=x~y&transform=level&px=360')
    expect(imageUrl('p', 'i', 'enhanced.jpg')).toContain('&variant=enhanced.jpg')
  })
})

describe('describe a transform', () => {
  it('says what level did, or why it did nothing', () => {
    expect(say('level', null)).toBe('Working…')
    expect(say('level', { applied: true, angle: -1.37, confidence: 0.49, nadir: true })).toMatch(/Turned 1\.37° clockwise/)
    expect(say('level', { applied: true, angle: 2, confidence: 0.9 })).toMatch(/anticlockwise/)
    expect(say('level', { applied: false, reason: 'already level', lines: 30, nadir: false })).toMatch(/Left as it is: already level\..*oblique/)
    expect(say('original', { applied: false })).toMatch(/as it comes/)
  })
})

import { detailText } from '../../src/components/GeneratePanel'
describe('detail text', () => {
  it('turns the measurement into pixels across', () => {
    expect(detailText(null)).toMatch(/Measuring/)
    expect(detailText({ ok: true, native: [8064, 6048], scale50: 0.35, scale30: 0.75 })).toMatch(/about 2822 px across and 30 % at 6048 px/)
    expect(detailText({ ok: false, reason: 'too little texture to measure' })).toMatch(/Could not measure.*too little texture/)
  })
})

import { pointIn } from '../../src/components/Loupe'
import { cropUrl } from '../../src/api'
describe('loupe helpers', () => {
  it('maps a mouse position to 0..1 of the element and refuses an element without size', () => {
    const el = { getBoundingClientRect: () => ({ left: 10, top: 20, width: 200, height: 100 }) } as unknown as Element
    expect(pointIn({ clientX: 110, clientY: 70 }, el)).toEqual({ x: 0.5, y: 0.5 })
    expect(pointIn({ clientX: -50, clientY: 500 }, el)).toEqual({ x: 0, y: 1 })
    expect(pointIn({ clientX: 1, clientY: 1 }, { getBoundingClientRect: () => ({ left: 0, top: 0, width: 0, height: 0 }) } as unknown as Element)).toBeNull()
  })
  it('builds the crop address', () => {
    expect(cropUrl('a/b', 'x~y', 'enhanced.jpg', 'level', 0.123456, 0.5, 100, 3)).toBe('/api/project/crop?id=a%2Fb&item=x~y&variant=enhanced.jpg&transform=level&x=0.1235&y=0.5000&size=100&zoom=3')
  })
})
