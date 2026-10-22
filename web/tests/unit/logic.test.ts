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
