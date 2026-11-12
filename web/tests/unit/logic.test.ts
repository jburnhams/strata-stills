import { describe, expect, it } from 'vitest'
import { href, sourceOf } from '../../src/route'
import { alertImageUrl, arenaImageUrl, flaggedImageUrl, imageUrl } from '../../src/api'
import { describe as say } from '../../src/components/ItemPage'
import { arenaQuery } from '../../src/components/ArenaPage'
import { jobSummary } from '../../src/components/AutoGenerate'
import { holds } from '../../src/components/ProjectPage'
import { variantWords } from '../../src/components/SourcePage'
import { items } from '../utils/server'

describe('routes', () => {
  it('encodes ids into one path segment each (project ids contain slashes)', () => {
    expect(href.project('2025/drone/two')).toBe('/p/2025%2Fdrone%2Ftwo')
    expect(href.source('demo', 'abcd1234')).toBe('/p/demo/s/abcd1234')
    expect(href.item('demo', 'abcd1234~A_1')).toBe('/p/demo/i/abcd1234~A_1')
    expect([href.projects(), href.arena(), href.alerts(), href.flagged()]).toEqual(['/', '/arena', '/alerts', '/flagged'])
  })
  it('finds the source folder of an item', () => expect(sourceOf('abcd1234~A_1')).toBe('abcd1234'))
})

describe('arena and flags addresses', () => {
  it('passes on only the arena\'s own query', () => {
    expect(arenaQuery(new URLSearchParams('set=hard&want=auto1,auto2&x=1'))).toBe('?set=hard&want=auto1%2Cauto2')
    expect(arenaQuery(new URLSearchParams(''))).toBe('')
  })
  it('builds picture addresses', () => {
    expect(arenaImageUrl('img/tok/left')).toBe('/arena/img/tok/left')
    expect(alertImageUrl('a 1')).toBe('/api/alerts/image/a%201')
    expect(alertImageUrl('a1', 'lightroom')).toBe('/api/alerts/image/a1?variant=lightroom')
    expect(flaggedImageUrl('C:\\x y.jpg')).toBe('/api/flagged/image?path=C%3A%5Cx%20y.jpg&px=600')
  })
})

describe('words', () => {
  it('says how an auto-generate run stands', () => {
    expect(jobSummary({ state: 'running', done: 1, total: 3, text: 'A_1' })).toBe('1 of 3: A_1')
    expect(jobSummary({ state: 'running' })).toBe('0 of …')
    expect(jobSummary({ state: 'stopped', done: 1, total: 3 })).toBe('Stopped after 1 of 3.')
    expect(jobSummary({ state: 'stopped' })).toBe('Stopped after 0 of 0.')
    expect(jobSummary({ state: 'done', done: 3, total: 3, failed: [] })).toBe('Made 3 of 3.')
    expect(jobSummary({ state: 'done', done: 2, total: 3, failed: [['x~A', 'broken']] })).toBe('Made 2 of 3; 1 failed.')
    expect(jobSummary({ state: 'done' })).toBe('Made 0 of 0.')
  })
  it('says what a folder holds, leaving out what it has none of', () => {
    expect(holds({ id: 'a', path: 'p', reachable: true, stills: 2, panos: 1, videos: 1 })).toBe('2 stills, 1 panoramas, 1 videos')
    expect(holds({ id: 'a', path: 'p', reachable: true, stills: 0, panos: 0, videos: 0 })).toBe('0 stills')
  })
  it('names the variants, primary first', () => {
    expect(variantWords(items[0])).toEqual(['dng', 'jpg'])
    expect(variantWords({ ...items[0], primary: 'jpeg' })).toEqual(['jpg', 'dng'])
    expect(variantWords({ ...items[0], variants: [{ id: 'g', label: 'G', path: '', kind: 'generated', ext: '.jpg' }] })).toEqual(['gen'])
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
