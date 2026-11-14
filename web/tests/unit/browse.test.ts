import { describe, expect, it } from 'vitest'
import type { BrowsePicture } from '../../src/api'
import { thumbUrl } from '../../src/api'
import { ancestorKeys, groupPictures, shownFrom, sortPictures, thumbSources, typeOf } from '../../src/browse'
import { href } from '../../src/route'

const pic = (stem: string, taken: string | null, files = [stem + '.ARW', stem + '.JPG'], projects: string[] = []): BrowsePicture => ({
  stem, kind: 'still', files, taken, bytes: 1, projects,
  sources: [{ id: 'lightroom', label: 'Lightroom export' }, { id: 'embedded', label: 'Camera preview' }, { id: 'jpeg', label: 'Camera JPEG' }, { id: 'raw', label: 'Raw' }],
})
const rows = [pic('DSC10', '2019-03-02T10:30:00Z'), pic('DSC9', '2019-03-02T11:05:00Z', ['DSC9.ARW']), pic('DSC100', null, ['DSC100.JPG'], ['trip'])]

describe('the Browser view', () => {
  it('sorts names naturally and times with the unknown last', () => {
    expect(sortPictures(rows, 'name').map(p => p.stem)).toEqual(['DSC9', 'DSC10', 'DSC100'])
    expect(sortPictures(rows, 'name-desc').map(p => p.stem)).toEqual(['DSC100', 'DSC10', 'DSC9'])
    expect(sortPictures(rows, 'time').map(p => p.stem)).toEqual(['DSC10', 'DSC9', 'DSC100'])
    expect(sortPictures(rows, 'time-desc').map(p => p.stem)).toEqual(['DSC9', 'DSC10', 'DSC100'])
  })

  it('groups by type, project and hour, in the order the pictures come', () => {
    expect(rows.map(typeOf)).toEqual(['Raw + JPEG', 'Raw', 'JPEG'])
    expect(groupPictures(rows, 'none')).toHaveLength(1)
    expect(groupPictures(rows, 'project').map(g => [g.label, g.rows.length])).toEqual([['In no project', 2], ['In trip', 1]])
    const hours = groupPictures(rows, 'hour')
    expect(hours).toHaveLength(3)
    expect(hours[2].label).toBe('No time')
    expect(groupPictures(rows, 'day').map(g => g.rows.length)).toEqual([2, 1])
  })

  it('tries the best develop first, or the camera picture first', () => {
    expect(thumbSources(rows[0], 'best')).toEqual(['lightroom', 'embedded', 'jpeg', 'raw'])
    expect(thumbSources(rows[0], 'camera')).toEqual(['embedded', 'jpeg', 'lightroom', 'raw'])
    expect(shownFrom(rows[0], 'best')).toBe('Lightroom export')
    expect(shownFrom({ ...rows[0], sources: [] }, 'best')).toBe('No picture')
  })

  it('builds folder keys, links and thumbnail addresses', () => {
    expect(ancestorKeys('raw/2019/2019-03-02')).toEqual(['raw', 'raw/2019'])
    expect(href.browse('raw/2019')).toBe('/browse?folder=raw%2F2019')
    expect(thumbUrl('drone/20250402 - Hazelmere', 'DJI_1', 'gen:p:abc', 320)).toBe('/api/browse/thumb?folder=drone%2F20250402%20-%20Hazelmere&stem=DJI_1&source=gen%3Ap%3Aabc&px=320')
  })
})
