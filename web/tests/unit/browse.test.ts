import { describe, expect, it } from 'vitest'
import type { BrowsePicture } from '../../src/api'
import { thumbUrl } from '../../src/api'
import { sortRows } from 'strata-react'
import { ancestorKeys, groupOf, shownFrom, takenAt, thumbPx, thumbSources, typeOf } from '../../src/browse'
import { href } from '../../src/route'
import { folderName } from '../../src/components/BrowserPage'

const pic = (stem: string, taken: string | null, files = [stem + '.ARW', stem + '.JPG'], projects: string[] = []): BrowsePicture => ({
  stem, kind: 'still', files, taken, bytes: 1, projects,
  sources: [{ id: 'lightroom', label: 'Lightroom export' }, { id: 'embedded', label: 'Camera preview' }, { id: 'jpeg', label: 'Camera JPEG' }, { id: 'raw', label: 'Raw' }],
})
const rows = [pic('DSC10', '2019-03-02T10:30:00Z'), pic('DSC9', '2019-03-02T11:05:00Z', ['DSC9.ARW']), pic('DSC100', null, ['DSC100.JPG'], ['trip'])]

describe('the Browser view', () => {
  it('sorts names naturally and times with the unknown last', () => {
    expect(sortRows(rows, p => p.stem).map(p => p.stem)).toEqual(['DSC9', 'DSC10', 'DSC100'])
    expect(sortRows(rows, takenAt).map(p => p.stem)).toEqual(['DSC10', 'DSC9', 'DSC100'])
    expect(sortRows(rows, takenAt, true).map(p => p.stem)).toEqual(['DSC9', 'DSC10', 'DSC100'])
  })

  it('names the group of each picture by type, project, hour and day', () => {
    expect(rows.map(typeOf)).toEqual(['Raw + JPEG', 'Raw', 'JPEG'])
    expect(rows.map(p => groupOf(p, 'type'))).toEqual(['Raw + JPEG', 'Raw', 'JPEG'])
    expect(rows.map(p => groupOf(p, 'project'))).toEqual(['In no project', 'In no project', 'In trip'])
    expect(new Set(rows.map(p => groupOf(p, 'hour'))).size).toBe(3)
    expect(groupOf(rows[2], 'hour')).toBe('No time')
    expect(groupOf(rows[0], 'day')).toBe(groupOf(rows[1], 'day'))
  })

  it('asks the server for a thumbnail twice the tile, in the sizes it keeps', () => {
    expect([64, 80, 81, 160, 161, 400].map(thumbPx)).toEqual([160, 160, 320, 320, 640, 640])
  })

  it('tries the best develop first, or the camera picture first', () => {
    expect(thumbSources(rows[0], 'best')).toEqual(['lightroom', 'embedded', 'jpeg', 'raw'])
    expect(thumbSources(rows[0], 'camera')).toEqual(['embedded', 'jpeg', 'lightroom', 'raw'])
    expect(shownFrom(rows[0], 'best')).toBe('Lightroom export')
    expect(shownFrom({ ...rows[0], sources: [] }, 'best')).toBe('No picture')
  })

  it('builds folder keys, links and thumbnail addresses', () => {
    expect(ancestorKeys('raw/2019/2019-03-02')).toEqual(['raw', 'raw/2019'])
    expect(folderName('raw/2026/2026-04-06')).toBe('2026-04-06')
    expect(folderName('drone/20250402 - Hazelmere')).toBe('20250402 - Hazelmere')
    expect(href.browse('raw/2019')).toBe('/browse?folder=raw%2F2019')
    expect(thumbUrl('drone/20250402 - Hazelmere', 'DJI_1', 'gen:p:abc', 320)).toBe('/api/browse/thumb?folder=drone%2F20250402%20-%20Hazelmere&stem=DJI_1&source=gen%3Ap%3Aabc&px=320')
  })
})
