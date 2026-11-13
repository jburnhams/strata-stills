import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { render, screen } from '@testing-library/react'
import { useRef } from 'react'
import { useFitFrames } from '../../src/fit'

// jsdom does no layout, so each element here says where it is (data-top, data-h) and getBoundingClientRect reads that; the window is 800 tall.
let restore: () => void
beforeEach(() => {
  const rect = Element.prototype.getBoundingClientRect
  Element.prototype.getBoundingClientRect = function (this: Element) {
    const top = Number(this.getAttribute('data-top') ?? 0), h = Number(this.getAttribute('data-h') ?? 0)
    return { x: 0, y: top, left: 0, right: 100, top, bottom: top + h, width: 100, height: h, toJSON: () => ({}) } as DOMRect
  }
  const ch = Object.getOwnPropertyDescriptor(HTMLElement.prototype, 'clientHeight')
  Object.defineProperty(HTMLElement.prototype, 'clientHeight', { configurable: true, get: () => 800 })
  restore = () => { Element.prototype.getBoundingClientRect = rect; if (ch) Object.defineProperty(HTMLElement.prototype, 'clientHeight', ch) }
})
afterEach(() => restore())

function Page({ full = false, open = false }: { full?: boolean; open?: boolean }) {
  const page = useRef<HTMLElement>(null), extra = useRef<HTMLDivElement>(null)
  const fit = useFitFrames('Pair', page, extra)
  const section = (
    <section aria-label="Pair" data-top="0" data-h="800">
      <div data-top="0" data-h="100">toolbar</div>
      <div role="group" aria-label="frame" style={{ aspectRatio: '1.5' }} data-top="100" data-h="600" />
    </section>
  )
  return (
    <main ref={page} data-top="0" data-h={open ? 1400 : 1000}>
      <output>{fit}</output>
      {full ? <div role="dialog" style={{ padding: '16px' }}>{section}</div> : section}
      <div ref={extra}>
        <button aria-expanded={open} aria-controls={open ? 'body' : undefined}>Ratings</button>
        {open && <div id="body" data-top="1000" data-h="400" />}
      </div>
    </main>
  )
}

describe('useFitFrames: the arena measures the page and gives the pictures the rest of the window', () => {
  it('gives the frame the window less everything else on the page', () => {
    render(<Page />) // 1000 of page with a 600 frame: 400 of other content, 400 left in an 800 window
    expect(screen.getByRole('status')).toHaveTextContent('400px')
  })

  it('leaves an opened Ratings table out: opening it is asking to scroll', () => {
    render(<Page open />)
    expect(screen.getByRole('status')).toHaveTextContent('400px')
  })

  it('in full screen, measures the comparison to its last child, not the stretched section', () => {
    render(<Page full />) // toolbar 100 + frame 600 = 700 of content plus 32 of padding; frame gets 800 - 132 = 668
    expect(screen.getByRole('status')).toHaveTextContent('668px')
  })
})
