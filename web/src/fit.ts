// The arena's pictures fill the window and no more: the header, the view's controls, the captions, the answer bar, a failure, the key legend and the
// Ratings line are all measured, and the pictures get the height that is left, so the whole screen is in view with no scroll bar (judging is done hundreds
// of times a day; scrolling to reach the answer, or to see the bottom of a picture, costs every time). Compare's own `maxHeight="viewport"` guesses the
// rest of the page at 14rem; here it is measured instead. The frames keep their shape, so a wide window gets pictures as tall as the space and a narrow
// one gets them as wide as the column. Only below a tiny window (a frame under MIN_FRAME pixels tall) does the page scroll. The opened Ratings table is
// left out: opening it is asking to scroll down to it. In full screen the space is the full-screen frame's.
import { useCallback, useLayoutEffect, useRef, useState, type RefObject } from 'react'

/** Below this many pixels a frame stops shrinking and the page scrolls instead: a picture smaller than this cannot be judged. */
export const MIN_FRAME = 120

/** One picture frame as laid out: where its top is and how tall it is (frames side by side share a top). */
export type FrameBox = { top: number; height: number }

/** The tallest each frame may be so that `content` (everything, frames included, as laid out now) fits in `space`; frames stacked in rows
 *  (side by side on a phone) share what is left. Null with no frames. Every height is in CSS pixels. */
export function frameFit(space: number, content: number, frames: readonly FrameBox[], min = MIN_FRAME): number | null {
  if (!frames.length) return null
  const rows = new Map<number, number>()
  for (const f of frames) { const k = Math.round(f.top); rows.set(k, Math.max(rows.get(k) ?? 0, f.height)) }
  let stacked = 0
  for (const h of rows.values()) stacked += h
  const fixed = content - stacked
  return Math.max(min, Math.floor((space - fixed) / rows.size))
}

const box = (e: Element): FrameBox => { const r = e.getBoundingClientRect(); return { top: r.top, height: r.height } }
const pad = (e: Element) => { const s = getComputedStyle(e); return (parseFloat(s.paddingTop) || 0) + (parseFloat(s.paddingBottom) || 0) }

/** Measures the page and returns the `maxHeight` for the Compare named `label` (its section's aria-label), as a CSS length; "viewport" (Compare's
 *  own guess) until the first measurement. `page` is the screen's outermost element; `extra` holds a Disclosure whose opened body is not counted. */
export function useFitFrames(label: string, page: RefObject<HTMLElement | null>, extra?: RefObject<HTMLElement | null>): string {
  const [fit, setFit] = useState<number | null>(null)
  const observer = useRef<ResizeObserver | null>(null), watched = useRef(new WeakSet<Element>())

  const measure = useCallback(() => {
    const main = page.current
    // The section may be in a full-screen modal (a portal), so it is found by its name, not under `page`.
    const section = document.querySelector(`section[aria-label="${CSS.escape(label)}"]`)
    if (!main || !section) return
    const ro = observer.current
    // A re-mounted section (full screen in a modal) is picked up here; each element is observed once, as an observer may call back as it starts.
    for (const e of [main, section]) if (ro && !watched.current.has(e)) { watched.current.add(e); ro.observe(e) }
    const frames = [...section.querySelectorAll('[role="group"][style*="aspect-ratio"]')].map(box)
    const space = document.documentElement.clientHeight
    const full = section.closest('[role="dialog"]') || document.fullscreenElement?.contains(section)
    let content: number
    // Full screen stretches the section to the window, so its content is measured to the bottom of its last child.
    if (full) content = Math.max(0, ...[...section.children].map(c => c.getBoundingClientRect().bottom)) - section.getBoundingClientRect().top + (section.parentElement ? pad(section.parentElement) : 0)
    else {
      const r = main.getBoundingClientRect()
      content = r.bottom + window.scrollY // from the top of the document: anything above the screen counts too
      const button = extra?.current?.querySelector('button[aria-expanded="true"][aria-controls]')
      const body = button && document.getElementById(button.getAttribute('aria-controls')!)
      if (body) content -= body.getBoundingClientRect().height
    }
    const next = frameFit(space, content, frames)
    if (next != null) setFit(f => (f != null && Math.abs(f - next) < 1 ? f : next))
  }, [label, page, extra])

  // Measured after every render of the screen (a new pair, a mode, a failure line), and again whenever the window, the screen or the comparison
  // changes size (full screen, the window, a caption wrapping): a changed fit resizes the frames, which measures again and settles at once.
  useLayoutEffect(measure)
  useLayoutEffect(() => {
    const ro = new ResizeObserver(() => measure())
    observer.current = ro; watched.current = new WeakSet()
    window.addEventListener('resize', measure)
    document.addEventListener('fullscreenchange', measure)
    measure()
    return () => { ro.disconnect(); observer.current = null; window.removeEventListener('resize', measure); document.removeEventListener('fullscreenchange', measure) }
  }, [measure])

  return fit == null ? 'viewport' : `${fit}px`
}
