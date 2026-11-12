// jsdom fetches no pictures, so an image never fires load or error. installImages makes every picture "load" on the next tick; a test that needs a picture
// still on its way calls images.hold(), and images.release() lets the held ones load. The arena waits for both pictures of a pair before showing it.
const held: HTMLImageElement[] = []
let holding = false
const fire = (i: HTMLImageElement) => i.dispatchEvent(new Event('load'))

export const images = {
  hold() { holding = true },
  release() { holding = false; for (const i of held.splice(0)) fire(i) },
  held: () => held.map(i => i.getAttribute('src')),
}

export function installImages() {
  Object.defineProperty(HTMLImageElement.prototype, 'src', {
    configurable: true,
    get(this: HTMLImageElement) { return this.getAttribute('src') ?? '' },
    set(this: HTMLImageElement, v: string) { this.setAttribute('src', v); if (holding) held.push(this); else setTimeout(() => fire(this)) },
  })
}
