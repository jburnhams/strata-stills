import { describe, expect, it, vi } from 'vitest'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import App from '../../src/App'
import { mock } from '../utils/server'

const open = (hash: string) => { window.location.hash = hash; return render(<App />) }
const rect = { left: 0, top: 0, width: 200, height: 100, right: 200, bottom: 100, x: 0, y: 0, toJSON: () => ({}) }
const hover = (el: Element, clientX: number, clientY: number) => { vi.spyOn(el, 'getBoundingClientRect').mockReturnValue(rect as DOMRect); fireEvent.mouseMove(el, { clientX, clientY }) }
const crops = () => screen.queryAllByRole('img').filter(i => i.getAttribute('src')?.includes('/api/project/crop')) as HTMLImageElement[]

describe('loupe', () => {
  it('shows the hovered point cut from every variant with every transform, and holds it when clicked', async () => {
    open('#/p/demo/i/abcd1234~A_1'); await screen.findByText(/Turned 1\.37/)
    expect(screen.getByText(/Move the pointer over the picture/)).toBeInTheDocument(); expect(crops()).toHaveLength(0)
    hover(screen.getByTestId('pic-level'), 100, 25)
    await waitFor(() => expect(crops()).toHaveLength(4))                                                  // 2 variants x 2 transforms
    expect(crops().map(i => i.getAttribute('alt'))).toEqual(['Raw Original', 'Raw Levelled', 'Camera JPEG Original', 'Camera JPEG Levelled'])
    expect(crops()[0].getAttribute('src')).toContain('x=0.5000&y=0.2500&size=160&zoom=2'); expect(crops()[1].getAttribute('src')).toContain('variant=raw&transform=level')
    fireEvent.click(screen.getByTestId('pic-level')); expect(screen.getByText(/Held: click the picture again to release/)).toBeInTheDocument()
    hover(screen.getByTestId('pic-level'), 20, 50); await new Promise(r => setTimeout(r, 300))
    expect(crops()[0].getAttribute('src')).toContain('x=0.5000&y=0.2500')                                 // held: the point did not move
    fireEvent.click(screen.getByTestId('pic-level')); hover(screen.getByTestId('pic-level'), 20, 50)
    await waitFor(() => expect(crops()[0].getAttribute('src')).toContain('x=0.1000&y=0.5000'))           // released: it follows again
  })
  it('follows the pointer over the original too, and changes area and zoom', async () => {
    open('#/p/demo/i/abcd1234~A_1'); await screen.findByText(/Turned 1\.37/)
    hover(screen.getByTestId('pic-original'), 40, 20); await waitFor(() => expect(crops().some(i => i.getAttribute('src')?.includes('x=0.2000&y=0.2000'))).toBe(true))
    await userEvent.selectOptions(screen.getByLabelText('Loupe zoom'), '4'); await userEvent.selectOptions(screen.getByLabelText('Loupe area'), '240')
    expect(crops().filter(i => i.getAttribute('src')?.includes('size=240&zoom=4'))).toHaveLength(4)
  })
  it('has one variant of cells for a picture with a single variant', async () => {
    open('#/p/demo/i/abcd1234~A_2'); await screen.findByText(/Camera JPEG/)
    hover(screen.getByTestId('pic-level'), 100, 25); await waitFor(() => expect(crops()).toHaveLength(2))   // one variant x 2 transforms
  })
})

describe('loupe waits for the full-resolution copy', () => {
  it('shows what is being prepared instead of a low-resolution picture, then the crop', async () => {
    mock.fullres = { state: 'running', text: 'developing the raw at full size (about a minute and a half, once)' }
    open('#/p/demo/i/abcd1234~A_1'); await screen.findByText(/Turned 1\.37/)
    hover(screen.getByTestId('pic-level'), 100, 25)
    expect((await screen.findAllByText(/Preparing the full-resolution view of Raw: developing the raw at full size/)).length).toBeGreaterThan(0); expect(crops()).toHaveLength(0)
    mock.fullres = { state: 'ready', text: '' }; await waitFor(() => expect(crops()).toHaveLength(4), { timeout: 4000 })
  })
  it('says when the full-size copy cannot be made', async () => {
    mock.fullres = { state: 'failed', text: 'cannot read' }; open('#/p/demo/i/abcd1234~A_1'); await screen.findByText(/Turned 1\.37/); hover(screen.getByTestId('pic-level'), 100, 25)
    expect((await screen.findAllByText(/Cannot make the full-size copy: cannot read/)).length).toBeGreaterThan(0)
  })
})
