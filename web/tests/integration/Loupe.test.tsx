import { describe, expect, it, vi } from 'vitest'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import App from '../../src/App'

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
    expect(crops()[0].getAttribute('src')).toContain('x=0.5000&y=0.2500&size=100&zoom=3'); expect(crops()[1].getAttribute('src')).toContain('variant=raw&transform=level')
    fireEvent.click(screen.getByTestId('pic-level')); expect(screen.getByText(/Held: click the picture again to release/)).toBeInTheDocument()
    hover(screen.getByTestId('pic-level'), 20, 50); await new Promise(r => setTimeout(r, 300))
    expect(crops()[0].getAttribute('src')).toContain('x=0.5000&y=0.2500')                                 // held: the point did not move
    fireEvent.click(screen.getByTestId('pic-level')); hover(screen.getByTestId('pic-level'), 20, 50)
    await waitFor(() => expect(crops()[0].getAttribute('src')).toContain('x=0.1000&y=0.5000'))           // released: it follows again
  })
  it('follows the pointer over the original too, and changes area and zoom', async () => {
    open('#/p/demo/i/abcd1234~A_1'); await screen.findByText(/Turned 1\.37/)
    hover(screen.getByTestId('pic-original'), 40, 20); await waitFor(() => expect(crops().some(i => i.getAttribute('src')?.includes('x=0.2000&y=0.2000'))).toBe(true))
    await userEvent.selectOptions(screen.getByLabelText('Loupe zoom'), '5'); await userEvent.selectOptions(screen.getByLabelText('Loupe area'), '160')
    expect(crops().filter(i => i.getAttribute('src')?.includes('size=160&zoom=5'))).toHaveLength(4)
  })
  it('has no loupe for a picture without variants', async () => {
    open('#/p/demo/i/abcd1234~A_2'); await screen.findByText(/Camera JPEG/)
    hover(screen.getByTestId('pic-level'), 100, 25); await waitFor(() => expect(crops()).toHaveLength(2))   // one variant x 2 transforms
  })
})
