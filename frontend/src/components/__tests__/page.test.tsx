import { render, screen } from '@testing-library/react'
import Home from '@/app/page'

describe('Home Page', () => {
  it('renders the main heading', () => {
    render(<Home />)
    
    const heading = screen.getByRole('heading', {
      name: /ai facultative reinsurance system/i,
    })
    
    expect(heading).toBeInTheDocument()
  })

  it('renders the welcome message', () => {
    render(<Home />)
    
    const welcomeText = screen.getByText(/welcome to the ai-powered decision support system/i)
    
    expect(welcomeText).toBeInTheDocument()
  })

  it('renders the description', () => {
    render(<Home />)
    
    const description = screen.getByText(/streamlining facultative reinsurance/i)
    
    expect(description).toBeInTheDocument()
  })
})