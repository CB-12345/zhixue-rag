interface BrandMarkProps {
  className?: string
}

export default function BrandMark({ className = 'h-10 w-10' }: BrandMarkProps) {
  return (
    <svg viewBox="0 0 48 48" fill="none" aria-hidden="true" className={`shrink-0 ${className}`}>
      <rect x="1" y="1" width="46" height="46" rx="14" fill="var(--color-accent-bg)" stroke="var(--color-border)" />
      <path d="M24 18c-3-3-7-4-13-3v18c6-1 10 0 13 3 3-3 7-4 13-3V18" fill="var(--color-card)" />
      <path d="M24 18c-3-3-7-4-13-3v18c6-1 10 0 13 3m0-18v18m0-18c2-2 4-3 6-3m-6 21c3-3 7-4 13-3V22" stroke="var(--color-accent)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
      <path d="m36 9 1.5 4.5L42 15l-4.5 1.5L36 21l-1.5-4.5L30 15l4.5-1.5L36 9Z" fill="var(--color-accent)" />
    </svg>
  )
}
