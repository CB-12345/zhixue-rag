/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      fontFamily: {
        heading: ['var(--font-heading)'],
        body: ['var(--font-body)'],
        mono: ['var(--font-mono)'],
      },
      colors: {
        bg: {
          DEFAULT: 'var(--color-bg)',
          secondary: 'var(--color-bg-secondary)',
          tertiary: 'var(--color-bg-tertiary)',
        },
        card: {
          DEFAULT: 'var(--color-card)',
          hover: 'var(--color-card-hover)',
        },
        text: {
          DEFAULT: 'var(--color-text)',
          secondary: 'var(--color-text-secondary)',
          tertiary: 'var(--color-text-tertiary)',
          placeholder: 'var(--color-text-placeholder)',
        },
        border: {
          DEFAULT: 'var(--color-border)',
          light: 'var(--color-border-light)',
        },
        divider: 'var(--color-divider)',
        accent: {
          DEFAULT: 'var(--color-accent)',
          bg: 'var(--color-accent-bg)',
        },
        success: {
          DEFAULT: 'var(--color-success)',
          bg: 'var(--color-success-bg)',
        },
        danger: {
          DEFAULT: 'var(--color-danger)',
          bg: 'var(--color-danger-bg)',
        },
        warning: {
          DEFAULT: 'var(--color-warning)',
          bg: 'var(--color-warning-bg)',
        },
      },
      spacing: {
        'xs': '4px',
        'sm': '8px',
        'md': '16px',
        'lg': '24px',
        'xl': '32px',
        '2xl': '48px',
        '3xl': '64px',
      },
      borderRadius: {
        sm: 'var(--radius-sm)',
        md: 'var(--radius-md)',
        lg: 'var(--radius-lg)',
      },
      lineHeight: {
        tight: '1.3',
        normal: '1.6',
        loose: '1.8',
      },
    },
  },
  plugins: [require('@tailwindcss/typography')],
}
