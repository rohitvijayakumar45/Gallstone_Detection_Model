/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        canvas:  '#F7F5F0',
        surface: '#FFFFFF',
        border:  '#E4E1D9',
        ink: {
          900: '#141210',
          700: '#3D3A35',
          500: '#706D66',
          300: '#B0ADA6',
          100: '#E8E5DE',
        },
        positive: { bg: '#EEF6F1', text: '#1B6535', border: '#B6D9C3' },
        negative: { bg: '#FDF1F0', text: '#B91C1C', border: '#F2C4C0' },
        warn:     { bg: '#FEF9EC', text: '#92400E', border: '#F0DFA0' },
        critical: { bg: '#FFF1F0', text: '#991B1B', border: '#FECACA' },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      fontSize: {
        '2xs': ['0.65rem', { lineHeight: '1rem' }],
      },
      boxShadow: {
        card: '0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04)',
        lift: '0 4px 16px rgba(0,0,0,0.08), 0 1px 4px rgba(0,0,0,0.04)',
      },
    },
  },
  plugins: [],
}
