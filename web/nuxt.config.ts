export default defineNuxtConfig({
  compatibilityDate: '2026-08-01',
  devtools: { enabled: false },
  ssr: true,

  css: ['~/assets/css/tokens.css', '~/assets/css/base.css'],

  runtimeConfig: {
    public: {
      // The gateway is the only public surface; nothing here talks to
      // market-data or intelligence directly.
      apiBase: process.env.NUXT_PUBLIC_API_BASE || 'http://localhost:8010',
    },
  },

  app: {
    head: {
      title: 'Intraday Trading AI — NSE',
      meta: [
        { charset: 'utf-8' },
        { name: 'viewport', content: 'width=device-width, initial-scale=1' },
        {
          name: 'description',
          content:
            'Intraday signal research for NSE equities. Publishes its own p-value.',
        },
      ],
    },
  },

  typescript: { strict: true },
})
