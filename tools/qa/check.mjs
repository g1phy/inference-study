#!/usr/bin/env node
// Headless-проверка одной главы (или всей страницы) в Chromium. Работает при запущенном dev-сервере (npm run dev, порт 5173).
//
//   node tools/qa/check.mjs attention                      # десктоп 1440, светлая тема
//   node tools/qa/check.mjs attention --theme dark
//   node tools/qa/check.mjs attention --mobile             # 375×812, тач
//   node tools/qa/check.mjs attention --actions /tmp/my-actions.json
//   node tools/qa/check.mjs kit                            # витрина ui-kit
//
// Что делает: открывает http://localhost:5173/#<id> (или --origin/--base), ждёт монтирования главы, делает скриншот каждой фигуры ([data-figure])
// и главы целиком (нарезкой по 1800px), печатает JSON-отчёт: ошибки и предупреждения консоли, горизонтальное переполнение,
// текст с контрастом ниже AA, фигуры без подписи «Попробуйте», длинные тире в тексте, высота главы, пути к PNG.
// Скриншоты потом смотрите через Read (это изображения).
//
// Снимки фигур и shot с sel делаются с экрана (captureBeyondViewport: false): элемент прокручивается к центру, окно при
// необходимости вытягивается по высоте (ширина та же, графики не перемериваются), пауза ~600 мс, потом снимок области.
// Раньше снимок элемента расширял поверхность за экран: ResizeObserver перемеривал графики, и motion-анимации (столбики
// BarChart, дуги attention) попадали в кадр сжатыми или недорисованными, хотя в DOM и на экране всё верно.
// Если анимированная фигура на снимке выглядит странно, сверьтесь с обычным shot без sel (снимок экрана).
//
// Опции: --theme light|dark, --width N, --height N, --mobile, --reduce (prefers-reduced-motion), --out DIR (по умолчанию /tmp/qa/<id>-<вариант>),
//        --settle MS (пауза после загрузки, по умолчанию 1500), --no-full (без нарезки главы), --no-figures, --port N,
//        --base PATH (путь сайта, по умолчанию /), --origin URL (например https://g1phy.github.io с --base /inference-study/),
//        --actions FILE: JSON-массив действий, выполняются после загрузки и перед скриншотами:
//          {"type":"click","sel":"css"} {"type":"tap","sel":"css"} {"type":"hover","sel":"css"} {"type":"type","sel":"css?","text":"..."}
//          {"type":"key","key":"ArrowRight","times":3} {"type":"wait","ms":500} {"type":"scroll","sel":"css"}
//          {"type":"shot","name":"after-click","sel":"css?"} {"type":"eval","js":"document.title"} {"type":"clickText","text":"Отправить","scope":"css?"}
//          {"type":"back","ms":1200} {"type":"forward"} (кнопки «Назад»/«Вперёд» браузера, потом пауза ms)
//          (clickText без scope ищет сначала внутри проверяемой главы, потом по всей странице)
import { createRequire } from 'node:module'
import { mkdirSync, readFileSync, existsSync } from 'node:fs'
import { join } from 'node:path'

const require = createRequire(import.meta.url)
const puppeteer = require('puppeteer-core')

const argv = process.argv.slice(2)
const target = argv.find((a) => !a.startsWith('--') && !isOptValue(a)) ?? 'hero'
function isOptValue(a) {
  const i = argv.indexOf(a)
  return i > 0 && ['--theme', '--width', '--height', '--out', '--settle', '--port', '--actions', '--base', '--origin'].includes(argv[i - 1])
}
const opt = (name, def) => {
  const i = argv.indexOf(`--${name}`)
  return i >= 0 && i + 1 < argv.length ? argv[i + 1] : def
}
const flag = (name) => argv.includes(`--${name}`)

const mobile = flag('mobile')
const theme = opt('theme', 'light')
const width = Number(opt('width', mobile ? 375 : 1440))
const height = Number(opt('height', mobile ? 812 : 900))
const port = Number(opt('port', 5173))
const base = opt('base', '/')
const origin = opt('origin', `http://localhost:${port}`)
const settle = Number(opt('settle', 1500))
const variant = `${mobile ? 'm' : width}-${theme}${flag('reduce') ? '-reduce' : ''}`
const out = opt('out', `/tmp/qa/${target}-${variant}`)
mkdirSync(out, { recursive: true })

const exeCandidates = [
  process.env.CHROME_PATH,
  `${process.env.HOME}/Library/Caches/ms-playwright/chromium_headless_shell-1228/chrome-headless-shell-mac-arm64/chrome-headless-shell`,
  `${process.env.HOME}/Library/Caches/ms-playwright/chromium-1228/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing`,
].filter(Boolean)
const executablePath = exeCandidates.find((p) => existsSync(p))
if (!executablePath) {
  console.error('Не найден Chromium. Задайте CHROME_PATH.')
  process.exit(2)
}

const url = target === '/' || target === 'top' ? `${origin}${base}` : `${origin}${base}#${target}`
const browser = await puppeteer.launch({ executablePath, headless: true, args: ['--no-sandbox', '--hide-scrollbars'] })
const report = { url, variant, out, console: [], pageErrors: [], shots: [] }
try {
  const page = await browser.newPage()
  await page.setViewport({ width, height, deviceScaleFactor: 1, isMobile: mobile, hasTouch: mobile })
  await page.emulateMediaFeatures([
    { name: 'prefers-color-scheme', value: theme },
    ...(flag('reduce') ? [{ name: 'prefers-reduced-motion', value: 'reduce' }] : []),
  ])
  // тема сайта читается из localStorage['inference-study:theme'] до отрисовки
  await page.evaluateOnNewDocument((t) => {
    try { localStorage.setItem('inference-study:theme', t) } catch {}
  }, theme)
  page.on('console', (m) => {
    const t = m.type()
    if (t === 'error' || t === 'warn' || t === 'warning') report.console.push(`${t}: ${m.text().slice(0, 400)}`)
  })
  page.on('pageerror', (e) => report.pageErrors.push(String(e.message).slice(0, 600)))

  await page.goto(url, { waitUntil: 'networkidle2', timeout: 60000 })
  const sel = target === 'kit' ? 'main, body' : target === '/' || target === 'top' ? 'body' : `section#${CSS_escape(target)}`
  if (sel.startsWith('section')) {
    await page.waitForSelector(`${sel}:not([aria-busy])`, { timeout: 30000 }).catch(() => {
      report.mountTimeout = true
    })
    await page.evaluate((s) => document.querySelector(s)?.scrollIntoView({ block: 'start', behavior: 'instant' }), sel)
  }
  await sleep(settle)

  // действия
  const actionsFile = opt('actions', null)
  if (actionsFile) {
    const actions = JSON.parse(readFileSync(actionsFile, 'utf8'))
    report.actions = []
    for (const a of actions) {
      try {
        if (a.type === 'click') await page.click(a.sel)
        else if (a.type === 'tap') await page.tap(a.sel)
        else if (a.type === 'hover') await page.hover(a.sel)
        else if (a.type === 'clickText') {
          const ok = await page.evaluate((text, scope, section) => {
            // без scope ищем сначала внутри проверяемой главы: одинаковые подписи кнопок есть в других главах
            const pick = (root) =>
              [...root.querySelectorAll('button, a, [role=button], [role=tab], [role=radio], label')].find((e) => e.textContent?.trim() === text)
            const el = scope
              ? pick(document.querySelector(scope) ?? document)
              : (section && document.querySelector(section) && pick(document.querySelector(section))) || pick(document)
            if (!el) return false
            el.scrollIntoView({ block: 'center', behavior: 'instant' })
            el.click()
            return true
          }, a.text, a.scope ?? null, sel.startsWith('section') ? sel : null)
          if (!ok) throw new Error(`нет элемента с текстом «${a.text}»`)
        } else if (a.type === 'type') {
          if (a.sel) {
            await page.click(a.sel, { clickCount: 3 })
            if (a.clear !== false) await page.keyboard.press('Backspace')
          }
          await page.keyboard.type(a.text)
        } else if (a.type === 'key') {
          for (let i = 0; i < (a.times ?? 1); i++) await page.keyboard.press(a.key)
        } else if (a.type === 'wait') await sleep(a.ms ?? 300)
        else if (a.type === 'scroll') {
          await page.evaluate((s, b) => document.querySelector(s)?.scrollIntoView({ block: b ?? 'start', behavior: 'instant' }), a.sel, a.block)
          await sleep(300)
        } else if (a.type === 'shot') {
          const path = join(out, `${a.name ?? 'action'}.png`)
          if (a.sel) {
            const el = await page.$(a.sel)
            if (el) await shootElement(page, el, path, { settle: a.settle ?? 600 })
            else throw new Error(`нет ${a.sel}`)
          } else await page.screenshot({ path, captureBeyondViewport: false })
          report.shots.push(path)
        } else if (a.type === 'eval') {
          const v = await page.evaluate(a.js)
          report.actions.push({ eval: a.label ?? a.js.slice(0, 60), value: v })
        } else if (a.type === 'back' || a.type === 'forward') {
          // «Назад»/«Вперёд» браузера: проверка возврата к месту чтения (popstate в App)
          await (a.type === 'back' ? page.goBack() : page.goForward())
          await sleep(a.ms ?? 1200)
        }
      } catch (e) {
        report.actions.push({ failed: JSON.stringify(a).slice(0, 160), error: String(e.message).slice(0, 200) })
      }
    }
    await sleep(300)
  }

  // аудит внутри главы
  const audit = await page.evaluate((s) => {
    const root = document.querySelector(s) ?? document.body
    const vw = document.documentElement.clientWidth
    const rr = root.getBoundingClientRect()
    const res = { chapterHeight: Math.round(rr.height), docScrollWidth: document.documentElement.scrollWidth, vw }
    // горизонтальное переполнение
    const overflow = []
    for (const el of root.querySelectorAll('*')) {
      const r = el.getBoundingClientRect()
      if (r.width === 0 || r.height === 0) continue
      if (r.right <= vw + 1 && r.left >= -1) continue
      let p = el.parentElement
      let clipped = false
      while (p && p !== document.body) {
        const cs = getComputedStyle(p)
        if (/(auto|scroll|hidden|clip)/.test(cs.overflowX)) {
          const pr = p.getBoundingClientRect()
          if (pr.right <= vw + 1 && pr.left >= -1) { clipped = true; break }
        }
        p = p.parentElement
      }
      if (!clipped && getComputedStyle(el).position !== 'fixed') overflow.push(`${el.tagName.toLowerCase()}.${String(el.className?.baseVal ?? el.className).slice(0, 70)} [${Math.round(r.left)}..${Math.round(r.right)}]`)
    }
    res.overflow = overflow.slice(0, 15)
    res.overflowCount = overflow.length
    // контраст текста (WCAG AA: 4.5 для < 18.66px bold / 24px, иначе 3)
    // rgb()/rgba() разбираем сразу; всё остальное (color-mix() даёт «color(srgb …)», oklab() и т.п.) рисуем
    // в пиксель канваса и читаем обратно: раньше такие фоны не распознавались и давали ложные 1,00:1
    const px = document.createElement('canvas')
    px.width = px.height = 1
    const pctx = px.getContext('2d', { willReadFrequently: true })
    const parse = (c) => {
      const m = c.match(/^rgba?\(([^)]+)\)$/)
      if (m) {
        const p = m[1].split(/[ ,/]+/).filter(Boolean).map(Number)
        return [p[0], p[1], p[2], p[3] ?? 1]
      }
      if (!c || c === 'none') return null
      pctx.clearRect(0, 0, 1, 1)
      pctx.fillStyle = 'rgba(0, 0, 0, 0)'
      pctx.fillStyle = c
      pctx.fillRect(0, 0, 1, 1)
      const d = pctx.getImageData(0, 0, 1, 1).data
      return [d[0], d[1], d[2], d[3] / 255]
    }
    const lum = ([r, g, b]) => {
      const f = (v) => { v /= 255; return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4 }
      return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)
    }
    const bgOf = (el) => {
      let layers = []
      let e = el
      while (e) {
        const c = parse(getComputedStyle(e).backgroundColor)
        if (c && c[3] > 0) { layers.push(c); if (c[3] >= 1) break }
        e = e.parentElement
      }
      let base = [255, 255, 255]
      for (let i = layers.length - 1; i >= 0; i--) {
        const [r, g, b, a] = layers[i]
        base = [r * a + base[0] * (1 - a), g * a + base[1] * (1 - a), b * a + base[2] * (1 - a)]
      }
      return base
    }
    const low = []
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT)
    const seen = new Set()
    while (walker.nextNode()) {
      const t = walker.currentNode
      const txt = t.textContent.trim()
      if (!txt) continue
      const el = t.parentElement
      if (!el || seen.has(el)) continue
      seen.add(el)
      const cs = getComputedStyle(el)
      if (cs.visibility === 'hidden' || cs.display === 'none' || Number(cs.opacity) < 0.05) continue
      const r = el.getBoundingClientRect()
      if (r.width === 0 || r.height === 0) continue
      if (el.closest('svg') && el.tagName.toLowerCase() !== 'text' && el.tagName.toLowerCase() !== 'tspan') continue
      if (el.closest('.katex')) continue
      const fg = parse(cs.color)
      if (!fg) continue
      const bg = bgOf(el)
      const a = fg[3]
      const fgc = [fg[0] * a + bg[0] * (1 - a), fg[1] * a + bg[1] * (1 - a), fg[2] * a + bg[2] * (1 - a)]
      const L1 = lum(fgc), L2 = lum(bg)
      const ratio = (Math.max(L1, L2) + 0.05) / (Math.min(L1, L2) + 0.05)
      const size = parseFloat(cs.fontSize)
      const bold = Number(cs.fontWeight) >= 700
      const need = size >= 24 || (bold && size >= 18.66) ? 3 : 4.5
      if (ratio < need) low.push(`${ratio.toFixed(2)} < ${need} «${txt.slice(0, 40)}» ${el.tagName.toLowerCase()} ${size}px`)
    }
    res.lowContrast = low.slice(0, 15)
    res.lowContrastCount = low.length
    // фигуры
    const figs = [...root.querySelectorAll('[data-figure]')]
    res.figures = figs.map((f, i) => {
      const r = f.getBoundingClientRect()
      const lb = f.getAttribute('aria-labelledby')
      const title = (lb ? document.getElementById(lb)?.textContent : f.getAttribute('aria-label'))?.trim().slice(0, 60) ?? ''
      const head = (f.textContent ?? '').trim().slice(0, 50)
      const capHasTry = /Попробуйте/.test(f.textContent ?? '')
      return { i, id: f.id || null, w: Math.round(r.width), h: Math.round(r.height), title: title || head, tryCaption: capHasTry }
    })
    // длинные тире в видимом тексте главы (DESIGN: тире минимально)
    const text = root.innerText ?? ''
    res.emDashCount = (text.match(/—/g) ?? []).length
    res.enDashCount = (text.match(/ – /g) ?? []).length
    res.straightQuotes = (text.match(/(^|[\s(])"[А-Яа-яЁё]/g) ?? []).length
    res.textLength = text.length
    return res
  }, sel)
  Object.assign(report, audit)

  // скриншоты фигур (липкую верхнюю панель прячем: на мобильном она закрывала верх высоких фигур)
  if (!flag('no-figures')) {
    await page.addStyleTag({ content: 'header.sticky{visibility:hidden!important}' })
    const figs = await page.$$(`${sel} [data-figure]`)
    for (let i = 0; i < figs.length; i++) {
      try {
        const path = join(out, `fig-${String(i + 1).padStart(2, '0')}.png`)
        await shootElement(page, figs[i], path, { settle: 600 })
        report.shots.push(path)
      } catch (e) {
        report.shots.push(`fig ${i + 1} failed: ${e.message}`)
      }
    }
    await page.setViewport({ width, height, deviceScaleFactor: 1, isMobile: mobile, hasTouch: mobile })
  }
  // глава целиком нарезкой. Окно растягиваем до высоты среза и снимаем то, что реально на экране:
  // со captureBeyondViewport графики на useElementSize успевали перемериться и выходили сжатыми или пустыми.
  // Липкую верхнюю панель на время нарезки прячем (иначе она закрывает верх каждого среза на мобильном).
  if (!flag('no-full') && sel.startsWith('section')) {
    const slice = 1800
    await page.setViewport({ width, height: slice, deviceScaleFactor: 1, isMobile: mobile, hasTouch: mobile })
    await page.addStyleTag({ content: 'header.sticky{visibility:hidden!important}' })
    await sleep(500)
    for (let k = 1; k <= 12; k++) {
      // рамку главы меряем перед каждым срезом: главы выше могут домонтироваться и сдвинуть страницу
      const box = await page.evaluate((s) => {
        const r = document.querySelector(s).getBoundingClientRect()
        return { top: r.top + scrollY, height: r.height }
      }, sel)
      const y = (k - 1) * slice
      if (y >= box.height) break
      const h = Math.min(slice, box.height - y)
      await page.evaluate((top) => scrollTo(0, top), box.top + y)
      await sleep(350)
      const path = join(out, `page-${String(k).padStart(2, '0')}.png`)
      await page.screenshot({ path, clip: { x: 0, y: box.top + y, width, height: h }, captureBeyondViewport: false })
      report.shots.push(path)
    }
  }
} finally {
  await browser.close()
}
console.log(JSON.stringify(report, null, 2))

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms))
}

/**
 * Снимок элемента с экрана: прокрутить к центру, при нужде вытянуть окно по высоте (ширину не трогаем,
 * чтобы графики на useElementSize не перемерились), подождать анимации и снять область.
 */
async function shootElement(page, handle, path, { settle = 600 } = {}) {
  // фигура внутри закрытого «Глубже» (hidden="until-found") не прокручивается и не видна: на время снимка открываем блок
  const opened = await handle.evaluate((el) => {
    const btn = el.closest('[hidden]')?.closest('.deeper')?.querySelector(':scope > button[aria-expanded="false"]')
    if (!btn) return false
    btn.click()
    return true
  })
  if (opened) await sleep(400)
  try {
    await shootVisible(page, handle, path, settle)
  } finally {
    if (opened) await handle.evaluate((el) => el.closest('.deeper')?.querySelector(':scope > button[aria-expanded="true"]')?.click())
  }
}

async function shootVisible(page, handle, path, settle) {
  const vp = page.viewport()
  const h0 = await handle.evaluate((el) => el.getBoundingClientRect().height)
  if (!(h0 > 0)) throw new Error('элемент скрыт (нулевая высота)')
  const need = Math.ceil(h0) + 24
  const tall = need > vp.height
  if (tall) await page.setViewport({ ...vp, height: Math.min(need, 16000) })
  try {
    await handle.evaluate((el) => el.scrollIntoView({ block: 'center', behavior: 'instant' }))
    await sleep(settle)
    const clip = await handle.evaluate((el) => {
      const r = el.getBoundingClientRect()
      return {
        x: Math.max(0, r.left + window.scrollX),
        y: r.top + window.scrollY,
        width: Math.max(1, Math.min(r.width, document.documentElement.clientWidth)),
        height: Math.max(1, r.height),
      }
    })
    await page.screenshot({ path, clip, captureBeyondViewport: false })
  } finally {
    if (tall) await page.setViewport(vp)
  }
}
function CSS_escape(s) {
  return String(s).replace(/[^a-zA-Z0-9_-]/g, (c) => `\\${c}`)
}
