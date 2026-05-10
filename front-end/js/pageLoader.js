/* ============================================================
   pageLoader.js
   بيحمّل HTML كل صفحة من ملف منفصل وبيحطه في الـ shell
   ليه؟ عشان ملف index.html يبقى نضيف ومش ضخم

   NOTE: في الـ production بنستخدم framework (Vue/React)
   هنا بنعمله manually بـ fetch
   ============================================================ */

/* ── قائمة الصفحات وأماكنها ── */
const PAGES = [
  { id: 'dashboard-content', file: '/static/pages/dashboard.html' },
  { id: 'player-content',    file: '/static/pages/player.html'    },
  { id: 'cohesion-content',  file: '/static/pages/cohesion.html'  },
  { id: 'injury-content',    file: '/static/pages/injury.html'    },
  { id: 'env-content',       file: '/static/pages/env.html'       },
  { id: 'winprob-content',   file: '/static/pages/winprob.html'   },
];

/**
 * loadPage(page)
 * تحمل HTML صفحة واحدة وتحطها في الـ container المناسب
 */
async function loadPage({ id, file }) {
  try {
    const res  = await fetch(file);
    if (!res.ok) throw new Error(`Failed: ${file}`);
    const html = await res.text();
    const container = document.getElementById(id);
    if (container) container.innerHTML = html;
  } catch (err) {
    console.error(`[pageLoader] ${err.message}`);
  }
}

/**
 * loadAllPages()
 * بتحمّل كل الصفحات بالتوازي (Promise.all)
 * بعد ما يخلصوا، بنعمل init للـ dashboard charts
 */
async function loadAllPages() {
  await Promise.all(PAGES.map(loadPage));
}

/* ── ابدأ التحميل ── */
window.pagesLoadedPromise = loadAllPages();
