/* ============================================================
   navigation.js
   مسؤول عن التنقل بين الصفحات وتغيير العنوان في الـ topbar
   ============================================================ */

// عنوان كل صفحة — بيتعرض في الـ topbar
const PAGE_TITLES = {
  dashboard: 'Analytics Dashboard',
  player:    'Player Efficiency & Style Profiling',
  cohesion:  'Team Cohesion Analysis',
  injury:    'Injury Risk Prediction',
  env:       'Environmental Impact Analysis',
  winprob:   'Win Probability Modeling',
};

/**
 * navigateTo(pageId)
 * بتنقل للصفحة المطلوبة وبتعمل lazy init للشارتات لو محتاج
 */
function navigateTo(pageId) {
  // 1. إخفاء كل الصفحات
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));

  // 2. إخفاء كل nav items
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));

  // 3. تفعيل الصفحة المطلوبة
  const targetPage = document.getElementById('page-' + pageId);
  if (targetPage) targetPage.classList.add('active');

  // 4. تفعيل الـ nav item المقابل
  const targetNav = document.querySelector(`.nav-item[data-page="${pageId}"]`);
  if (targetNav) targetNav.classList.add('active');

  // 5. تحديث العنوان في الـ topbar
  const titleEl = document.getElementById('pageTitle');
  if (titleEl) titleEl.textContent = PAGE_TITLES[pageId] || '';

}

/**
 * init()
 * بنربط كل الـ nav items بالـ click event
 */
function initNavigation() {
  document.querySelectorAll('.nav-item').forEach(item => {
    item.addEventListener('click', () => {
      navigateTo(item.dataset.page);
    });
  });
}

// Export للاستخدام في main.js
const Navigation = { init: initNavigation, navigateTo };
