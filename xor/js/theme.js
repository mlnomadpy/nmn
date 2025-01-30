export class ThemeController {
  constructor() {
    this.themeKey = 'neural-theme';
    this.transitionDuration = 800;
    this.themes = {
      light: {
        name: 'light',
        icon: `<path d="M12 16a4 4 0 1 0 0-8 4 4 0 0 0 0 8z"></path>
               <path d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707"></path>`,
        label: 'Activate dark mode'
      },
      dark: {
        name: 'dark',
        icon: `<path d="M21 12.79A9 9 0 1 1 11.21 3 A7 7 0 0 0 21 12.79z"></path>`,
        label: 'Activate light mode'
      }
    };
    this.isThemeChanging = false;
    this.init();
  }

  init() {
    this.button = document.querySelector('.theme-toggle');
    this.setInitialTheme();
    this.addEventListeners();
    this.createThemeTransitionOverlay();
  }

  setInitialTheme() {
    const savedTheme = localStorage.getItem(this.themeKey);
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    const initialTheme = savedTheme || (prefersDark ? 'dark' : 'light');
    this.setTheme(initialTheme, false); // false = no animation on initial load
  }

  createThemeTransitionOverlay() {
    const overlay = document.createElement('div');
    overlay.className = 'theme-transition-overlay';
    overlay.style.cssText = `
      position: fixed;
      pointer-events: none;
      top: 0;
      left: 0;
      width: 100vw;
      height: 100vh;
      z-index: 9999;
      opacity: 0;
      transition: opacity ${this.transitionDuration}ms ease;
      mix-blend-mode: difference;
      background: #fff;
    `;
    document.body.appendChild(overlay);
    this.overlay = overlay;
  }

  addEventListeners() {
    this.button.addEventListener('click', () => this.toggleTheme());
    
    // Add hover effect
    this.button.addEventListener('mousemove', (e) => this.handleButtonHover(e));
    this.button.addEventListener('mouseleave', () => this.handleButtonLeave());
  }

  handleButtonHover(e) {
    const rect = e.target.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    this.button.style.transform = `
      perspective(500px)
      rotateX(${(y - rect.height / 2) / 10}deg)
      rotateY(${-(x - rect.width / 2) / 10}deg)
      translateZ(10px)
    `;
  }

  handleButtonLeave() {
    this.button.style.transform = 'none';
  }

  async toggleTheme() {
    if (this.isThemeChanging) return;
    this.isThemeChanging = true;

    const newTheme = document.documentElement.getAttribute('data-theme') === 'dark' 
      ? 'light' 
      : 'dark';
    
    await this.setTheme(newTheme, true);
    this.isThemeChanging = false;
  }

  async setTheme(theme, animate = true) {
    const { icon, label } = this.themes[theme];
    
    if (animate) {
      await this.playThemeAnimation();
    }

    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem(this.themeKey, theme);
    
    this.button.querySelector('svg').innerHTML = icon;
    this.button.setAttribute('aria-label', label);
    
    // Dispatch event for other components
    window.dispatchEvent(new CustomEvent('themechange', { detail: { theme } }));
  }

  async playThemeAnimation() {
    return new Promise(resolve => {
      requestAnimationFrame(() => {
        this.overlay.style.opacity = '1';
        this.button.style.transform = 'rotate(360deg) scale(0.8)';
        
        setTimeout(() => {
          this.button.style.transform = 'none';
          this.overlay.style.opacity = '0';
          resolve();
        }, this.transitionDuration);
      });
    });
  }
}

