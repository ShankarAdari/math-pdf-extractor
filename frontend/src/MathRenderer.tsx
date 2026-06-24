import React, { useEffect, useState, useRef } from 'react';

// Load KaTeX dynamically from CDN
const loadKaTeX = (): Promise<any> => {
  return new Promise((resolve, reject) => {
    if ((window as any).katex) {
      resolve((window as any).katex);
      return;
    }

    // Load CSS
    const link = document.createElement('link');
    link.rel = 'stylesheet';
    link.href = 'https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css';
    link.crossOrigin = 'anonymous';
    document.head.appendChild(link);

    // Load JS
    const script = document.createElement('script');
    script.src = 'https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js';
    script.crossOrigin = 'anonymous';
    script.onload = () => { resolve((window as any).katex); };
    script.onerror = (err) => { reject(err); };
    document.head.appendChild(script);
  });
};

interface MathRendererProps {
  text: string | any[] | any;
}

/**
 * Normalize text coming from the backend:
 * - Unescape double backslashes that JSON encoding produces (\\theta → \theta)
 * - Normalize line endings
 */
function normalizeLatex(s: string): string {
  // Replace double backslashes with single (JSON double-escaping artifact)
  // But only when they precede known LaTeX commands or are true escape sequences
  return s
    .replace(/\\\\([a-zA-Z{}\[\]()^_|.,!;:'"&% ])/g, '\\$1')
    .replace(/\\n/g, '\n')   // literal \n strings → newline
    .replace(/\\t(?=[^a-z])/g, '\t'); // only \t not followed by letter (preserve \theta etc.)
}

/**
 * Render a KaTeX formula safely. Returns an HTML element.
 */
function renderKatex(formula: string, katex: any, displayMode: boolean): HTMLElement {
  const el = document.createElement(displayMode ? 'div' : 'span');
  el.className = displayMode ? 'math-block-container' : 'math-inline-container';
  try {
    katex.render(formula, el, {
      displayMode,
      throwOnError: false,
      trust: false,
      strict: false,
    });
  } catch {
    el.textContent = displayMode ? `$$${formula}$$` : `$${formula}$`;
  }
  return el;
}

/**
 * Render inline content: handles $...$ math, **bold**, *italic*, plain text.
 */
function renderInlineContent(text: string, katex: any, container: HTMLElement) {
  // Split by inline math $...$ (not $$)
  const parts = text.split(/(?<!\$)(\$(?!\$)[^\$]*?\$)(?!\$)/g);

  parts.forEach((part) => {
    if (!part) return;

    if (part.startsWith('$') && part.endsWith('$') && !part.startsWith('$$')) {
      const formula = part.slice(1, -1).trim();
      if (formula) {
        container.appendChild(renderKatex(formula, katex, false));
      }
      return;
    }

    // Bold and italic
    const boldParts = part.split(/(\*\*[^*]+?\*\*)/g);
    boldParts.forEach((bp) => {
      if (!bp) return;
      if (bp.startsWith('**') && bp.endsWith('**')) {
        const strong = document.createElement('strong');
        strong.textContent = bp.slice(2, -2);
        container.appendChild(strong);
      } else {
        const italicParts = bp.split(/(\*[^*]+?\*)/g);
        italicParts.forEach((ip) => {
          if (!ip) return;
          if (ip.startsWith('*') && ip.endsWith('*') && !ip.startsWith('**')) {
            const em = document.createElement('em');
            em.textContent = ip.slice(1, -1);
            container.appendChild(em);
          } else {
            container.appendChild(document.createTextNode(ip));
          }
        });
      }
    });
  });
}

/**
 * Process a single line into an HTML element.
 */
function processLine(line: string, katex: any): HTMLElement | null {
  const trimmed = line.trim();

  // Empty line → spacer
  if (!trimmed) {
    const spacer = document.createElement('div');
    spacer.className = 'math-line-spacer';
    spacer.style.height = '0.4rem';
    return spacer;
  }

  // Display math $$...$$ on a single line
  if (trimmed.startsWith('$$') && trimmed.endsWith('$$') && trimmed.length > 4) {
    const formula = trimmed.slice(2, -2).trim();
    return renderKatex(formula, katex, true);
  }

  // Blockquote (> ...)
  if (trimmed.startsWith('> ')) {
    const el = document.createElement('blockquote');
    el.className = 'math-blockquote';
    renderInlineContent(trimmed.slice(2), katex, el);
    return el;
  }

  // Horizontal rule
  if (/^[-_*]{3,}$/.test(trimmed)) {
    const hr = document.createElement('hr');
    hr.className = 'math-hr';
    return hr;
  }

  // Headings (# or ##)
  if (trimmed.startsWith('## ')) {
    const el = document.createElement('h4');
    el.className = 'math-heading math-subheading';
    renderInlineContent(trimmed.slice(3), katex, el);
    return el;
  }
  if (trimmed.startsWith('# ')) {
    const el = document.createElement('h3');
    el.className = 'math-heading';
    renderInlineContent(trimmed.slice(2), katex, el);
    return el;
  }

  // Step header: "Step N: Title" or "**Step N: Title**"
  const stepMatch = trimmed.match(/^\*?\*?Step\s+(\d+)\s*:\s*(.*?)\*?\*?$/i);
  if (stepMatch) {
    const el = document.createElement('div');
    el.className = 'math-step-header';
    const badge = document.createElement('span');
    badge.className = 'math-step-badge';
    badge.textContent = `Step ${stepMatch[1]}`;
    el.appendChild(badge);
    if (stepMatch[2]?.trim()) {
      const title = document.createElement('span');
      title.className = 'math-step-title';
      renderInlineContent(stepMatch[2].replace(/\*+/g, ''), katex, title);
      el.appendChild(title);
    }
    return el;
  }

  // Formula name line: "Formula Name: $$...$$" or "Name: $$...$$"
  // These lines come from formulas_used sections like "Pythagorean Identity: $$\sin^2\theta...$$"
  const formulaNameMatch = trimmed.match(/^([^:$]+):\s*(\$\$?.*\$\$?)$/);
  if (formulaNameMatch) {
    const wrapper = document.createElement('div');
    wrapper.className = 'math-formula-item';
    const nameSpan = document.createElement('span');
    nameSpan.className = 'math-formula-name';
    nameSpan.textContent = formulaNameMatch[1].trim().replace(/^\d+\.\s*/, '');
    wrapper.appendChild(nameSpan);
    const mathSpan = document.createElement('span');
    mathSpan.className = 'math-formula-expr';
    renderInlineContent(formulaNameMatch[2].trim(), katex, mathSpan);
    wrapper.appendChild(mathSpan);
    return wrapper;
  }

  // Numbered list item: "1. content"
  // Only treat as numbered list if the content is meaningful (not just "Name:")
  if (/^\d+\.\s/.test(trimmed)) {
    const numberMatch = trimmed.match(/^(\d+)\.\s+([\s\S]*)$/);
    if (numberMatch) {
      const content = numberMatch[2];
      // If the content is only "Name:" style with no formula, render as formula label
      if (/^[A-Za-z\s]+:$/.test(content.trim())) {
        // It's a "Name:" label — skip rendering as list item, show as label
        const el = document.createElement('div');
        el.className = 'math-formula-label';
        el.textContent = content.trim();
        return el;
      }
      const el = document.createElement('div');
      el.className = 'math-list-item math-numbered-item';
      const numSpan = document.createElement('span');
      numSpan.className = 'math-list-number';
      numSpan.textContent = `${numberMatch[1]}.`;
      el.appendChild(numSpan);
      const contentSpan = document.createElement('span');
      contentSpan.className = 'math-list-content';
      renderInlineContent(content, katex, contentSpan);
      el.appendChild(contentSpan);
      return el;
    }
  }

  // Bullet list item
  if (trimmed.startsWith('- ') || trimmed.startsWith('• ')) {
    const el = document.createElement('div');
    el.className = 'math-list-item math-bullet-item';
    const bullet = document.createElement('span');
    bullet.className = 'math-list-bullet';
    bullet.textContent = '•';
    el.appendChild(bullet);
    const contentSpan = document.createElement('span');
    contentSpan.className = 'math-list-content';
    renderInlineContent(trimmed.slice(2), katex, contentSpan);
    el.appendChild(contentSpan);
    return el;
  }

  // Regular paragraph
  const el = document.createElement('div');
  el.className = 'math-paragraph';
  renderInlineContent(trimmed, katex, el);
  return el;
}

export const MathRenderer: React.FC<MathRendererProps> = ({ text }) => {
  const [katexLoaded, setKatexLoaded] = useState(false);
  const [loadError, setLoadError] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  // Normalize input to string
  let stringText = '';
  if (typeof text === 'string') {
    stringText = text;
  } else if (Array.isArray(text)) {
    stringText = text.map((step, idx) => {
      const stepStr = typeof step === 'string' ? step : JSON.stringify(step);
      if (/^\s*(?:\d+\.|[*\-])\s+/.test(stepStr)) return stepStr;
      return `${idx + 1}. ${stepStr}`;
    }).join('\n\n');
  } else if (text !== undefined && text !== null) {
    stringText = String(text);
  }

  // Fix double-backslash escaping from JSON transport
  stringText = normalizeLatex(stringText);

  useEffect(() => {
    loadKaTeX()
      .then(() => setKatexLoaded(true))
      .catch((err) => {
        console.error('KaTeX failed to load:', err);
        setLoadError(true);
      });
  }, []);

  useEffect(() => {
    if (!katexLoaded || !containerRef.current) return;
    const katex = (window as any).katex;
    if (!katex) return;

    containerRef.current.innerHTML = '';

    // Split text into display-math blocks ($$...$$) and text blocks
    const blocks: { type: 'text' | 'displaymath'; content: string }[] = [];
    const displayMathRegex = /\$\$([\s\S]*?)\$\$/g;
    let lastIndex = 0;
    let match;

    while ((match = displayMathRegex.exec(stringText)) !== null) {
      if (match.index > lastIndex) {
        blocks.push({ type: 'text', content: stringText.slice(lastIndex, match.index) });
      }
      blocks.push({ type: 'displaymath', content: match[1].trim() });
      lastIndex = match.index + match[0].length;
    }
    if (lastIndex < stringText.length) {
      blocks.push({ type: 'text', content: stringText.slice(lastIndex) });
    }

    blocks.forEach((block) => {
      if (block.type === 'displaymath') {
        containerRef.current?.appendChild(renderKatex(block.content, katex, true));
      } else {
        block.content.split('\n').forEach((line) => {
          const el = processLine(line, katex);
          if (el) containerRef.current?.appendChild(el);
        });
      }
    });
  }, [stringText, katexLoaded]);

  if (loadError) {
    return <div className="math-raw">{stringText}</div>;
  }

  if (!katexLoaded) {
    return <div className="math-loading">{stringText}</div>;
  }

  return <div ref={containerRef} className="math-rendered-text" />;
};
