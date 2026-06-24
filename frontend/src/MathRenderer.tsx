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
    script.onload = () => {
      resolve((window as any).katex);
    };
    script.onerror = (err) => {
      reject(err);
    };
    document.head.appendChild(script);
  });
};

interface MathRendererProps {
  text: string | any[] | any;
}

/**
 * Render inline math within a text node.
 * Splits text by $...$ patterns and renders KaTeX for math parts,
 * while also handling **bold** and *italic* markdown.
 */
function renderInlineContent(text: string, katex: any, container: HTMLElement) {
  // Split by $...$ (inline math) — careful not to match $$
  const parts = text.split(/(?<!\$)(\$(?!\$)[^\$]*?\$)(?!\$)/g);

  parts.forEach((part) => {
    if (!part) return;

    if (part.startsWith('$') && part.endsWith('$') && !part.startsWith('$$')) {
      // Inline math
      const formula = part.slice(1, -1).trim();
      if (formula) {
        const el = document.createElement('span');
        el.className = 'math-inline-container';
        try {
          katex.render(formula, el, { displayMode: false, throwOnError: false });
        } catch {
          el.innerText = part;
        }
        container.appendChild(el);
      }
    } else {
      // Process markdown bold (**...**) and italic (*...*)
      const boldParts = part.split(/(\*\*[^*]+?\*\*)/g);
      boldParts.forEach((bp) => {
        if (!bp) return;
        if (bp.startsWith('**') && bp.endsWith('**')) {
          const strong = document.createElement('strong');
          strong.textContent = bp.slice(2, -2);
          container.appendChild(strong);
        } else {
          // Check for italic
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
    }
  });
}

/**
 * Process a single line of text that may contain markdown-like formatting.
 * Returns an HTML element representing this line.
 */
function processLine(line: string, katex: any): HTMLElement {
  const trimmed = line.trim();

  // Empty line → spacer
  if (!trimmed) {
    const spacer = document.createElement('div');
    spacer.className = 'math-line-spacer';
    spacer.style.height = '0.5rem';
    return spacer;
  }

  // Display math $$...$$
  if (trimmed.startsWith('$$') && trimmed.endsWith('$$') && trimmed.length > 4) {
    const formula = trimmed.slice(2, -2).trim();
    const el = document.createElement('div');
    el.className = 'math-block-container';
    try {
      katex.render(formula, el, { displayMode: true, throwOnError: false });
    } catch {
      el.innerText = trimmed;
    }
    return el;
  }

  // Blockquote (> ...)
  if (trimmed.startsWith('> ')) {
    const el = document.createElement('blockquote');
    el.className = 'math-blockquote';
    renderInlineContent(trimmed.slice(2), katex, el);
    return el;
  }

  // Horizontal rule (--- or ___ or ***)
  if (/^[-_*]{3,}$/.test(trimmed)) {
    const hr = document.createElement('hr');
    hr.className = 'math-hr';
    return hr;
  }

  // Heading with emoji prefix or markdown # (e.g., "📖 **Concept:...**" or "## Title")
  if (trimmed.startsWith('# ')) {
    const el = document.createElement('h3');
    el.className = 'math-heading';
    renderInlineContent(trimmed.slice(2), katex, el);
    return el;
  }

  // Numbered list item (e.g., "1. ", "2. ")
  if (/^\d+\.\s/.test(trimmed)) {
    const el = document.createElement('div');
    el.className = 'math-list-item math-numbered-item';
    const numberMatch = trimmed.match(/^(\d+)\.\s/);
    const number = numberMatch ? numberMatch[1] : '1';
    const numSpan = document.createElement('span');
    numSpan.className = 'math-list-number';
    numSpan.textContent = `${number}.`;
    el.appendChild(numSpan);
    const contentSpan = document.createElement('span');
    contentSpan.className = 'math-list-content';
    renderInlineContent(trimmed.replace(/^\d+\.\s/, ''), katex, contentSpan);
    el.appendChild(contentSpan);
    return el;
  }

  // Bullet list item (- ... or • ...)
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

  // Step header detection (e.g., "Step 1: Title" or "**Step 1: Title**")
  const stepMatch = trimmed.match(/^\*?\*?Step\s+(\d+)\s*:\s*(.*?)\*?\*?$/i);
  if (stepMatch) {
    const el = document.createElement('div');
    el.className = 'math-step-header';
    const badge = document.createElement('span');
    badge.className = 'math-step-badge';
    badge.textContent = `Step ${stepMatch[1]}`;
    el.appendChild(badge);
    if (stepMatch[2]) {
      const title = document.createElement('span');
      title.className = 'math-step-title';
      renderInlineContent(stepMatch[2].replace(/\*+/g, ''), katex, title);
      el.appendChild(title);
    }
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

  // Normalize text
  let stringText = '';
  if (typeof text === 'string') {
    stringText = text;
  } else if (Array.isArray(text)) {
    stringText = text.map((step, idx) => {
      const stepStr = typeof step === 'string' ? step : JSON.stringify(step);
      if (/^\s*(?:\d+\.|[*\-])\s+/.test(stepStr)) {
        return stepStr;
      }
      return `${idx + 1}. ${stepStr}`;
    }).join('\n\n');
  } else if (text !== undefined && text !== null) {
    stringText = String(text);
  }

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

    // Clear previous contents
    containerRef.current.innerHTML = '';

    // First, handle multi-line display math ($$...$$) that span multiple lines
    // Split into blocks separated by $$...$$
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

    // Process each block
    blocks.forEach((block) => {
      if (block.type === 'displaymath') {
        const el = document.createElement('div');
        el.className = 'math-block-container';
        try {
          katex.render(block.content, el, { displayMode: true, throwOnError: false });
        } catch {
          el.innerText = `$$${block.content}$$`;
        }
        containerRef.current?.appendChild(el);
      } else {
        // Split text block into lines and process each
        const lines = block.content.split('\n');
        lines.forEach((line) => {
          const el = processLine(line, katex);
          containerRef.current?.appendChild(el);
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

  return (
    <div
      ref={containerRef}
      className="math-rendered-text"
    />
  );
};
