import { useState, useRef, useEffect } from 'react';
import { MathRenderer } from './MathRenderer';

/* ── Types ─────────────────────────────────── */
interface QuestionResult {
  id: string;
  question_number: string;
  image_url?: string | null;
  text?: string;
  has_diagram?: boolean;
  diagram_url?: string | null;
}

interface ApiResponse {
  user_name: string;
  questions_count: number;
  output_format: 'text' | 'image';
  extraction_mode: string;
  questions: QuestionResult[];
}

interface OcrResponse {
  filename: string;
  extraction_mode: string;
  text: string;
  char_count: number;
  image_url?: string | null;
}

interface SolutionState {
  loading: boolean;
  error: string | null;
  result: {
    solver_mode: string;
    provider_model?: string;
    problem_type: string;
    difficulty_level?: string;
    formulas_used?: string;
    theory_explanation?: string;
    reasoning?: string;
    solution_steps: string;
    final_answer: string;
  } | null;
}

interface HistoryItem {
  id: string;
  timestamp: number;
  question: string;
  hasImage: boolean;
  imageUrl?: string | null;
  result: {
    solver_mode: string;
    provider_model?: string;
    problem_type: string;
    difficulty_level?: string;
    formulas_used?: string;
    theory_explanation?: string;
    reasoning?: string;
    solution_steps: string;
    final_answer: string;
  };
}

const API_BASE = 'http://localhost:8000';

const STEPS = [
  'Loading PDF and rendering pages…',
  'Running AI / local extraction pipeline…',
  'Parsing questions and bounding boxes…',
  'Cropping question regions and saving images…',
];

type Tab = 'pdf' | 'ocr' | 'direct' | 'history';

/* ═══════════════════════════════════════════════════════════ */
export default function App() {
  const [tab, setTab] = useState<Tab>('direct');

  /* ── Theme state ── */
  const [darkMode, setDarkMode] = useState(false);

  const toggleDarkMode = () => {
    const nextMode = !darkMode;
    setDarkMode(nextMode);
    if (nextMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  };

  /* ── Shared Config state ── */
  const [apiKey, setApiKey] = useState(() => {
    return localStorage.getItem('gemini_api_key_extractor') || '';
  });

  const [keyStatus, setKeyStatus] = useState<'unchecked' | 'validating' | 'valid' | 'invalid' | 'missing' | 'quota'>('unchecked');
  const [keyError, setKeyError] = useState<string | null>(null);

  /* ── AI Provider Config ── */
  const [aiProvider, setAiProvider] = useState<'auto' | 'gemini' | 'groq' | 'ollama' | 'offline'>(() =>
    (localStorage.getItem('ai_provider') as any) || 'auto'
  );
  const [groqKey, setGroqKey] = useState(() => localStorage.getItem('groq_api_key') || import.meta.env.VITE_GROQ_API_KEY || '');
  const [ollamaModel, setOllamaModel] = useState(() => localStorage.getItem('ollama_model') || 'llama3');
  const [ollamaUrl, setOllamaUrl] = useState(() => localStorage.getItem('ollama_url') || 'http://localhost:11434');
  const [showProviderSettings, setShowProviderSettings] = useState(false);

  useEffect(() => { localStorage.setItem('ai_provider', aiProvider); }, [aiProvider]);
  useEffect(() => { localStorage.setItem('groq_api_key', groqKey); }, [groqKey]);
  useEffect(() => { localStorage.setItem('ollama_model', ollamaModel); }, [ollamaModel]);
  useEffect(() => { localStorage.setItem('ollama_url', ollamaUrl); }, [ollamaUrl]);

  /* Helper: build shared AI provider headers */
  const buildProviderHeaders = () => {
    const h: Record<string, string> = { 'X-AI-Provider': aiProvider };
    if (apiKey.trim()) h['X-Gemini-API-Key'] = apiKey.trim();
    if (groqKey.trim()) h['X-Groq-API-Key'] = groqKey.trim();
    if (ollamaModel.trim()) h['X-Ollama-Model'] = ollamaModel.trim();
    if (ollamaUrl.trim()) h['X-Ollama-Base-URL'] = ollamaUrl.trim();
    return h;
  };

  const getSolverBadgeText = (mode?: string, modelInfo?: string) => {
    if (mode === 'gemini_ai') return `🤖 Gemini AI${modelInfo ? ` (${modelInfo})` : ''}`;
    if (mode === 'groq_ai') return `⚡ Groq AI${modelInfo ? ` (${modelInfo})` : ''}`;
    if (mode === 'ollama_ai') return `🦙 Ollama AI${modelInfo ? ` (${modelInfo})` : ''}`;
    if (mode === 'free_local_sympy') return '🔓 SymPy Offline';
    if (mode === 'free_local_tutor') return '📝 Local Tutor';
    return '🔓 Offline Solver';
  };

  const formatFinalAnswer = (ans: any): string => {
    if (!ans) return '';
    let str = '';
    if (typeof ans === 'string') {
      str = ans;
    } else if (typeof ans === 'object') {
      str = ans.answer || ans.result || ans.final_answer || JSON.stringify(ans);
    } else {
      str = String(ans);
    }
    str = str.trim();
    
    // If it already has LaTeX delimiters or backslashes, return as-is
    if (str.startsWith('$') || str.includes('$$') || str.includes('\\')) {
      return str;
    }
    
    // Check if it's plain text rather than math.
    // If it contains letters and spaces and does not contain typical math operators/notations
    // (=, +, -, *, /, ^, _, <, >, {, }), then it's plain text. Do not wrap in $.
    const hasLetters = /[a-zA-Z]/.test(str);
    const hasSpaces = /\s/.test(str);
    const hasMathOperators = /[=\+\-\*\/_^{}<>]/.test(str);
    
    if (hasLetters && hasSpaces && !hasMathOperators) {
      return str;
    }
    
    return `$${str}$`;
  };

  const getSpeakableAnswer = (ans: any): string => {
    if (!ans) return '';
    if (typeof ans === 'string') return ans;
    if (typeof ans === 'object') return ans.explanation || ans.answer || ans.result || JSON.stringify(ans);
    return String(ans);
  };

  /* Global Paste Event Listener for Clipboard Snips */
  useEffect(() => {
    const handleGlobalPaste = (e: ClipboardEvent) => {
      const items = e.clipboardData?.items;
      if (!items) return;
      for (let i = 0; i < items.length; i++) {
        if (items[i].type.startsWith('image/')) {
          const file = items[i].getAsFile();
          if (file) {
            e.preventDefault();
            if (tab === 'direct') {
              setDirectFile(file);
              setDirectError(null);
            } else if (tab === 'ocr') {
              setOcrFile(file);
              setOcrError(null);
            }
            break;
          }
        }
      }
    };
    window.addEventListener('paste', handleGlobalPaste);
    return () => {
      window.removeEventListener('paste', handleGlobalPaste);
    };
  }, [tab]);

  useEffect(() => {
    localStorage.setItem('gemini_api_key_extractor', apiKey);
    
    if (!apiKey.trim()) {
      setKeyStatus('missing');
      setKeyError(null);
      return;
    }

    setKeyStatus('validating');
    const timer = setTimeout(async () => {
      try {
        const res = await fetch(`${API_BASE}/api/validate-key`, {
          method: 'POST',
          headers: {
            'X-Gemini-API-Key': apiKey.trim()
          }
        });
        if (!res.ok) {
          throw new Error('Validation request failed.');
        }
        const data = await res.json();
        if (data.valid) {
          if (data.status === 'quota') {
            setKeyStatus('quota');
            setKeyError(data.message || 'Free-tier quota exhausted. App uses SymPy fallback.');
          } else {
            setKeyStatus('valid');
            setKeyError(null);
          }
        } else {
          setKeyStatus('invalid');
          setKeyError(data.message || 'Key is invalid.');
        }
      } catch (err: any) {
        setKeyStatus('invalid');
        setKeyError(err.message || 'Failed to validate key.');
      }
    }, 800);

    return () => clearTimeout(timer);
  }, [apiKey]);


  /* ── PDF state ── */
  const [name, setName]           = useState('');
  const [numQs, setNumQs]         = useState(5);
  const [fmt, setFmt]             = useState<'text' | 'image'>('text');
  const [pdfFile, setPdfFile]     = useState<File | null>(null);
  const [drag, setDrag]           = useState(false);
  const [loading, setLoading]     = useState(false);
  const [step, setStep]           = useState(-1);
  const [error, setError]         = useState<string | null>(null);
  const [results, setResults]     = useState<ApiResponse | null>(null);
  const pdfRef = useRef<HTMLInputElement>(null);

  /* ── OCR state ── */
  const [ocrFile, setOcrFile]     = useState<File | null>(null);
  const [ocrDrag, setOcrDrag]     = useState(false);
  const [ocrLoading, setOcrLoading] = useState(false);
  const [ocrError, setOcrError]   = useState<string | null>(null);
  const [ocrResult, setOcrResult] = useState<OcrResponse | null>(null);
  const ocrRef = useRef<HTMLInputElement>(null);

  /* ── Direct Solver state ── */
  const [directText, setDirectText]       = useState('');
  const [directFile, setDirectFile]       = useState<File | null>(null);
  const [directDrag, setDirectDrag]       = useState(false);
  const [directLoading, setDirectLoading] = useState(false);
  const [directError, setDirectError]     = useState<string | null>(null);
  const [directResult, setDirectResult]   = useState<SolutionState['result']>(null);
  const directInputRef = useRef<HTMLInputElement>(null);

  /* ── Saved History state ── */
  const [history, setHistory] = useState<HistoryItem[]>(() => {
    try {
      const stored = localStorage.getItem('math_solver_history');
      return stored ? JSON.parse(stored) : [];
    } catch {
      return [];
    }
  });

  /* ── Speech Synthesis state ── */
  const [speakingId, setSpeakingId] = useState<string | null>(null);

  useEffect(() => {
    return () => {
      window.speechSynthesis.cancel();
    };
  }, []);

  /* ── Solver state & handler (PDF / OCR) ── */
  const [solutions, setSolutions] = useState<Record<string, SolutionState>>({});
  const [selectedQuestions, setSelectedQuestions] = useState<Set<string>>(new Set());
  const [batchSolving, setBatchSolving] = useState(false);

  const toggleQuestionSelection = (qid: string) => {
    setSelectedQuestions(prev => {
      const next = new Set(prev);
      if (next.has(qid)) {
        next.delete(qid);
      } else {
        next.add(qid);
      }
      return next;
    });
  };

  const selectAllQuestions = () => {
    if (!results) return;
    setSelectedQuestions(new Set(results.questions.map(q => q.id)));
  };

  const deselectAllQuestions = () => {
    setSelectedQuestions(new Set());
  };

  const handleSolveSelected = async () => {
    if (selectedQuestions.size === 0 || !results) return;
    setBatchSolving(true);
    const qids = Array.from(selectedQuestions);
    for (const qid of qids) {
      const q = results.questions.find(x => x.id === qid);
      if (q) {
        await handleSolve(q.id, q.text, q.image_url);
      }
    }
    setBatchSolving(false);
  };

  const saveToHistory = (question: string, imageUrl: string | null, result: SolutionState['result']) => {
    if (!result) return;
    const newItem: HistoryItem = {
      id: Math.random().toString(36).substring(2, 9),
      timestamp: Date.now(),
      question,
      hasImage: !!imageUrl,
      imageUrl,
      result
    };
    setHistory(prev => {
      const updated = [newItem, ...prev].slice(0, 50); // limit to 50 items
      localStorage.setItem('math_solver_history', JSON.stringify(updated));
      return updated;
    });
  };

  const clearHistory = () => {
    localStorage.removeItem('math_solver_history');
    setHistory([]);
  };


  const handleSolve = async (qid: string, questionText?: string, imageUrl?: string | null) => {
    setSolutions(prev => ({
      ...prev,
      [qid]: { loading: true, error: null, result: null }
    }));

    try {
      const res = await fetch(`${API_BASE}/api/solve`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...buildProviderHeaders()
        },
        body: JSON.stringify({
          text: questionText || '',
          image_url: imageUrl || ''
        })
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || 'Failed to solve question.');
      }

      const data = await res.json();
      setSolutions(prev => ({
        ...prev,
        [qid]: { loading: false, error: null, result: data }
      }));

      // Save to local history
      saveToHistory(questionText || 'Visual / Image Question', imageUrl || null, data);
    } catch (err: any) {
      setSolutions(prev => ({
        ...prev,
        [qid]: { loading: false, error: err.message || 'Unexpected error.', result: null }
      }));
    }
  };

  /* ── Speech synthesis helper ── */
  const cleanMathForSpeech = (text: string): string => {
    let s = text;
    // Replace fractions: \frac{A}{B} -> A over B
    s = s.replace(/\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}/g, '$1 over $2');
    // Replace exponents: x^2 -> x squared, x^3 -> x cubed, x^n -> x to the power of n
    s = s.replace(/([a-zA-Z0-9])\^2\b/g, '$1 squared');
    s = s.replace(/([a-zA-Z0-9])\^3\b/g, '$1 cubed');
    s = s.replace(/([a-zA-Z0-9])\^\{?([a-zA-Z0-9\-+]+)\}?/g, '$1 to the power of $2');
    // Replace integrals
    s = s.replace(/\\int\b/g, 'the integral of ');
    s = s.replace(/\\int_\{?([a-zA-Z0-9]+)\}?\^\{?([a-zA-Z0-9]+)\}?/g, 'the integral from $1 to $2 of ');
    // Replace standard math symbols
    s = s.replace(/\\pi\b/g, 'pi');
    s = s.replace(/\\theta\b/g, 'theta');
    s = s.replace(/\\infty\b/g, 'infinity');
    s = s.replace(/\\sqrt\s*\{([^{}]+)\}/g, 'the square root of $1');
    s = s.replace(/\\sqrt\b/g, 'square root');
    s = s.replace(/\\sum\b/g, 'the sum of');
    s = s.replace(/\\pm\b/g, 'plus or minus');
    s = s.replace(/\\neq\b/g, 'is not equal to');
    s = s.replace(/\\le\b/g, 'less than or equal to');
    s = s.replace(/\\ge\b/g, 'greater than or equal to');
    // Clean Markdown / LaTeX blocks
    s = s.replace(/\$\$/g, ' ');
    s = s.replace(/\$/g, ' ');
    s = s.replace(/\\cdot/g, ' times ');
    s = s.replace(/\\times/g, ' times ');
    s = s.replace(/\*/g, ' times ');
    s = s.replace(/\\left/g, '');
    s = s.replace(/\\right/g, '');
    s = s.replace(/\\/g, ' ');
    s = s.replace(/\*\*([^*]+)\*\*/g, '$1');
    s = s.replace(/\*([^*]+)\*/g, '$1');
    s = s.replace(/#+\s+/g, '');
    return s;
  };

  const handleSpeak = (id: string, textToSpeak: string) => {
    if (speakingId === id) {
      window.speechSynthesis.cancel();
      setSpeakingId(null);
      return;
    }
    window.speechSynthesis.cancel();
    const cleanText = cleanMathForSpeech(textToSpeak);
    const utterance = new SpeechSynthesisUtterance(cleanText);
    utterance.onend = () => setSpeakingId(null);
    utterance.onerror = () => setSpeakingId(null);
    setSpeakingId(id);
    window.speechSynthesis.speak(utterance);
  };

  /* ── Direct Solver Handlers ── */
  const onDirectDrag = (e: React.DragEvent) => {
    e.preventDefault(); e.stopPropagation();
    setDirectDrag(e.type === 'dragenter' || e.type === 'dragover');
  };
  const onDirectDrop = (e: React.DragEvent) => {
    e.preventDefault(); e.stopPropagation(); setDirectDrag(false);
    const f = e.dataTransfer.files?.[0];
    if (f && f.type.startsWith('image/')) {
      setDirectFile(f); setDirectError(null);
    } else setDirectError('Only image files are accepted.');
  };
  const onDirectFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f && f.type.startsWith('image/')) {
      setDirectFile(f); setDirectError(null);
    } else setDirectError('Only image files are accepted.');
  };
  const handleDirectReset = () => {
    setDirectText('');
    setDirectFile(null);
    setDirectError(null);
    setDirectResult(null);
    if (speakingId === 'direct') {
      window.speechSynthesis.cancel();
      setSpeakingId(null);
    }
  };

  useEffect(() => {
    if (directFile) {
      handleDirectSolve();
    }
  }, [directFile]);

  const handleDirectSolve = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!directText.trim() && !directFile) {
      setDirectError('Please type a math question or upload a photo/screenshot.');
      return;
    }
    setDirectLoading(true);
    setDirectError(null);
    setDirectResult(null);

    let finalQueryText = directText;
    let uploadedImageUrl: string | null = null;


    try {
      // 1. If image attached, run OCR first to get text and image_url on server
      if (directFile) {
        const fd = new FormData();
        fd.append('image', directFile);
        
        const ocrRes = await fetch(`${API_BASE}/api/ocr`, {
          method: 'POST',
          headers: buildProviderHeaders(),
          body: fd
        });

        if (!ocrRes.ok) {
          const err = await ocrRes.json().catch(() => ({}));
          throw new Error(err.detail || 'Failed to analyze uploaded image.');
        }

        const ocrData = await ocrRes.json();
        uploadedImageUrl = ocrData.image_url || null;
        
        if (!finalQueryText.trim()) {
          finalQueryText = ocrData.text;
          setDirectText(ocrData.text);
        }
      }

      // 2. Solve the text question (and optional image)
      const solveRes = await fetch(`${API_BASE}/api/solve`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...buildProviderHeaders()
        },
        body: JSON.stringify({
          text: finalQueryText,
          image_url: uploadedImageUrl
        })
      });

      if (!solveRes.ok) {
        const err = await solveRes.json().catch(() => ({}));
        throw new Error(err.detail || 'Failed to solve math problem.');
      }

      const solveData = await solveRes.json();
      if (solveData.status === 'error') {
        throw new Error(solveData.solution_steps || 'Error solving problem.');
      }

      setDirectResult(solveData);
      saveToHistory(finalQueryText || 'Visual Question', uploadedImageUrl, solveData);
    } catch (err: any) {
      setDirectError(err.message || 'Failed to resolve equation.');
    } finally {
      setDirectLoading(false);
    }
  };

  /* ── PDF drag handlers ── */
  const onDrag = (e: React.DragEvent) => {
    e.preventDefault(); e.stopPropagation();
    setDrag(e.type === 'dragenter' || e.type === 'dragover');
  };
  const onDrop = (e: React.DragEvent) => {
    e.preventDefault(); e.stopPropagation(); setDrag(false);
    const f = e.dataTransfer.files?.[0];
    if (f && (f.type === 'application/pdf' || f.name.toLowerCase().endsWith('.pdf'))) {
      setPdfFile(f); setError(null);
    } else setError('Only PDF files are accepted.');
  };
  const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f && (f.type === 'application/pdf' || f.name.toLowerCase().endsWith('.pdf'))) {
      setPdfFile(f); setError(null);
    } else setError('Only PDF files are accepted.');
  };

  /* ── OCR drag handlers ── */
  const onOcrDrag = (e: React.DragEvent) => {
    e.preventDefault(); e.stopPropagation();
    setOcrDrag(e.type === 'dragenter' || e.type === 'dragover');
  };
  const onOcrDrop = (e: React.DragEvent) => {
    e.preventDefault(); e.stopPropagation(); setOcrDrag(false);
    const f = e.dataTransfer.files?.[0];
    if (f && f.type.startsWith('image/')) { setOcrFile(f); setOcrError(null); }
    else setOcrError('Only image files (PNG, JPG, WEBP, BMP…) are accepted.');
  };
  const onOcrFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f && f.type.startsWith('image/')) { setOcrFile(f); setOcrError(null); }
    else setOcrError('Only image files are accepted.');
  };

  /* ── PDF submit ── */
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!pdfFile) { setError('Please upload a PDF file.'); return; }
    setLoading(true); setError(null); setResults(null); setStep(0);
    const timers = [
      setTimeout(() => setStep(1), 1800),
      setTimeout(() => setStep(2), 4000),
      setTimeout(() => setStep(3), 6500),
    ];
    const fd = new FormData();
    fd.append('pdf', pdfFile);
    fd.append('num_questions', numQs.toString());
    fd.append('output_format', fmt);
    fd.append('name', name);
    const headers: Record<string, string> = buildProviderHeaders();
    try {
      const res = await fetch(`${API_BASE}/api/process`, { method: 'POST', headers, body: fd });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || 'Extraction failed.');
      }
      setResults(await res.json());
      setStep(4);
    } catch (err: any) {
      setError(err.message || 'Unexpected error.');
    } finally {
      timers.forEach(clearTimeout);
      setLoading(false);
    }
  };

  useEffect(() => {
    if (ocrFile) {
      handleOcr();
    }
  }, [ocrFile]);

  /* ── OCR submit ── */
  const handleOcr = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!ocrFile) { setOcrError('Please upload an image.'); return; }
    setOcrLoading(true); setOcrError(null); setOcrResult(null);
    const fd = new FormData();
    fd.append('image', ocrFile);
    const headers: Record<string, string> = buildProviderHeaders();
    try {
      const res = await fetch(`${API_BASE}/api/ocr`, { method: 'POST', headers, body: fd });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || 'OCR failed.');
      }
      const data = await res.json();
      setOcrResult(data);
      // Automatically trigger solve for the extracted text!
      handleSolve('ocr-solver', data.text, data.image_url);
    } catch (err: any) {
      setOcrError(err.message || 'Unexpected error.');
    } finally {
      setOcrLoading(false);
    }
  };


  const downloadJson = () => {
    if (!results) return;
    const blob = new Blob([JSON.stringify(results, null, 2)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `math_questions_${name || 'export'}.json`;
    a.click();
  };

  const copyOcr = () => {
    if (ocrResult) navigator.clipboard.writeText(ocrResult.text);
  };

  const reset = () => { setResults(null); setPdfFile(null); setError(null); setStep(-1); setSelectedQuestions(new Set()); };

  /* ═══════ RENDER ═══════ */
  return (
    <div className="container">
      <div className="glow-orb glow-orb-1" />
      <div className="glow-orb glow-orb-2" />

      {/* App Header */}
      <header className="app-header">
        <h1 className="app-title">Math PDF Question Extractor &amp; Solver</h1>
        <p className="app-subtitle">
          Input your own questions, upload screenshots, or extract entire math exam papers from PDFs. 
          Get detailed formulas, step-by-step chalkboard proofs, and speech output.
        </p>
      </header>

      {/* Global Configuration Panel */}
      <div className="glass-panel" style={{ marginBottom: '1.5rem', padding: '1rem' }}>
        <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap', justifyContent: 'space-between', alignItems: 'center' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flex: 1, minWidth: '280px' }}>
            <span style={{ fontSize: '0.9rem', fontWeight: 600, color: 'var(--teal)' }}>🔑 Gemini API Key:</span>
            <input
              className="input-text"
              type="password"
              style={{ margin: 0, padding: '0.35rem 0.75rem', fontSize: '0.85rem', flex: 1 }}
              placeholder="AIza… (Optional — for advanced Vision &amp; AI features)"
              value={apiKey}
              onChange={e => setApiKey(e.target.value)}
            />
            {keyStatus === 'validating' && (
              <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', display: 'inline-flex', alignItems: 'center', gap: '0.25rem' }}>
                <span className="spinner-mini" style={{ borderLeftColor: 'var(--teal)', width: '12px', height: '12px', borderWidth: '2px', display: 'inline-block' }}></span> Checking…
              </span>
            )}
            {keyStatus === 'valid' && (
              <span style={{ fontSize: '0.85rem', color: '#a3be8c', fontWeight: 600, display: 'inline-flex', alignItems: 'center', gap: '0.25rem' }} title="API Key is valid and working!">
                🟢 Valid
              </span>
            )}
            {keyStatus === 'quota' && (
              <span style={{ fontSize: '0.85rem', color: '#ebcb8b', fontWeight: 600, display: 'inline-flex', alignItems: 'center', gap: '0.25rem' }} title="Free-tier daily quota used up. App works via SymPy. Resets tomorrow.">
                ⚠️ Quota Limit
              </span>
            )}
            {keyStatus === 'invalid' && (
              <span style={{ fontSize: '0.85rem', color: '#bf616a', fontWeight: 600, display: 'inline-flex', alignItems: 'center', gap: '0.25rem' }} title={keyError || 'API Key is invalid.'}>
                🔴 Invalid
              </span>
            )}
          </div>
          
          <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
            <button
              onClick={() => setShowProviderSettings(!showProviderSettings)}
              className="btn-secondary"
              style={{ padding: '0.4rem 1rem', fontSize: '0.85rem', width: 'auto', margin: 0, border: showProviderSettings ? '1px solid var(--teal)' : undefined }}
            >
              ⚙️ AI Provider Settings
            </button>
            <button
              onClick={toggleDarkMode}
              className="btn-secondary"
              style={{ padding: '0.4rem 1rem', fontSize: '0.85rem', width: 'auto', margin: 0 }}
            >
              {darkMode ? '☀️ Modern Clean Paper' : '🌙 Classic Chalkboard'}
            </button>
          </div>
        </div>

        {/* Expandable Advanced AI Settings */}
        {showProviderSettings && (
          <div style={{ marginTop: '1rem', paddingTop: '1rem', borderTop: '1px solid var(--border)', display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '1rem', animation: 'slideDown 0.25s ease-out' }}>
            <div className="form-group" style={{ margin: 0 }}>
              <label className="form-label" style={{ fontSize: '0.8rem', color: 'var(--teal)' }}>Preferred AI Backend</label>
              <select
                className="input-text"
                style={{ fontSize: '0.85rem', padding: '0.35rem 0.5rem', margin: 0 }}
                value={aiProvider}
                onChange={e => setAiProvider(e.target.value as any)}
              >
                <option value="auto">Auto Fallback Chain (Gemini → Groq → Ollama → SymPy)</option>
                <option value="gemini">Gemini API Only (Vision &amp; Solving)</option>
                <option value="groq">Groq Cloud API Only (Text Solving)</option>
                <option value="ollama">Local Ollama LLM Only (Offline &amp; Private)</option>
                <option value="offline">Offline / SymPy Solver Only</option>
              </select>
            </div>

            <div className="form-group" style={{ margin: 0 }}>
              <label className="form-label" style={{ fontSize: '0.8rem', color: 'var(--teal)' }}>
                Groq API Key <a href="https://console.groq.com" target="_blank" rel="noopener noreferrer" style={{ textDecoration: 'underline', color: 'var(--teal)', fontSize: '0.75rem' }}>(Get free key)</a>
              </label>
              <input
                className="input-text"
                type="password"
                style={{ fontSize: '0.85rem', padding: '0.35rem 0.5rem', margin: 0 }}
                placeholder="gsk_..."
                value={groqKey}
                onChange={e => setGroqKey(e.target.value)}
              />
            </div>

            <div className="form-group" style={{ margin: 0 }}>
              <label className="form-label" style={{ fontSize: '0.8rem', color: 'var(--teal)' }}>Ollama Local Model</label>
              <input
                className="input-text"
                type="text"
                style={{ fontSize: '0.85rem', padding: '0.35rem 0.5rem', margin: 0 }}
                placeholder="e.g. llama3, qwen2.5:7b"
                value={ollamaModel}
                onChange={e => setOllamaModel(e.target.value)}
              />
            </div>

            <div className="form-group" style={{ margin: 0 }}>
              <label className="form-label" style={{ fontSize: '0.8rem', color: 'var(--teal)' }}>Ollama Endpoint</label>
              <input
                className="input-text"
                type="text"
                style={{ fontSize: '0.85rem', padding: '0.35rem 0.5rem', margin: 0 }}
                placeholder="http://localhost:11434"
                value={ollamaUrl}
                onChange={e => setOllamaUrl(e.target.value)}
              />
            </div>
          </div>
        )}
      </div>

      {/* Tab Switcher */}
      <div className="tab-bar">
        <button
          className={`tab-btn ${tab === 'direct' ? 'tab-active' : ''}`}
          onClick={() => { setTab('direct'); }}
        >
          ✨ Ask a Question
        </button>
        <button
          className={`tab-btn ${tab === 'pdf' ? 'tab-active' : ''}`}
          onClick={() => { setTab('pdf'); }}
        >
          📄 PDF Exam Paper Extractor
        </button>
        <button
          className={`tab-btn ${tab === 'ocr' ? 'tab-active' : ''}`}
          onClick={() => { setTab('ocr'); }}
        >
          📷 Screenshot / Photo OCR
        </button>
        <button
          className={`tab-btn ${tab === 'history' ? 'tab-active' : ''}`}
          onClick={() => { setTab('history'); }}
        >
          📜 Saved Solutions ({history.length})
        </button>
      </div>

      {/* ══════════ DIRECT SOLVER TAB ══════════ */}
      {tab === 'direct' && (
        <div className="direct-solver-container">
          {/* Left Column: Direct Solve Form */}
          <div className="glass-panel">
            <h2 style={{ marginBottom: '1rem', fontFamily: 'var(--font-ui)' }}>Ask a Question</h2>
            <p style={{ color: 'var(--text-secondary)', marginBottom: '1.5rem', fontSize: '0.92rem' }}>
              Type your math problem directly or upload a photo/screenshot. The solver handles algebra, calculus, and general word problems.
            </p>

            {directError && (
              <div className="alert-box alert-danger">
                <span>⚠️</span><span>{directError}</span>
              </div>
            )}

            <form onSubmit={handleDirectSolve}>
              {/* Question Text */}
              <div className="form-group">
                <label className="form-label" htmlFor="direct-text-input">Your Math Question</label>
                <textarea
                  id="direct-text-input"
                  className="input-text"
                  style={{ minHeight: '120px', resize: 'vertical', fontFamily: 'var(--font-body)', fontSize: '1.05rem', lineHeight: '1.6' }}
                  placeholder="e.g. Solve x^2 - 5x + 6 = 0 or Find the derivative of x^3 + 2x"
                  value={directText}
                  onChange={e => setDirectText(e.target.value)}
                />
              </div>

              {/* Optional Image Upload */}
              <div className="form-group">
                <label className="form-label">Attach Image/Photo (Optional)</label>
                <div
                  className={`file-dropzone ${directDrag ? 'drag-active' : ''}`}
                  onDragEnter={onDirectDrag} onDragOver={onDirectDrag}
                  onDragLeave={onDirectDrag} onDrop={onDirectDrop}
                  onClick={() => directInputRef.current?.click()}
                >
                  <input ref={directInputRef} type="file" accept="image/*"
                    style={{ display: 'none' }} onChange={onDirectFileChange} />
                  <span className="upload-icon">📸</span>
                  <p>Drag &amp; drop a math image here</p>
                  <p>or click to browse</p>
                  {directFile && (
                    <span className="file-name-preview">
                      📎&nbsp;{directFile.name}&nbsp;({(directFile.size / 1024 / 1024).toFixed(2)} MB)
                    </span>
                  )}
                </div>
              </div>

              <div style={{ display: 'flex', gap: '0.75rem', marginTop: '1.5rem' }}>
                <button className="btn-submit" type="submit" disabled={directLoading} style={{ flex: 2, margin: 0 }}>
                  {directLoading ? '⏳ Solving Question…' : '✨ Solve Math Problem'}
                </button>
                {(directText || directFile || directResult) && (
                  <button
                    className="btn-secondary"
                    type="button"
                    style={{ flex: 1, margin: 0 }}
                    onClick={handleDirectReset}
                  >
                    Clear
                  </button>
                )}
              </div>
            </form>
          </div>

          {/* Right Column: Chalkboard Solution Board */}
          <div className="glass-panel" style={{ minHeight: '350px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem' }}>
              <h2 style={{ fontFamily: 'var(--font-ui)' }}>Solution Blackboard</h2>
              {directResult && (
                <button
                  className={`btn-speak ${speakingId === 'direct' ? 'speaking' : ''}`}
                  onClick={() => handleSpeak('direct', `${directText}. The answer is: ${getSpeakableAnswer(directResult.final_answer)}. Here are the steps: ${directResult.solution_steps}`)}
                >
                  {speakingId === 'direct' ? '⏹ Stop Speaking' : '🔊 Speak Solution'}
                </button>
              )}
            </div>

            {directLoading && (
              <div className="loader-container" style={{ minHeight: '200px' }}>
                <div className="spinner-glow" style={{ width: '48px', height: '48px' }} />
                <p style={{ color: 'var(--text-secondary)', marginTop: '1rem' }}>
                  {apiKey ? 'Gemini AI is analyzing the question...' : 'SymPy is computing the solution...'}
                </p>
              </div>
            )}

            {!directLoading && !directResult && (
              <div className="placeholder-empty" style={{ minHeight: '200px', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                <div className="placeholder-empty-icon" style={{ fontSize: '2.5rem' }}>✍️</div>
                <h3>Blackboard Idle</h3>
                <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', marginTop: '0.5rem' }}>
                  Ask a question on the left and see the step-by-step math proof rendered here.
                </p>
              </div>
            )}

            {directResult && (
              <div className="solution-panel" style={{ marginTop: 0, borderLeft: '3px solid var(--gold)', animation: 'slideDown 0.3s ease-out' }}>
                <div className="solution-meta" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem', flexWrap: 'wrap', gap: '0.5rem' }}>
                  <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                    <span className="solution-badge-mode">
                      {getSolverBadgeText(directResult.solver_mode, directResult.provider_model)}
                    </span>
                    <span className="solution-badge-type">{directResult.problem_type}</span>
                    {directResult.difficulty_level && (
                      <span className={`solution-badge-difficulty difficulty-${(directResult.difficulty_level || '').toLowerCase()}`}>
                        {directResult.difficulty_level === 'Easy' && '🟢'}
                        {directResult.difficulty_level === 'Medium' && '🟡'}
                        {directResult.difficulty_level === 'Hard' && '🟠'}
                        {directResult.difficulty_level === 'Advanced' && '🔴'}
                        {' '}{directResult.difficulty_level}
                      </span>
                    )}
                  </div>
                </div>

                {directResult.theory_explanation && (
                  <div className="solution-theory-section">
                    <div className="solution-section-label">📖 Concept & Theory</div>
                    <div className="solution-theory-content">
                      <MathRenderer text={directResult.theory_explanation} />
                    </div>
                  </div>
                )}

                {directResult.formulas_used && (
                  <div className="solution-formulas-section">
                    <div className="solution-section-label">📐 Formulas Used</div>
                    <div className="solution-formulas-content">
                      <MathRenderer text={directResult.formulas_used} />
                    </div>
                  </div>
                )}

                <div className="solution-steps-section">
                  <div className="solution-section-label">📝 Step-by-Step Solution</div>
                  <div className="solution-steps-content">
                    <MathRenderer text={directResult.solution_steps} />
                  </div>
                </div>

                {directResult.reasoning && (
                  <details className="reasoning-collapse" style={{ marginTop: '1rem', borderTop: '1px dashed var(--border)', paddingTop: '0.75rem' }}>
                    <summary style={{ cursor: 'pointer', fontSize: '0.85rem', color: 'var(--teal)', fontWeight: 600, userSelect: 'none' }}>🧠 AI Reasoning Path</summary>
                    <div style={{ marginTop: '0.5rem', fontSize: '0.9rem', color: 'var(--text-secondary)', lineHeight: 1.7 }}>
                      <MathRenderer text={directResult.reasoning} />
                    </div>
                  </details>
                )}

                <div className="solution-final-answer-box">
                  <div className="solution-answer-label">
                    <strong>✅ Final Answer</strong>
                  </div>
                  <div className="final-answer-val">
                    <MathRenderer text={formatFinalAnswer(directResult.final_answer)} />
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ══════════ PDF TAB ══════════ */}
      {tab === 'pdf' && (
        <div className="app-grid">
          {/* Form */}
          {!results && !loading && (
            <div className="glass-panel">
              <h2 style={{ marginBottom: '1.75rem', fontFamily: 'var(--font-ui)' }}>Extraction Parameters</h2>

              {error && (
                <div className="alert-box alert-danger">
                  <span>⚠️</span><span>{error}</span>
                </div>
              )}

              <form onSubmit={handleSubmit}>
                {/* Name */}
                <div className="form-group">
                  <label className="form-label" htmlFor="name-input">Full Name</label>
                  <input id="name-input" className="input-text" type="text"
                    placeholder="e.g. Dr. Emily Clarke" value={name}
                    onChange={e => setName(e.target.value)} required />
                </div>

                {/* Num questions */}
                <div className="form-group">
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.4rem' }}>
                    <label className="form-label" htmlFor="num-q-input" style={{ margin: 0 }}>Questions to Extract</label>
                    <label style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', fontSize: '0.85rem', cursor: 'pointer', color: 'var(--teal)', userSelect: 'none' }}>
                      <input
                        type="checkbox"
                        checked={numQs === -1}
                        onChange={e => setNumQs(e.target.checked ? -1 : 5)}
                      />
                      Extract All Questions
                    </label>
                  </div>
                  {numQs !== -1 && (
                    <input id="num-q-input" className="input-text" type="number"
                      min={1} max={500} value={numQs}
                      onChange={e => setNumQs(parseInt(e.target.value) || 1)} required />
                  )}
                  {numQs === -1 && (
                    <div style={{ padding: '0.4rem 0.75rem', background: 'rgba(20,184,166,0.08)', borderRadius: 'var(--radius-sm)', border: '1px solid rgba(20,184,166,0.2)', color: 'var(--teal)', fontSize: '0.85rem', fontWeight: 500 }}>
                      ♾️ All detected questions in the PDF will be extracted.
                    </div>
                  )}
                </div>


                {/* Format */}
                <div className="form-group">
                  <label className="form-label">Output Format</label>
                  <div className="radio-group">
                    {(['text', 'image'] as const).map(f => (
                      <div key={f} className={`radio-card ${fmt === f ? 'active' : ''}`} onClick={() => setFmt(f)}>
                        <input type="radio" name="format" value={f} readOnly checked={fmt === f} />
                        <span className="radio-icon">{f === 'text' ? '📝' : '🖼️'}</span>
                        <span>{f === 'text' ? 'Structured Text' : 'Cropped Image'}</span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Free-mode notice */}
                <div className="alert-box alert-info" style={{ marginBottom: '1rem' }}>
                  <span>✅</span>
                  <span>
                    <strong>Free mode active.</strong> No API key needed — questions are detected using built-in PDF text analysis.
                    Enter a Gemini key in the top settings bar for AI-powered vision extraction.
                  </span>
                </div>

                {/* PDF Upload */}
                <div className="form-group" style={{ marginTop: '1.25rem' }}>
                  <label className="form-label">Upload Exam Paper (PDF)</label>
                  <div
                    className={`file-dropzone ${drag ? 'drag-active' : ''}`}
                    onDragEnter={onDrag} onDragOver={onDrag}
                    onDragLeave={onDrag} onDrop={onDrop}
                    onClick={() => pdfRef.current?.click()}
                  >
                    <input ref={pdfRef} type="file" accept=".pdf"
                       style={{ display: 'none' }} onChange={onFileChange} />
                    <span className="upload-icon">📂</span>
                    <p>Drag &amp; drop your Mathematics PDF here</p>
                    <p>or click to browse</p>
                    {pdfFile && (
                      <span className="file-name-preview">
                        📎&nbsp;{pdfFile.name}&nbsp;({(pdfFile.size / 1024 / 1024).toFixed(2)} MB)
                      </span>
                    )}
                  </div>
                </div>

                <button className="btn-submit" type="submit" disabled={!pdfFile}>
                  ✦ Run Extraction Pipeline
                </button>
              </form>
            </div>
          )}

          {/* Loading */}
          {loading && (
            <div className="glass-panel" style={{ gridColumn: '1 / -1', minHeight: 480 }}>
              <div className="loader-container">
                <div className="spinner-glow" />
                <h2 style={{ marginBottom: '0.5rem' }}>Pipeline Running</h2>
                <p style={{ color: 'var(--text-secondary)', marginBottom: '2rem', maxWidth: 440 }}>
                  {apiKey ? 'Using Gemini AI Vision…' : 'Using free local extraction (PyMuPDF)…'}
                </p>
                <div className="loading-steps">
                  {STEPS.map((s, i) => (
                    <div key={i} className={`loading-step-item ${step === i ? 'active' : step > i ? 'completed' : ''}`}>
                      <div className="step-indicator-dot" />
                      <span>{s}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* Results */}
          {results && (
            <div className="glass-panel" style={{ gridColumn: '1 / -1' }}>
              <div className="results-header">
                <div>
                  <h2>Extracted Questions</h2>
                  <p className="results-meta">
                    For&nbsp;<strong style={{ color: 'var(--teal)' }}>{results.user_name || 'Guest'}</strong>
                    &ensp;·&ensp;{results.questions_count} question{results.questions_count !== 1 ? 's' : ''}
                    &ensp;·&ensp;
                    <span style={{ color: 'var(--gold)' }}>
                      {results.extraction_mode === 'gemini_ai' ? '🤖 Gemini AI' : '🔓 Free Local'}
                    </span>
                  </p>
                </div>
                <div className="results-actions">
                  <button className="btn-secondary" onClick={downloadJson}>📥 Export JSON</button>
                  <button className="btn-submit" style={{ width: 'auto', margin: 0, padding: '0.5rem 1.25rem', fontSize: '0.85rem' }} onClick={reset}>
                    🔄 New Extraction
                  </button>
                </div>
              </div>

              {/* Batch Solve Bar */}
              {results.questions.length > 0 && (
                <div className="batch-solve-bar" style={{ display: 'flex', gap: '0.75rem', alignItems: 'center', padding: '0.75rem 1rem', background: 'rgba(20,184,166,0.06)', borderRadius: 'var(--radius-sm)', border: '1px solid rgba(20,184,166,0.15)', marginBottom: '1.5rem', flexWrap: 'wrap' }}>
                  <span style={{ fontWeight: 600, fontSize: '0.9rem', color: 'var(--teal)', marginRight: 'auto' }}>
                    ☑ {selectedQuestions.size} of {results.questions.length} selected
                  </span>
                  <button
                    className="btn-secondary"
                    style={{ width: 'auto', margin: 0, padding: '0.3rem 0.8rem', fontSize: '0.8rem' }}
                    onClick={selectAllQuestions}
                  >
                    Select All
                  </button>
                  <button
                    className="btn-secondary"
                    style={{ width: 'auto', margin: 0, padding: '0.3rem 0.8rem', fontSize: '0.8rem' }}
                    onClick={deselectAllQuestions}
                  >
                    Deselect All
                  </button>
                  <button
                    className="btn-submit"
                    style={{ width: 'auto', margin: 0, padding: '0.4rem 1rem', fontSize: '0.85rem' }}
                    disabled={selectedQuestions.size === 0 || batchSolving}
                    onClick={handleSolveSelected}
                  >
                    {batchSolving ? `⏳ Solving ${selectedQuestions.size} questions…` : `✨ Solve Selected (${selectedQuestions.size})`}
                  </button>
                </div>
              )}

              {results.questions.length === 0 ? (
                <div className="placeholder-empty">
                  <div className="placeholder-empty-icon">🔍</div>
                  <h3>No Questions Detected</h3>
                  <p style={{ marginTop: '0.5rem' }}>
                    Ensure the PDF has numbered questions (e.g. "1.", "Q1", "Question 1").<br />
                    Try providing a Gemini API key in the top settings bar for AI-powered detection.
                  </p>
                </div>
              ) : results.output_format === 'image' ? (
                <div className="images-grid">
                  {results.questions.map(q => {
                    const sol = solutions[q.id];
                    const res = sol?.result;
                    return (
                      <div key={q.id} className="glass-card cropped-image-card" style={{ height: 'auto', display: 'flex', flexDirection: 'column', outline: selectedQuestions.has(q.id) ? '2px solid var(--teal)' : 'none' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.25rem 0.25rem 0' }}>
                          <input
                            type="checkbox"
                            checked={selectedQuestions.has(q.id)}
                            onChange={() => toggleQuestionSelection(q.id)}
                            style={{ width: '18px', height: '18px', accentColor: 'var(--teal)', cursor: 'pointer' }}
                          />
                        </div>
                        <div className="cropped-image-wrapper">
                          {q.image_url
                            ? <img src={`${API_BASE}${q.image_url}`} alt={`Q${q.question_number}`} className="cropped-image-file" />
                            : <span style={{ color: '#999', fontSize: '0.85rem' }}>Crop unavailable</span>}
                        </div>
                        <div className="cropped-image-title-bar" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '0.75rem', padding: '0 0.25rem' }}>
                          <div className="cropped-image-title">Question {q.question_number}</div>
                          <button
                            className="btn-solve-mini"
                            disabled={sol?.loading}
                            onClick={() => handleSolve(q.id, undefined, q.image_url)}
                            style={{ fontSize: '0.75rem', padding: '0.25rem 0.6rem', height: 'auto', width: 'auto', display: 'inline-flex' }}
                          >
                            {sol?.loading ? '⏳ Solving...' : '✨ Solve'}
                          </button>
                        </div>
                        {/* Solution rendering */}
                        {sol && (
                          <div className="solution-panel" style={{ marginTop: '0.75rem', width: '100%', fontSize: '0.88rem', borderTop: '1px solid var(--border-hover)', paddingTop: '0.75rem' }}>
                            {sol.loading && <div className="solution-loading">⏳ Solving...</div>}
                            {sol.error && <div className="solution-error">⚠️ {sol.error}</div>}
                            {res && (
                              <div className="solution-body">
                                <div className="solution-meta" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', width: '100%', marginBottom: '0.5rem', flexWrap: 'wrap', gap: '0.25rem' }}>
                                  <div style={{ display: 'flex', gap: '0.25rem', flexWrap: 'wrap' }}>
                                    <span className="solution-badge-mode" style={{ fontSize: '0.7rem', padding: '0.1rem 0.3rem' }}>
                                      {getSolverBadgeText(res.solver_mode, res.provider_model)}
                                    </span>
                                    <span className="solution-badge-type" style={{ fontSize: '0.7rem', padding: '0.1rem 0.3rem' }}>{res.problem_type}</span>
                                    {res.difficulty_level && (
                                      <span className={`solution-badge-difficulty difficulty-${(res.difficulty_level || '').toLowerCase()}`} style={{ fontSize: '0.7rem', padding: '0.1rem 0.3rem' }}>
                                        {res.difficulty_level === 'Easy' && '🟢'}
                                        {res.difficulty_level === 'Medium' && '🟡'}
                                        {res.difficulty_level === 'Hard' && '🟠'}
                                        {res.difficulty_level === 'Advanced' && '🔴'}
                                        {' '}{res.difficulty_level}
                                      </span>
                                    )}
                                  </div>
                                  <button
                                    className={`btn-speak ${speakingId === q.id ? 'speaking' : ''}`}
                                    onClick={() => handleSpeak(q.id, `Question. The answer is: ${getSpeakableAnswer(res.final_answer)}. Here are the steps: ${res.solution_steps}`)}
                                    style={{ padding: '0.15rem 0.4rem', fontSize: '0.7rem' }}
                                  >
                                    {speakingId === q.id ? '⏹ Stop' : '🔊 Speak'}
                                  </button>
                                </div>

                                {res.theory_explanation && (
                                  <div className="solution-theory-section" style={{ fontSize: '0.85rem', marginBottom: '0.5rem' }}>
                                    <div className="solution-section-label" style={{ fontSize: '0.75rem', padding: '0.15rem 0.4rem' }}>📖 Concept & Theory</div>
                                    <div className="solution-theory-content">
                                      <MathRenderer text={res.theory_explanation} />
                                    </div>
                                  </div>
                                )}

                                {res.formulas_used && (
                                  <div className="solution-formulas-section" style={{ fontSize: '0.85rem', marginBottom: '0.5rem' }}>
                                    <div className="solution-section-label" style={{ fontSize: '0.75rem', padding: '0.15rem 0.4rem' }}>📐 Formulas Used</div>
                                    <div className="solution-formulas-content">
                                      <MathRenderer text={res.formulas_used} />
                                    </div>
                                  </div>
                                )}

                                <div className="solution-steps-content">
                                  <MathRenderer text={res.solution_steps} />
                                </div>
                                {res.reasoning && (
                                  <details className="reasoning-collapse" style={{ marginTop: '0.5rem', borderTop: '1px dashed var(--border)', paddingTop: '0.4rem' }}>
                                    <summary style={{ cursor: 'pointer', fontSize: '0.75rem', color: 'var(--teal)', fontWeight: 600, userSelect: 'none' }}>🧠 AI Reasoning Path</summary>
                                    <div style={{ marginTop: '0.4rem', fontSize: '0.78rem', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
                                      <MathRenderer text={res.reasoning} />
                                    </div>
                                  </details>
                                )}
                                <div className="solution-final-answer-box" style={{ padding: '0.4rem', marginTop: '0.5rem', fontSize: '0.82rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                                  <strong>Answer:</strong>
                                  <span className="final-answer-val" style={{ fontSize: '0.88rem' }}>
                                    <MathRenderer text={formatFinalAnswer(res.final_answer)} />
                                  </span>
                                </div>
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              ) : (
                <div className="questions-list">
                  {results.questions.map(q => {
                    const sol = solutions[q.id];
                    const res = sol?.result;
                    return (
                      <div key={q.id} className={`glass-card question-item-card ${q.has_diagram ? 'has-diagram' : ''}`} style={{ display: 'flex', flexDirection: 'column', gap: '1rem', outline: selectedQuestions.has(q.id) ? '2px solid var(--teal)' : 'none' }}>
                        <div style={{ display: 'flex', gap: '1.25rem', width: '100%' }}>
                          <div style={{ display: 'flex', alignItems: 'flex-start', paddingTop: '0.2rem' }}>
                            <input
                              type="checkbox"
                              checked={selectedQuestions.has(q.id)}
                              onChange={() => toggleQuestionSelection(q.id)}
                              style={{ width: '20px', height: '20px', accentColor: 'var(--teal)', cursor: 'pointer', flexShrink: 0 }}
                            />
                          </div>
                          {q.image_url && (
                            <div className="question-crop-preview">
                              <img src={`${API_BASE}${q.image_url}`} alt={`Q${q.question_number} crop`} className="question-crop-img" />
                            </div>
                          )}
                          <div style={{ flex: 1 }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
                              <span className="question-number-badge">Question {q.question_number}</span>
                              <button
                                className="btn-solve-mini"
                                disabled={sol?.loading}
                                onClick={() => handleSolve(q.id, q.text, q.image_url)}
                              >
                                {sol?.loading ? '⏳ Solving...' : '✨ Solve Question'}
                              </button>
                            </div>
                            <div className="question-text-content">
                              <MathRenderer text={q.text || ''} />
                            </div>
                          </div>
                          {q.has_diagram && q.diagram_url && (
                            <div className="question-diagram-wrapper">
                              <img src={`${API_BASE}${q.diagram_url}`} alt={`Diagram Q${q.question_number}`} className="question-diagram-image" />
                            </div>
                          )}
                        </div>

                        {/* Solution rendering */}
                        {sol && (
                          <div className="solution-panel" style={{ borderTop: '1px solid var(--border-hover)', paddingTop: '1rem', marginTop: '0.5rem' }}>
                            {sol.loading && (
                              <div className="solution-loading">
                                <span className="spinner-mini"></span> Analyzing question &amp; computing solution...
                              </div>
                            )}
                            {sol.error && (
                              <div className="solution-error">
                                ⚠️ {sol.error}
                              </div>
                            )}
                            {res && (
                              <div className="solution-body">
                                <div className="solution-meta" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', width: '100%', marginBottom: '0.75rem', flexWrap: 'wrap', gap: '0.5rem' }}>
                                  <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                                    <span className="solution-badge-mode">
                                      {getSolverBadgeText(res.solver_mode, res.provider_model)}
                                    </span>
                                    <span className="solution-badge-type">{res.problem_type}</span>
                                    {res.difficulty_level && (
                                      <span className={`solution-badge-difficulty difficulty-${(res.difficulty_level || '').toLowerCase()}`}>
                                        {res.difficulty_level === 'Easy' && '🟢'}
                                        {res.difficulty_level === 'Medium' && '🟡'}
                                        {res.difficulty_level === 'Hard' && '🟠'}
                                        {res.difficulty_level === 'Advanced' && '🔴'}
                                        {' '}{res.difficulty_level}
                                      </span>
                                    )}
                                  </div>
                                  <button
                                    className={`btn-speak ${speakingId === q.id ? 'speaking' : ''}`}
                                    onClick={() => handleSpeak(q.id, `${q.text || 'Question'}. The answer is: ${getSpeakableAnswer(res.final_answer)}. Here are the steps: ${res.solution_steps}`)}
                                    style={{ padding: '0.2rem 0.5rem', fontSize: '0.75rem' }}
                                  >
                                    {speakingId === q.id ? '⏹ Stop' : '🔊 Speak'}
                                  </button>
                                </div>

                                {res.theory_explanation && (
                                  <div className="solution-theory-section">
                                    <div className="solution-section-label">📖 Concept & Theory</div>
                                    <div className="solution-theory-content">
                                      <MathRenderer text={res.theory_explanation} />
                                    </div>
                                  </div>
                                )}

                                {res.formulas_used && (
                                  <div className="solution-formulas-section">
                                    <div className="solution-section-label">📐 Formulas Used</div>
                                    <div className="solution-formulas-content">
                                      <MathRenderer text={res.formulas_used} />
                                    </div>
                                  </div>
                                )}

                                <div className="solution-steps-section">
                                  <div className="solution-section-label">📝 Step-by-Step Solution</div>
                                  <div className="solution-steps-content">
                                    <MathRenderer text={res.solution_steps} />
                                  </div>
                                </div>

                                {res.reasoning && (
                                  <details className="reasoning-collapse" style={{ marginTop: '0.75rem', borderTop: '1px dashed var(--border)', paddingTop: '0.5rem' }}>
                                    <summary style={{ cursor: 'pointer', fontSize: '0.8rem', color: 'var(--teal)', fontWeight: 600, userSelect: 'none' }}>🧠 AI Reasoning Path</summary>
                                    <div style={{ marginTop: '0.5rem', fontSize: '0.85rem', color: 'var(--text-secondary)', lineHeight: 1.7 }}>
                                      <MathRenderer text={res.reasoning} />
                                    </div>
                                  </details>
                                )}
                                <div className="solution-final-answer-box">
                                  <strong>Final Answer:</strong>
                                  <span className="final-answer-val">
                                    <MathRenderer text={formatFinalAnswer(res.final_answer)} />
                                  </span>
                                </div>
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          )}

          {/* Info panel */}
          {!results && !loading && (
            <div className="glass-panel" style={{ height: 'max-content' }}>
              <p className="info-panel-title">⬡ How It Works</p>
              <ol className="workflow-list">
                <li>Enter your name and how many questions to extract.</li>
                <li>Choose <strong style={{ color: 'var(--text-chalk)' }}>Cropped Image</strong> or <strong style={{ color: 'var(--text-chalk)' }}>Structured Text</strong> output.</li>
                <li>Upload any numbered math exam PDF.</li>
                <li>The extractor compiles a list of detected questions using local text layers.</li>
                <li>Solve individual questions offline using SymPy/Tutor fallback or online via Gemini.</li>
              </ol>
              <div className="alert-box alert-info" style={{ marginTop: '1.5rem' }}>
                <strong style={{ color: 'var(--text-chalk)', display: 'block', marginBottom: '0.4rem' }}>ƒ Math Rendering</strong>
                <span style={{ color: 'var(--text-secondary)', fontSize: '0.88rem' }}>
                  Inline and display LaTeX are both supported:
                </span>
                <code className="latex-example">
                  {'$x^2 - 5x + 6 = 0$         (inline)\n$$\\int_0^{\\pi} \\sin x\\,dx$$  (display)'}
                </code>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ══════════ OCR TAB ══════════ */}
      {tab === 'ocr' && (
        <div className="app-grid">
          <div className="glass-panel">
            <h2 style={{ marginBottom: '1.5rem', fontFamily: 'var(--font-ui)' }}>Screenshot / Photo OCR</h2>
            <p style={{ color: 'var(--text-secondary)', marginBottom: '1.5rem', lineHeight: 1.7 }}>
              Upload any screenshot, photo, or scanned image to extract its text content.
              Works free for images with a selectable text layer. Provide a Gemini API key for AI-powered OCR of handwriting or raster images.
            </p>

            {ocrError && (
              <div className="alert-box alert-danger">
                <span>⚠️</span><span>{ocrError}</span>
              </div>
            )}

            <form onSubmit={handleOcr}>
              {/* Image Upload */}
              <div className="form-group">
                <label className="form-label">Upload Image (PNG, JPG, WEBP, BMP…)</label>
                <div
                  className={`file-dropzone ${ocrDrag ? 'drag-active' : ''}`}
                  onDragEnter={onOcrDrag} onDragOver={onOcrDrag}
                  onDragLeave={onOcrDrag} onDrop={onOcrDrop}
                  onClick={() => ocrRef.current?.click()}
                >
                  <input ref={ocrRef} type="file" accept="image/*"
                    style={{ display: 'none' }} onChange={onOcrFileChange} />
                  <span className="upload-icon">🖼️</span>
                  <p>Drag &amp; drop your screenshot or photo here</p>
                  <p>or click to browse</p>
                  {ocrFile && (
                    <span className="file-name-preview">
                      📎&nbsp;{ocrFile.name}&nbsp;({(ocrFile.size / 1024 / 1024).toFixed(2)} MB)
                    </span>
                  )}
                </div>
              </div>

              <button className="btn-submit" type="submit" disabled={!ocrFile || ocrLoading}>
                {ocrLoading ? '⏳ Extracting…' : '🔍 Extract Text'}
              </button>
            </form>
          </div>

          {/* OCR Result */}
          <div className="glass-panel" style={{ height: 'max-content' }}>
            {!ocrResult && !ocrLoading && (
              <>
                <p className="info-panel-title">📋 OCR Result</p>
                <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', marginTop: '0.75rem' }}>
                  Extracted text will appear here after you upload and process an image.
                </p>
                <div className="alert-box alert-info" style={{ marginTop: '1.5rem' }}>
                  <strong style={{ color: 'var(--text-chalk)', display: 'block', marginBottom: '0.4rem' }}>💡 Tips</strong>
                  <ul style={{ paddingLeft: '1.2rem', color: 'var(--text-secondary)', fontSize: '0.88rem', lineHeight: 1.8 }}>
                    <li>PDFs exported to PNG work best without a Gemini key</li>
                    <li>For handwritten or scanned photos, use a Gemini API key</li>
                    <li>Math formulas are rendered in LaTeX when Gemini AI is used</li>
                  </ul>
                </div>
              </>
            )}

            {ocrLoading && (
              <div className="loader-container" style={{ minHeight: 200 }}>
                <div className="spinner-glow" style={{ width: 48, height: 48 }} />
                <p style={{ color: 'var(--text-secondary)', marginTop: '1rem' }}>
                  {apiKey ? 'Gemini AI is reading the image…' : 'Extracting text layer…'}
                </p>
              </div>
            )}

            {ocrResult && (
              <div>
                <div className="results-header" style={{ marginBottom: '1rem' }}>
                  <div>
                    <h3 style={{ fontFamily: 'var(--font-ui)' }}>Extracted Text</h3>
                    <p className="results-meta">
                      {ocrResult.char_count} chars
                      &ensp;·&ensp;
                      <span style={{ color: 'var(--gold)' }}>
                        {ocrResult.extraction_mode === 'gemini_ai' ? '🤖 Gemini AI' : '🔓 Free Local'}
                      </span>
                    </p>
                  </div>
                  <button className="btn-secondary" onClick={copyOcr}>📋 Copy</button>
                </div>

                {ocrResult.image_url && (
                  <div className="question-crop-preview" style={{ maxWidth: '240px', marginBottom: '1rem' }}>
                    <img src={`${API_BASE}${ocrResult.image_url}`} alt="OCR crop preview" className="question-crop-img" />
                  </div>
                )}

                <div className="ocr-result-box">
                  <MathRenderer text={ocrResult.text} />
                </div>
                <div style={{ display: 'flex', gap: '0.75rem', marginTop: '1rem' }}>
                  <button
                    className="btn-submit"
                    style={{ flex: 1, margin: 0 }}
                    disabled={solutions['ocr-solver']?.loading}
                    onClick={() => handleSolve('ocr-solver', ocrResult.text, ocrResult.image_url)}
                  >
                    {solutions['ocr-solver']?.loading ? '⏳ Solving...' : '✨ Solve Extracted Text'}
                  </button>
                  <button
                    className="btn-secondary"
                    style={{ flex: 1, margin: 0 }}
                    onClick={() => { setOcrResult(null); setOcrFile(null); setSolutions(prev => { const n = {...prev}; delete n['ocr-solver']; return n; }); }}
                  >
                    🔄 Extract Another
                  </button>
                </div>

                {/* Solution rendering */}
                {(() => {
                  const sol = solutions['ocr-solver'];
                  const res = sol?.result;
                  return sol && (
                    <div className="solution-panel" style={{ marginTop: '1.5rem', borderTop: '1px solid var(--border-hover)', paddingTop: '1.25rem' }}>
                      {sol.loading && (
                        <div className="solution-loading">
                          <span className="spinner-mini"></span> Analyzing question &amp; computing solution...
                        </div>
                      )}
                      {sol.error && (
                        <div className="solution-error">
                          ⚠️ {sol.error}
                        </div>
                      )}
                      {res && (
                        <div className="solution-body">
                          <div className="solution-meta" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem', flexWrap: 'wrap', gap: '0.5rem' }}>
                            <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                              <span className="solution-badge-mode">
                                {getSolverBadgeText(res.solver_mode, res.provider_model)}
                              </span>
                              <span className="solution-badge-type">{res.problem_type}</span>
                              {res.difficulty_level && (
                                <span className={`solution-badge-difficulty difficulty-${(res.difficulty_level || '').toLowerCase()}`}>
                                  {res.difficulty_level === 'Easy' && '🟢'}
                                  {res.difficulty_level === 'Medium' && '🟡'}
                                  {res.difficulty_level === 'Hard' && '🟠'}
                                  {res.difficulty_level === 'Advanced' && '🔴'}
                                  {' '}{res.difficulty_level}
                                </span>
                              )}
                            </div>
                            <button
                              className={`btn-speak ${speakingId === 'ocr-solver' ? 'speaking' : ''}`}
                              onClick={() => handleSpeak('ocr-solver', `${ocrResult?.text || 'Question'}. The answer is: ${getSpeakableAnswer(res.final_answer)}. Here are the steps: ${res.solution_steps}`)}
                              style={{ padding: '0.2rem 0.5rem', fontSize: '0.75rem' }}
                            >
                              {speakingId === 'ocr-solver' ? '⏹ Stop' : '🔊 Speak'}
                            </button>
                          </div>

                          {res.theory_explanation && (
                            <div className="solution-theory-section">
                              <div className="solution-section-label">📖 Concept & Theory</div>
                              <div className="solution-theory-content">
                                <MathRenderer text={res.theory_explanation} />
                              </div>
                            </div>
                          )}

                          {res.formulas_used && (
                            <div className="solution-formulas-section">
                              <div className="solution-section-label">📐 Formulas Used</div>
                              <div className="solution-formulas-content">
                                <MathRenderer text={res.formulas_used} />
                              </div>
                            </div>
                          )}

                          <div className="solution-steps-section">
                            <div className="solution-section-label">📝 Step-by-Step Solution</div>
                            <div className="solution-steps-content">
                              <MathRenderer text={res.solution_steps} />
                            </div>
                          </div>

                          {res.reasoning && (
                            <details className="reasoning-collapse" style={{ marginTop: '0.75rem', borderTop: '1px dashed var(--border)', paddingTop: '0.5rem' }}>
                              <summary style={{ cursor: 'pointer', fontSize: '0.8rem', color: 'var(--teal)', fontWeight: 600, userSelect: 'none' }}>🧠 AI Reasoning Path</summary>
                              <div style={{ marginTop: '0.5rem', fontSize: '0.85rem', color: 'var(--text-secondary)', lineHeight: 1.7 }}>
                                <MathRenderer text={res.reasoning} />
                              </div>
                            </details>
                          )}
                          <div className="solution-final-answer-box">
                            <strong>Final Answer:</strong>
                            <span className="final-answer-val">
                              <MathRenderer text={formatFinalAnswer(res.final_answer)} />
                            </span>
                          </div>
                        </div>
                      )}
                    </div>
                  );
                })()}
              </div>
            )}
          </div>
        </div>
      )}

      {/* ══════════ HISTORY TAB ══════════ */}
      {tab === 'history' && (
        <div className="glass-panel" style={{ minHeight: '400px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
            <div>
              <h2 style={{ fontFamily: 'var(--font-ui)' }}>Saved Solutions History</h2>
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.92rem' }}>
                Review past questions you have solved. All questions are saved locally on this browser.
              </p>
            </div>
            {history.length > 0 && (
              <button className="btn-secondary" style={{ color: 'var(--red)', borderColor: 'rgba(191,97,106,0.3)', width: 'auto', margin: 0 }} onClick={clearHistory}>
                🗑 Clear All History
              </button>
            )}
          </div>

          {history.length === 0 ? (
            <div className="placeholder-empty">
              <div className="placeholder-empty-icon" style={{ fontSize: '3rem' }}>📜</div>
              <h3>No History Found</h3>
              <p style={{ marginTop: '0.5rem' }}>
                Solve some math problems under the "Ask a Question", "PDF Extractor", or "Screenshot OCR" tabs to see them here.
              </p>
            </div>
          ) : (
            <div className="history-container">
              {history.map(item => (
                <div key={item.id} className="glass-card history-item" style={{ cursor: 'default' }}>
                  <div className="history-item-header">
                    <span style={{ fontWeight: 600, color: 'var(--teal)' }}>{item.result.problem_type}</span>
                    <span>{new Date(item.timestamp).toLocaleString()}</span>
                  </div>

                  <div className="question-text-content" style={{ fontSize: '1rem', marginBottom: '1rem', borderBottom: '1px dashed var(--border)', paddingBottom: '0.75rem' }}>
                    <strong>Question:</strong>
                    <div style={{ marginTop: '0.25rem' }}>
                      <MathRenderer text={item.question} />
                    </div>
                    {item.imageUrl && (
                      <div className="question-crop-preview" style={{ marginTop: '0.5rem', maxWidth: '240px' }}>
                        <img src={`${API_BASE}${item.imageUrl}`} alt="Original uploaded file" className="question-crop-img" />
                      </div>
                    )}
                  </div>

                  <div className="solution-panel" style={{ borderLeft: '3px solid var(--gold)', background: 'rgba(0,0,0,0.15)', marginTop: 0 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem', flexWrap: 'wrap', gap: '0.5rem' }}>
                      <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                        <span className="solution-badge-mode" style={{ fontSize: '0.7rem' }}>
                          {getSolverBadgeText(item.result.solver_mode, item.result.provider_model)}
                        </span>
                        {item.result.difficulty_level && (
                          <span className={`solution-badge-difficulty difficulty-${(item.result.difficulty_level || '').toLowerCase()}`} style={{ fontSize: '0.7rem', padding: '0.1rem 0.3rem' }}>
                            {item.result.difficulty_level === 'Easy' && '🟢'}
                            {item.result.difficulty_level === 'Medium' && '🟡'}
                            {item.result.difficulty_level === 'Hard' && '🟠'}
                            {item.result.difficulty_level === 'Advanced' && '🔴'}
                            {' '}{item.result.difficulty_level}
                          </span>
                        )}
                      </div>
                      <button
                        className={`btn-speak ${speakingId === item.id ? 'speaking' : ''}`}
                        onClick={() => handleSpeak(item.id, `${item.question}. The answer is: ${getSpeakableAnswer(item.result.final_answer)}. Here are the steps: ${item.result.solution_steps}`)}
                      >
                        {speakingId === item.id ? '⏹ Stop' : '🔊 Speak'}
                      </button>
                    </div>

                    {item.result.theory_explanation && (
                      <div className="solution-theory-section" style={{ fontSize: '0.88rem', marginBottom: '0.75rem' }}>
                        <div className="solution-section-label" style={{ fontSize: '0.8rem', padding: '0.15rem 0.4rem' }}>📖 Concept & Theory</div>
                        <div className="solution-theory-content">
                          <MathRenderer text={item.result.theory_explanation} />
                        </div>
                      </div>
                    )}

                    {item.result.formulas_used && (
                      <div className="solution-formulas-section" style={{ fontSize: '0.88rem', marginBottom: '0.75rem' }}>
                        <div className="solution-section-label" style={{ fontSize: '0.8rem', padding: '0.15rem 0.4rem' }}>📐 Formulas Used</div>
                        <div className="solution-formulas-content">
                          <MathRenderer text={item.result.formulas_used} />
                        </div>
                      </div>
                    )}

                    <div className="solution-steps-section" style={{ fontSize: '0.92rem' }}>
                      <div className="solution-section-label" style={{ fontSize: '0.8rem', padding: '0.15rem 0.4rem' }}>📝 Step-by-Step Solution</div>
                      <div className="solution-steps-content">
                        <MathRenderer text={item.result.solution_steps} />
                      </div>
                    </div>

                    {item.result.reasoning && (
                      <details className="reasoning-collapse" style={{ marginTop: '0.75rem', borderTop: '1px dashed var(--border)', paddingTop: '0.5rem' }}>
                        <summary style={{ cursor: 'pointer', fontSize: '0.8rem', color: 'var(--teal)', fontWeight: 600, userSelect: 'none' }}>🧠 AI Reasoning Path</summary>
                        <div style={{ marginTop: '0.5rem', fontSize: '0.85rem', color: 'var(--text-secondary)', lineHeight: 1.7 }}>
                          <MathRenderer text={item.result.reasoning} />
                        </div>
                      </details>
                    )}

                    <div className="solution-final-answer-box" style={{ padding: '0.5rem 0.75rem', fontSize: '0.9rem' }}>
                      <strong>Final Answer:</strong>
                      <span className="final-answer-val" style={{ fontSize: '0.95rem' }}>
                        <MathRenderer text={formatFinalAnswer(item.result.final_answer)} />
                      </span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}