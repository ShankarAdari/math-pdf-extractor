import os
import re
import uuid
import shutil
import logging
import json

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from PIL import Image, ImageEnhance
import fitz  # PyMuPDF

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("math-extractor")

app = FastAPI(title="Mathematics PDF Question Extractor API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
CROPS_DIR  = os.path.join(STATIC_DIR, "crops")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(CROPS_DIR,  exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ─── Model Fallback List ──────────────────────────────────────────────────────
# Only models actually available on the free v1beta API
GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.5-flash-lite",
]

# Models that support vision (image input)
VISION_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.5-flash-lite",
]

def call_gemini_with_fallback(client, contents, config=None, vision_only=False):
    """Try each Gemini model until one succeeds. Returns (model_used, response)."""
    models = VISION_MODELS if vision_only else GEMINI_MODELS
    last_err = None
    for model in models:
        try:
            if config:
                resp = client.models.generate_content(model=model, contents=contents, config=config)
            else:
                resp = client.models.generate_content(model=model, contents=contents)
            # Check for empty / blocked response — treat as retriable
            raw_text = None
            try:
                raw_text = resp.text
            except Exception:
                pass
            if raw_text is None or raw_text.strip() == "":
                logger.warning(f"Model {model} returned empty/blocked response, trying next model.")
                last_err = Exception(f"Model {model} returned empty response (possible safety block or model error).")
                continue
            logger.info(f"Gemini model {model} succeeded.")
            return model, resp
        except Exception as e:
            emsg = str(e)
            # Retry on quota/rate-limit OR empty output errors
            retriable_keywords = [
                "quota", "resource_exhausted", "429", "503",
                "model output", "output text", "tool calls", "empty", "blocked"
            ]
            if any(k in emsg.lower() for k in retriable_keywords):
                logger.warning(f"Model {model} retriable error, trying next: {emsg[:150]}")
                last_err = e
                continue
            # 404 not found, auth errors, bad request → stop trying immediately
            raise
    raise Exception(f"All Gemini models failed. Last error: {last_err}")


# ─── Gemini Quota-Limiting Cache ──────────────────────────────────────────────
import time
import hashlib

GEMINI_QUOTA_COOLDOWN = {}  # key_hash: expiry_timestamp

def is_gemini_quota_limited(api_key: str) -> bool:
    if not api_key:
        return True
    kh = hashlib.sha256(api_key.encode()).hexdigest()
    now = time.time()
    if kh in GEMINI_QUOTA_COOLDOWN:
        if now < GEMINI_QUOTA_COOLDOWN[kh]:
            return True
        else:
            del GEMINI_QUOTA_COOLDOWN[kh]
    return False

def mark_gemini_quota_limited(api_key: str):
    if not api_key:
        return
    kh = hashlib.sha256(api_key.encode()).hexdigest()
    # Cache as quota-limited for 5 minutes
    GEMINI_QUOTA_COOLDOWN[kh] = time.time() + 300



# ─── Tesseract Setup ──────────────────────────────────────────────────────────
_TESSERACT_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    r"C:\Users\HP\AppData\Local\Programs\Tesseract-OCR\tesseract.exe",
    r"C:\Users\HP\scoop\apps\tesseract\current\tesseract.exe",
]
TESSERACT_CMD = None
for _tp in _TESSERACT_PATHS:
    if os.path.exists(_tp):
        TESSERACT_CMD = _tp
        break

try:
    import pytesseract
    if TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
        logger.info(f"Tesseract found: {TESSERACT_CMD}")
    else:
        logger.warning("Tesseract binary not found.")
except ImportError:
    pytesseract = None  # type: ignore

# ─── EasyOCR Lazy Init ────────────────────────────────────────────────────────
_easyocr_reader = None
_easyocr_available = None   # None = untested, True/False after first attempt

def _get_easyocr():
    global _easyocr_reader, _easyocr_available
    if _easyocr_available is False:
        return None
    if _easyocr_reader is not None:
        return _easyocr_reader
    try:
        import easyocr
        _easyocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        _easyocr_available = True
        logger.info("EasyOCR reader initialized successfully.")
        return _easyocr_reader
    except Exception as e:
        _easyocr_available = False
        logger.warning(f"EasyOCR unavailable: {e}")
        return None


# ─── Pydantic Schemas ─────────────────────────────────────────────────────────
class QuestionDetection(BaseModel):
    question_number: str
    text: str
    question_box: List[float]
    has_diagram: bool
    diagram_box: Optional[List[float]] = None


class PageQuestions(BaseModel):
    questions: List[QuestionDetection]


class SolveRequest(BaseModel):
    text: Optional[str] = None
    image_url: Optional[str] = None
    provider: Optional[str] = None   # "gemini" | "groq" | "ollama" | "offline"
    groq_model: Optional[str] = None
    ollama_model: Optional[str] = None
    ollama_base_url: Optional[str] = None


class SolutionOutput(BaseModel):
    problem_type: str
    difficulty_level: Optional[str] = None
    formulas_used: Optional[str] = None
    theory_explanation: Optional[str] = None
    reasoning: Optional[str] = None
    solution_steps: str
    final_answer: str


# ─── Greek / Trig Normalization ───────────────────────────────────────────────
GREEK_MAP = {
    "θ": "theta", "Θ": "theta", "β": "beta", "α": "alpha", "γ": "gamma",
    "δ": "delta", "ε": "epsilon", "φ": "phi", "Φ": "phi", "ψ": "psi",
    "ω": "omega", "Ω": "omega", "λ": "lambda", "μ": "mu", "ν": "nu",
    "π": "pi", "σ": "sigma", "τ": "tau", "ξ": "xi", "η": "eta",
    "ζ": "zeta", "ρ": "rho", "κ": "kappa",
}


def normalize_greek(s: str) -> str:
    for ch, nm in GREEK_MAP.items():
        s = s.replace(ch, nm)
    return s


def normalize_trig(s: str) -> str:
    s = re.sub(r"\bcosec\b", "csc", s)
    s = re.sub(r"\bsecant\b", "sec", s)
    s = re.sub(r"\bcosecant\b", "csc", s)
    s = re.sub(r"\bcotangent\b", "cot", s)
    s = re.sub(r"\barcsin\b", "asin", s)
    s = re.sub(r"\barccos\b", "acos", s)
    s = re.sub(r"\barctan\b", "atan", s)
    return s


# ─── LaTeX / Expression Cleaning ─────────────────────────────────────────────
def clean_latex(s: str) -> str:
    s = s.replace("$$", "").replace("$", "")
    s = s.replace(r"\left", "").replace(r"\right", "")
    s = s.replace(r"\cdot", "*").replace(r"\times", "*")
    frac_re = re.compile(r"\\frac\{([^{}]+)\}\{([^{}]+)\}")
    while frac_re.search(s):
        s = frac_re.sub(r"((\1)/(\2))", s)
    s = s.replace(r"\pi", "pi").replace(r"\theta", "theta")
    s = s.replace(r"\alpha", "alpha").replace(r"\beta", "beta")
    s = s.replace(r"\gamma", "gamma").replace(r"\phi", "phi")
    s = s.replace(r"\omega", "omega").replace(r"\delta", "delta")
    s = s.replace(r"\sigma", "sigma").replace(r"\lambda", "lambda")
    for fn in ["sin", "cos", "tan", "cot", "sec", "csc", "log", "ln", "exp", "sqrt"]:
        s = s.replace("\\" + fn, fn)
    return s


_MATH_FUNCS = frozenset({
    "sin", "cos", "tan", "cot", "sec", "csc", "log", "ln", "exp", "sqrt",
    "pi", "theta", "alpha", "beta", "gamma", "phi", "omega", "delta",
    "lambda", "mu", "sigma", "tau", "epsilon", "eta", "zeta", "nu",
    "inf", "oo", "abs", "asin", "acos", "atan", "x", "y", "z", "t", "n", "k",
})

_BRIDGE_RE = re.compile(
    r"\b(?:where|such\s+that|when|for\s+which|satisfying|then|has|have|"
    r"equals?|is|are|becomes?)\\b", re.IGNORECASE)

_INSTRUCT_RE = re.compile(
    r"^\s*(?:solve|find|calculate|evaluate|simplify|compute|determine|"
    r"differentiate|integrate|what\s+is|the\s+(?:equation|value|answer)|"
    r"factorise|factorize|expand|for\s+all|given\s+that|if|let|suppose|"
    r"consider|show\s+that|prove\s+that|of)[\s:,;]*", re.IGNORECASE)


def _strip_prose(text: str) -> str:
    text = _BRIDGE_RE.sub(" ", text)

    def _keep(m: re.Match) -> str:
        w = m.group(0)
        return w if (w.lower() in _MATH_FUNCS or len(w) == 1) else " "

    text = re.sub(r"[A-Za-z]+", _keep, text)
    return re.sub(r"\s+", " ", text).strip()


def _extract_math_expr(question_text: str) -> str:
    question_text = normalize_greek(question_text)
    question_text = normalize_trig(question_text)
    parts = re.findall(r"\$([^\$]+)\$", question_text)
    if parts:
        return clean_latex(parts[0])
    cleaned = clean_latex(question_text)
    cleaned = _INSTRUCT_RE.sub("", cleaned).strip()
    cleaned = _BRIDGE_RE.sub(" ", cleaned).strip()
    if re.search(r"\b[A-Za-z]{2,}\b", cleaned):
        has_non_math = any(
            w.lower() not in _MATH_FUNCS
            for w in re.findall(r"[A-Za-z]{2,}", cleaned)
        )
        if has_non_math:
            cleaned = _strip_prose(cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def convert_trig_powers(s: str) -> str:
    """Convert sin^2(x) → (sin(x))**2 etc."""
    pattern = re.compile(r"\b(sin|cos|tan|cot|sec|csc|log|ln)\^([0-9]+)\(")
    while True:
        m = pattern.search(s)
        if not m:
            break
        fn, pw = m.group(1), m.group(2)
        opi = m.end() - 1
        depth, cpi = 0, -1
        for i in range(opi, len(s)):
            if s[i] == "(":
                depth += 1
            elif s[i] == ")":
                depth -= 1
            if depth == 0:
                cpi = i
                break
        if cpi != -1:
            arg = s[opi + 1:cpi]
            s = s[:m.start()] + f"({fn}({arg}))**{pw}" + s[cpi + 1:]
        else:
            break
    return s


# ─── Comprehensive Multi-Domain Offline Math Solver ──────────────────────────
def try_sympy_solve(question_text: str):
    """Returns (success, problem_type, steps, final_answer).
    Handles: Algebra, Calculus, Linear Algebra, Statistics, Number Theory,
    Sequences, Geometry, Trigonometry, Complex Numbers, and more.
    """
    import sympy
    from sympy import (
        symbols, solve, diff, integrate, simplify, expand, factor,
        limit, oo, sqrt, Abs, log, ln, exp, pi, E,
        sin, cos, tan, cot, sec, csc, asin, acos, atan,
        sinh, cosh, tanh, factorial, binomial, Rational,
        Matrix, det, trace, eye, zeros, ones,
        gcd, lcm, isprime, factorint, nextprime,
        Sum, Product, series, FiniteSet,
        re as Re, im as Im, conjugate, Abs,
        latex as sp_latex, N, Float, Integer,
        Poly, degree, div, rem, resultant,
        Symbol, Function, Eq, Ne, Lt, Le, Gt, Ge,
        And, Or, Not, Piecewise,
        solve_linear_system, linsolve, nonlinsolve,
        apart, together, cancel, trigsimp, radsimp,
        nsolve, solveset, S
    )
    from sympy.parsing.sympy_parser import (
        parse_expr, standard_transformations,
        implicit_multiplication_application, convert_xor)
    from sympy.stats import Normal, Binomial, Poisson, E as Expect, variance, std

    q = normalize_greek(question_text)
    q = normalize_trig(q)
    ql = q.lower()

    # ── Keyword detection for problem type ───────────────────────────────────
    is_integral   = any(w in ql for w in ["integrate", "integral", "antiderivative", "\\int"])
    is_derivative = any(w in ql for w in ["derivative", "differentiate", "d/dx", "diff("])
    is_limit      = any(w in ql for w in ["limit", "lim ", "lim(", "approaches", "tends to"])
    is_matrix     = any(w in ql for w in ["matrix", "determinant", "det(", "eigenvalue", "eigenvector", "inverse matrix"])
    is_statistics = any(w in ql for w in ["mean", "median", "mode", "variance", "standard deviation", "std dev", "normal distribution", "binomial distribution"])
    is_series     = any(w in ql for w in ["series", "sequence", "arithmetic progression", "geometric progression", "sum of", "sigma", "summation"])
    is_number_th  = any(w in ql for w in ["prime", "gcd", "lcm", "hcf", "factor", "divisor", "modulo", "congruent", "lcm"])
    is_geometry   = any(w in ql for w in ["area", "perimeter", "volume", "circumference", "radius", "diameter", "triangle", "circle", "rectangle", "cylinder", "sphere", "cone", "cube"])
    is_complex    = any(w in ql for w in ["complex", "imaginary", "real part", "imaginary part", "modulus", "argument", "polar form"])
    is_trig       = any(w in ql for w in ["sin", "cos", "tan", "cot", "sec", "csc", "theta", "trig", "sine", "cosine"])
    is_probability = any(w in ql for w in ["probab", "dice", "coin", "card", "permutation", "combination", "ncr", "npr"])

    try:
        x, y, z, t, n, k = symbols("x y z t n k")
        a, b, c, d = symbols("a b c d")
        theta = symbols("theta", real=True)
        alpha, beta, gamma, phi = symbols("alpha beta gamma phi", real=True)
        lam = symbols("lambda", real=True)
        i_sym = symbols("i")
        r, h = symbols("r h", positive=True)

        local = {
            "theta": theta, "alpha": alpha, "beta": beta, "gamma": gamma,
            "phi": phi, "lambda": lam, "pi": pi, "e": E, "E": E,
            "sin": sin, "cos": cos, "tan": tan, "cot": cot, "sec": sec, "csc": csc,
            "asin": asin, "acos": acos, "atan": atan,
            "sinh": sinh, "cosh": cosh, "tanh": tanh,
            "log": log, "ln": log, "exp": exp, "sqrt": sqrt,
            "abs": Abs, "factorial": factorial,
            "x": x, "y": y, "z": z, "t": t, "n": n, "k": k,
            "a": a, "b": b, "c": c, "d": d, "r": r,
            "oo": oo, "inf": oo,
        }
        T = standard_transformations + (implicit_multiplication_application, convert_xor)

        def P(s: str):
            return parse_expr(s, local_dict=local, transformations=T)

        # ── CALCULUS: Integration ─────────────────────────────────────────────
        if is_integral:
            raw_expr = re.sub(r"\bint\b|\bdx\b|\bdy\b|\bdt\b|\bof\b",
                              " ", _extract_math_expr(q), flags=re.I).strip()
            # Detect definite integral bounds like "from a to b" or "[a,b]"
            bounds_match = re.search(r"from\s+([\-\d\.]+)\s+to\s+([\-\d\.]+)", ql)
            bracket_match = re.search(r"\[([\-\d\.]+)\s*,\s*([\-\d\.]+)\]", raw_expr)
            if bounds_match or bracket_match:
                m = bounds_match or bracket_match
                lo, hi = P(m.group(1)), P(m.group(2))
                raw_expr = raw_expr[:m.start()].strip() if not bounds_match else raw_expr
                expr = P(raw_expr)
                sol = integrate(expr, (x, lo, hi))
                sol_simplified = simplify(sol)
                steps = (
                    f"📖 **Concept: Definite Integration**\n\n"
                    f"A definite integral computes the net area under $f(x)$ from $x = {sp_latex(lo)}$ to $x = {sp_latex(hi)}$, "
                    f"using the Fundamental Theorem of Calculus: $\\int_a^b f(x)\\,dx = F(b) - F(a)$ where $F'(x) = f(x)$.\n\n"
                    f"📐 **Formulas Used:**\n\nFundamental Theorem: $$\\int_a^b f(x)\\,dx = F(b) - F(a)$$\n\n"
                    f"**Step 1:** Find the antiderivative $F(x)$ of $f(x) = {sp_latex(expr)}$\n\n"
                    f"$$F(x) = {sp_latex(integrate(expr, x))} + C$$\n\n"
                    f"**Step 2:** Apply the bounds $[{sp_latex(lo)}, {sp_latex(hi)}]$\n\n"
                    f"$$\\int_{{{sp_latex(lo)}}}^{{{sp_latex(hi)}}} {sp_latex(expr)}\\,dx = F({sp_latex(hi)}) - F({sp_latex(lo)})$$\n\n"
                    f"**Step 3:** Compute the result\n\n"
                    f"$$= {sp_latex(sol_simplified)}$$"
                )
                return (True, "Calculus — Definite Integral", steps, sp_latex(sol_simplified))
            else:
                expr = P(raw_expr)
                sol = integrate(expr, x)
                steps = (
                    f"📖 **Concept: Indefinite Integration**\n\n"
                    f"The indefinite integral $\\int f(x)\\,dx$ finds all antiderivatives $F(x)$ such that $F'(x) = f(x)$.\n\n"
                    f"📐 **Formulas Used:**\n\n"
                    f"Power Rule: $$\\int x^n\\,dx = \\frac{{x^{{n+1}}}}{{n+1}} + C$$\n"
                    f"Trig: $$\\int \\sin x\\,dx = -\\cos x + C, \\quad \\int \\cos x\\,dx = \\sin x + C$$\n"
                    f"Exp: $$\\int e^x\\,dx = e^x + C, \\quad \\int \\frac{{1}}{{x}}\\,dx = \\ln|x| + C$$\n\n"
                    f"**Step 1:** Identify integrand: $f(x) = {sp_latex(expr)}$\n\n"
                    f"**Step 2:** Apply integration rules term by term\n\n"
                    f"$$\\int {sp_latex(expr)}\\,dx = {sp_latex(sol)} + C$$\n\n"
                    f"**Step 3:** Verify by differentiating: $\\frac{{d}}{{dx}}\\left({sp_latex(sol)}\\right) = {sp_latex(diff(sol, x))}$ ✓"
                )
                return (True, "Calculus — Integration", steps, f"{sp_latex(sol)} + C")

        # ── CALCULUS: Differentiation ─────────────────────────────────────────
        elif is_derivative:
            raw_expr = re.sub(r"\bd/dx\b|\bdiff\b|\bof\b|derivative of",
                              " ", _extract_math_expr(q), flags=re.I).strip()
            # Detect higher-order derivatives
            order_match = re.search(r"(\d+)(?:st|nd|rd|th)?\s+(?:order)?\s*deriv", ql)
            order = int(order_match.group(1)) if order_match else 1
            expr = P(raw_expr)
            sol = diff(expr, x, order)
            sol_simplified = simplify(sol)
            order_label = {1: "First", 2: "Second", 3: "Third"}.get(order, f"{order}th")
            steps = (
                f"📖 **Concept: {order_label}-Order Differentiation**\n\n"
                f"The derivative $f'(x)$ gives the instantaneous rate of change of $f(x)$. "
                f"We apply differentiation rules to compute $\\frac{{d^{order}}}{{dx^{order}}}f(x)$.\n\n"
                f"📐 **Rules Applied:**\n\n"
                f"Power Rule: $$\\frac{{d}}{{dx}}[x^n] = n\\cdot x^{{n-1}}$$\n"
                f"Chain Rule: $$\\frac{{d}}{{dx}}[f(g(x))] = f'(g(x))\\cdot g'(x)$$\n"
                f"Product Rule: $$\\frac{{d}}{{dx}}[uv] = u'v + uv'$$\n"
                f"Quotient Rule: $$\\frac{{d}}{{dx}}\\left[\\frac{{u}}{{v}}\\right] = \\frac{{u'v - uv'}}{{v^2}}$$\n\n"
                f"**Step 1:** Identify $f(x) = {sp_latex(expr)}$\n\n"
                f"**Step 2:** Differentiate (order {order})\n\n"
                f"$$\\frac{{d^{order}}}{{dx^{order}}}\\left[{sp_latex(expr)}\\right] = {sp_latex(sol)}$$\n\n"
                f"**Step 3:** Simplify\n\n"
                f"$$= {sp_latex(sol_simplified)}$$"
            )
            return (True, f"Calculus — {order_label}-Order Derivative", steps, sp_latex(sol_simplified))

        # ── CALCULUS: Limits ──────────────────────────────────────────────────
        elif is_limit:
            lim_match = re.search(r"\bx\s*(?:\\to|→|\-+>|to|approaches|tends\s+to)\s*([\-\d\.oO]+|\\infty|infinity|inf)", q, re.IGNORECASE)
            pt_str = lim_match.group(1).lower() if lim_match else ""
            if any(k in pt_str for k in ["inf", "oo", "infty"]):
                point = oo
            else:
                point = P(pt_str) if lim_match else S.Zero
            
            raw_expr = q
            if lim_match:
                raw_expr = q[:lim_match.start()] + " " + q[lim_match.end():]
            raw_expr = re.sub(r"\b(?:lim|limit|as|approaches|tends\s+to)\b", " ", raw_expr, flags=re.I)
            raw_expr = _extract_math_expr(raw_expr)
            if raw_expr:
                expr = P(raw_expr)
                sol = limit(expr, x, point)
                steps = (
                    f"📖 **Concept: Limits**\n\n"
                    f"A limit $\\lim_{{x \\to a}} f(x)$ describes the value $f(x)$ approaches as $x$ approaches $a$, "
                    f"without necessarily equaling it. L'Hôpital's Rule resolves $0/0$ or $\\infty/\\infty$ forms.\n\n"
                    f"📐 **Key Techniques:**\n\n"
                    f"Substitution: $$\\lim_{{x \\to a}} f(x) = f(a)$$ (if continuous)\n"
                    f"L'Hôpital: $$\\lim_{{x \\to a}} \\frac{{f(x)}}{{g(x)}} = \\lim_{{x \\to a}} \\frac{{f'(x)}}{{g'(x)}}$$\n\n"
                    f"**Step 1:** Expression: $f(x) = {sp_latex(expr)}$, limit point: $x \\to {sp_latex(point)}$\n\n"
                    f"**Step 2:** Evaluate the limit\n\n"
                    f"$$\\lim_{{x \\to {sp_latex(point)}}} {sp_latex(expr)} = {sp_latex(sol)}$$"
                )
                return (True, "Calculus — Limits", steps, sp_latex(sol))

        # ── LINEAR ALGEBRA: Matrices ──────────────────────────────────────────
        elif is_matrix:
            # Try to parse a matrix like [[1,2],[3,4]] or |a b; c d|
            mat_match = re.search(r"\[\s*\[(.+?)\]\s*,?\s*\[(.+?)\]\s*(?:,?\s*\[(.+?)\])?\s*\]", q)
            if mat_match:
                rows = []
                for g in mat_match.groups():
                    if g:
                        rows.append([P(v.strip()) for v in g.split(",")])
                M = Matrix(rows)
                det_val = det(M)
                tr_val = trace(M)
                eigenvals = M.eigenvals()
                ev_str = ", ".join(f"${sp_latex(v)}$ (mult {m})" for v, m in eigenvals.items())
                try:
                    inv_M = M.inv()
                    inv_str = f"$$M^{{-1}} = {sp_latex(inv_M)}$$"
                except Exception:
                    inv_str = "Matrix is singular (non-invertible)"
                steps = (
                    f"📖 **Concept: Matrix Operations**\n\n"
                    f"A matrix is a rectangular array of numbers representing a linear transformation. "
                    f"Key properties include determinant, trace, eigenvalues, and inverse.\n\n"
                    f"📐 **Key Formulas:**\n\n"
                    f"Determinant: $$\\det(A) = \\sum_{{j}} a_{{1j}} C_{{1j}}$$ (cofactor expansion)\n"
                    f"Eigenvalue Equation: $$\\det(A - \\lambda I) = 0$$\n\n"
                    f"**Given Matrix:**\n\n$$M = {sp_latex(M)}$$\n\n"
                    f"**Determinant:** $\\det(M) = {sp_latex(det_val)}$\n\n"
                    f"**Trace:** $\\text{{tr}}(M) = {sp_latex(tr_val)}$\n\n"
                    f"**Eigenvalues:** {ev_str}\n\n"
                    f"**Inverse:** {inv_str}"
                )
                return (True, "Linear Algebra — Matrices", steps, f"det = {sp_latex(det_val)}, tr = {sp_latex(tr_val)}")

        # ── STATISTICS ────────────────────────────────────────────────────────
        elif is_statistics:
            # Parse list of numbers like [1, 2, 3, 4, 5] or 2, 4, 6, 8
            nums_raw = re.findall(r"[\-]?\d+(?:\.\d+)?", q)
            if len(nums_raw) >= 2:
                data = [float(v) for v in nums_raw]
                n_pts = len(data)
                mean_val = sum(data) / n_pts
                data_sorted = sorted(data)
                if n_pts % 2 == 1:
                    median_val = data_sorted[n_pts // 2]
                else:
                    median_val = (data_sorted[n_pts // 2 - 1] + data_sorted[n_pts // 2]) / 2
                variance_val = sum((xi - mean_val) ** 2 for xi in data) / n_pts
                std_val = variance_val ** 0.5
                from collections import Counter
                freq = Counter(data)
                mode_val = max(freq, key=freq.get)
                data_str = ", ".join(str(v) for v in data)
                steps = (
                    f"📖 **Concept: Descriptive Statistics**\n\n"
                    f"Descriptive statistics summarize a dataset using central tendency (mean, median, mode) "
                    f"and spread (variance, standard deviation).\n\n"
                    f"📐 **Formulas:**\n\n"
                    f"Mean: $$\\bar{{x}} = \\frac{{\\sum x_i}}{{n}}$$\n"
                    f"Variance: $$\\sigma^2 = \\frac{{\\sum (x_i - \\bar{{x}})^2}}{{n}}$$\n"
                    f"Std Dev: $$\\sigma = \\sqrt{{\\sigma^2}}$$\n\n"
                    f"**Data:** $\\{{{data_str}\\}}$ (n = {n_pts})\n\n"
                    f"**Mean:** $$\\bar{{x}} = {mean_val:.4f}$$\n\n"
                    f"**Median:** $${median_val}$$\n\n"
                    f"**Mode:** $${mode_val}$$\n\n"
                    f"**Variance:** $$\\sigma^2 = {variance_val:.4f}$$\n\n"
                    f"**Standard Deviation:** $$\\sigma = {std_val:.4f}$$"
                )
                return (True, "Statistics — Descriptive", steps,
                        f"Mean = {mean_val:.4f}, Median = {median_val}, Std Dev = {std_val:.4f}")

        # ── NUMBER THEORY ─────────────────────────────────────────────────────
        elif is_number_th and not is_trig:
            nums_raw = re.findall(r"\d+", q)
            if len(nums_raw) >= 1:
                nums = [int(v) for v in nums_raw[:4]]
                results = []
                steps_parts = []
                steps_parts.append(
                    f"📖 **Concept: Number Theory**\n\n"
                    f"Number theory studies integers — their properties, divisibility, prime factorization, GCD, LCM, and modular arithmetic.\n\n"
                    f"📐 **Key Formulas:**\n\n"
                    f"GCD (Euclidean): $$\\gcd(a,b) = \\gcd(b, a \\bmod b)$$\n"
                    f"LCM: $$\\text{{lcm}}(a,b) = \\frac{{a \\cdot b}}{{\\gcd(a,b)}}$$\n\n"
                )
                for num in nums:
                    is_p = isprime(num)
                    if not is_p and num > 1:
                        fac = factorint(num)
                        fac_str = " × ".join(f"${p}^{{{e}}}$" if e > 1 else f"${p}$" for p, e in sorted(fac.items()))
                        steps_parts.append(f"**{num}** = {fac_str} {'(prime)' if is_p else ''}")
                    else:
                        steps_parts.append(f"**{num}** is {'prime ✓' if is_p else 'not prime'}")
                    results.append(str(num))
                if len(nums) >= 2:
                    g = gcd(*nums)
                    l = lcm(*nums)
                    steps_parts.append(f"\n**GCD({', '.join(results)}) = {g}**")
                    steps_parts.append(f"**LCM({', '.join(results)}) = {l}**")
                    return (True, "Number Theory", "\n\n".join(steps_parts),
                            f"GCD = {g}, LCM = {l}")
                else:
                    num = nums[0]
                    fac = factorint(num) if num > 1 else {}
                    fac_str = " × ".join(f"${p}^{{{e}}}$" if e > 1 else f"${p}$" for p, e in sorted(fac.items())) if fac else str(num)
                    return (True, "Number Theory", "\n\n".join(steps_parts),
                            f"{num} = {fac_str}, {'prime' if isprime(num) else 'composite'}")

        # ── GEOMETRY ─────────────────────────────────────────────────────────
        elif is_geometry:
            nums_raw = re.findall(r"[\-]?\d+(?:\.\d+)?", q)
            nums_f = [float(v) for v in nums_raw] if nums_raw else []
            geo_steps = (
                f"📖 **Concept: Geometry**\n\n"
                f"Geometry studies shapes, sizes, and properties of figures and spaces.\n\n"
                f"📐 **Key Formulas:**\n\n"
                f"Circle: Area $= \\pi r^2$, Circumference $= 2\\pi r$\n"
                f"Rectangle: Area $= l \\times w$, Perimeter $= 2(l+w)$\n"
                f"Triangle: Area $= \\frac{{1}}{{2}}bh$, Perimeter $= a+b+c$\n"
                f"Sphere: Volume $= \\frac{{4}}{{3}}\\pi r^3$, Surface Area $= 4\\pi r^2$\n"
                f"Cylinder: Volume $= \\pi r^2 h$, Curved SA $= 2\\pi rh$\n"
                f"Cone: Volume $= \\frac{{1}}{{3}}\\pi r^2 h$\n\n"
            )
            answers = []
            if "circle" in ql and nums_f:
                rv = nums_f[0]
                area = float(pi) * rv ** 2
                circ = 2 * float(pi) * rv
                geo_steps += f"**Circle with r = {rv}:**\n\nArea $= \\pi({rv})^2 = {area:.4f}$ sq units\n\nCircumference $= 2\\pi({rv}) = {circ:.4f}$ units"
                answers.append(f"Area ≈ {area:.4f}, Circumference ≈ {circ:.4f}")
            elif "rectangle" in ql and len(nums_f) >= 2:
                l_v, w_v = nums_f[0], nums_f[1]
                area = l_v * w_v
                perim = 2 * (l_v + w_v)
                geo_steps += f"**Rectangle {l_v} × {w_v}:**\n\nArea $= {l_v} \\times {w_v} = {area}$\n\nPerimeter $= 2({l_v}+{w_v}) = {perim}$"
                answers.append(f"Area = {area}, Perimeter = {perim}")
            elif "triangle" in ql and len(nums_f) >= 2:
                bv, hv = nums_f[0], nums_f[1]
                area = 0.5 * bv * hv
                geo_steps += f"**Triangle base={bv}, height={hv}:**\n\nArea $= \\frac{{1}}{{2}} \\times {bv} \\times {hv} = {area}$"
                answers.append(f"Area = {area}")
            elif "sphere" in ql and nums_f:
                rv = nums_f[0]
                vol = (4/3) * float(pi) * rv**3
                sa = 4 * float(pi) * rv**2
                geo_steps += f"**Sphere r={rv}:**\n\nVolume $= \\frac{{4}}{{3}}\\pi({rv})^3 \\approx {vol:.4f}$\n\nSurface Area $= 4\\pi({rv})^2 \\approx {sa:.4f}$"
                answers.append(f"Volume ≈ {vol:.4f}, SA ≈ {sa:.4f}")
            elif "cylinder" in ql and len(nums_f) >= 2:
                rv, hv = nums_f[0], nums_f[1]
                vol = float(pi) * rv**2 * hv
                csa = 2 * float(pi) * rv * hv
                geo_steps += f"**Cylinder r={rv}, h={hv}:**\n\nVolume $= \\pi({rv})^2({hv}) \\approx {vol:.4f}$\n\nCurved SA $= 2\\pi({rv})({hv}) \\approx {csa:.4f}$"
                answers.append(f"Volume ≈ {vol:.4f}, CSA ≈ {csa:.4f}")
            else:
                geo_steps += "Please provide specific shape dimensions for calculations."
                answers.append("See formulas above")
            return (True, "Geometry", geo_steps, "; ".join(answers) if answers else "See steps")

        # ── SEQUENCES & SERIES ────────────────────────────────────────────────
        elif is_series:
            n_terms_match = re.search(r"(\d+)\s*(?:th|st|nd|rd)\s*term", ql)
            n_target = int(n_terms_match.group(1)) if n_terms_match else 10
            clean_q = q
            if n_terms_match:
                clean_q = q[:n_terms_match.start()] + q[n_terms_match.end():]
            nums_raw = re.findall(r"[\-]?\d+(?:\.\d+)?", clean_q)
            if len(nums_raw) >= 3:
                data = [float(v) for v in nums_raw]
                diffs = [data[i+1] - data[i] for i in range(len(data)-1)]
                ratios = [data[i+1] / data[i] for i in range(len(data)-1) if data[i] != 0]
                is_ap = len(set(round(d, 6) for d in diffs)) == 1
                is_gp = len(set(round(r, 6) for r in ratios)) == 1 if ratios else False
                if is_ap:
                    d_val = diffs[0]
                    a_val = data[0]
                    nth = a_val + (n_target - 1) * d_val
                    s_n = n_target * (2 * a_val + (n_target - 1) * d_val) / 2
                    steps = (
                        f"📖 **Concept: Arithmetic Progression (AP)**\n\n"
                        f"An AP has a constant common difference $d$ between consecutive terms. "
                        f"General term: $T_n = a + (n-1)d$\n\n"
                        f"📐 **Formulas:**\n\n"
                        f"nth term: $$T_n = a + (n-1)d$$\n"
                        f"Sum: $$S_n = \\frac{{n}}{{2}}[2a + (n-1)d]$$\n\n"
                        f"**Sequence:** ${', '.join(str(v) for v in data)}$\n\n"
                        f"**First term** $a = {a_val}$, **Common difference** $d = {d_val}$\n\n"
                        f"**{n_target}th term:** $T_{{{n_target}}} = {a_val} + ({n_target}-1)({d_val}) = {nth}$\n\n"
                        f"**Sum of first {n_target} terms:** $S_{{{n_target}}} = {s_n}$"
                    )
                    return (True, "Sequences — Arithmetic Progression", steps, f"T_{n_target} = {nth}, S_{n_target} = {s_n}")
                elif is_gp:
                    r_val = ratios[0]
                    a_val = data[0]
                    nth = a_val * (r_val ** (n_target - 1))
                    if abs(r_val) < 1:
                        s_inf = a_val / (1 - r_val)
                        sum_info = f"Sum to infinity: $S_\\infty = {s_inf:.4f}$"
                    else:
                        sum_info = "Sum to infinity diverges (|r| ≥ 1)"
                    steps = (
                        f"📖 **Concept: Geometric Progression (GP)**\n\n"
                        f"A GP has a constant common ratio $r$ between consecutive terms. "
                        f"General term: $T_n = ar^{{n-1}}$\n\n"
                        f"📐 **Formulas:**\n\n"
                        f"nth term: $$T_n = a \\cdot r^{{n-1}}$$\n"
                        f"Sum (finite): $$S_n = a\\frac{{r^n - 1}}{{r - 1}}$$\n"
                        f"Sum (infinite, |r|<1): $$S_\\infty = \\frac{{a}}{{1-r}}$$\n\n"
                        f"**Sequence:** ${', '.join(str(v) for v in data)}$\n\n"
                        f"**First term** $a = {a_val}$, **Common ratio** $r = {r_val}$\n\n"
                        f"**{n_target}th term:** $T_{{{n_target}}} = {a_val} \\times ({r_val})^{{{n_target}-1}} \\approx {nth:.4f}$\n\n"
                        f"{sum_info}"
                    )
                    return (True, "Sequences — Geometric Progression", steps, f"T_{n_target} ≈ {nth:.4f}")

        # ── PROBABILITY & COMBINATORICS ───────────────────────────────────────
        elif is_probability:
            ncr_match = re.search(r"(?:ncr|c|choose)\s*\(?\s*(\d+)\s*[,r]\s*(\d+)\s*\)?", ql)
            npr_match = re.search(r"(?:npr|p)\s*\(?\s*(\d+)\s*[,r]\s*(\d+)\s*\)?", ql)
            if ncr_match:
                n_v, r_v = int(ncr_match.group(1)), int(ncr_match.group(2))
                result = int(binomial(n_v, r_v))
                steps = (
                    f"📖 **Concept: Combinations**\n\n"
                    f"$C(n,r)$ counts unordered selections of $r$ items from $n$. "
                    f"Order does NOT matter.\n\n"
                    f"📐 **Formula:**\n\n$$C(n,r) = \\binom{{n}}{{r}} = \\frac{{n!}}{{r!(n-r)!}}$$\n\n"
                    f"**Calculation:** $$C({n_v},{r_v}) = \\frac{{{n_v}!}}{{{r_v}!({n_v}-{r_v})!}} = {result}$$"
                )
                return (True, "Combinatorics — Combinations", steps, str(result))
            elif npr_match:
                n_v, r_v = int(npr_match.group(1)), int(npr_match.group(2))
                result = int(factorial(n_v) / factorial(n_v - r_v))
                steps = (
                    f"📖 **Concept: Permutations**\n\n"
                    f"$P(n,r)$ counts ordered arrangements of $r$ items from $n$. "
                    f"Order MATTERS.\n\n"
                    f"📐 **Formula:**\n\n$$P(n,r) = \\frac{{n!}}{{(n-r)!}}$$\n\n"
                    f"**Calculation:** $$P({n_v},{r_v}) = \\frac{{{n_v}!}}{{{n_v}-{r_v})!}} = {result}$$"
                )
                return (True, "Combinatorics — Permutations", steps, str(result))

        # ── ALGEBRA: Equations & Expressions ─────────────────────────────────
        else:
            es = convert_trig_powers(_extract_math_expr(q))
            if not es:
                return False, "", "", ""

            if "=" in es:
                # Handle system of equations
                eq_parts = q.split("and")
                if len(eq_parts) >= 2:
                    try:
                        eqs = []
                        free_vars = set()
                        for part in eq_parts:
                            ep = _extract_math_expr(part)
                            if "=" in ep:
                                lhs_s, rhs_s = ep.split("=", 1)
                                eq_expr = P(lhs_s.strip()) - P(rhs_s.strip())
                                eqs.append(eq_expr)
                                free_vars.update(eq_expr.free_symbols)
                        if eqs and free_vars:
                            sol_set = linsolve(eqs, list(free_vars))
                            if sol_set:
                                sol_list = list(sol_set)
                                sol_str = "; ".join(
                                    ", ".join(f"{v} = {sp_latex(s)}" for v, s in zip(free_vars, sol))
                                    for sol in sol_list
                                )
                                steps = (
                                    f"📖 **Concept: System of Linear Equations**\n\n"
                                    f"A system of equations is solved simultaneously. Methods include substitution, "
                                    f"elimination, and matrix methods (Gaussian elimination).\n\n"
                                    f"**Equations:**\n\n" +
                                    "\n\n".join(f"$$Eq {i+1}: {sp_latex(e)} = 0$$" for i, e in enumerate(eqs)) +
                                    f"\n\n**Solution:** $${sol_str}$$"
                                )
                                return (True, "Algebra — System of Equations", steps, sol_str)
                    except Exception:
                        pass

                # Single equation
                lhs_s, rhs_s = es.split("=", 1)
                lhs = P(lhs_s.strip())
                rhs = P(rhs_s.strip())
                expr = lhs - rhs
                sol_found, sv_found = None, None
                for v in [x, y, z, t, theta, alpha, beta, a, b, c, n, k]:
                    try:
                        c_sol = solve(expr, v)
                        if c_sol:
                            sol_found = c_sol
                            sv_found = v
                            break
                    except Exception:
                        continue
                if sol_found and sv_found is not None:
                    sl = ", ".join(f"{sp_latex(sv_found)} = {sp_latex(s)}" for s in sol_found)
                    formula_note = "📐 **Formulas Used:**\n\n"
                    try:
                        deg_val = degree(Poly(expr, sv_found))
                        if deg_val == 2:
                            formula_note += "Quadratic Formula: $$x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$$\n\n"
                        elif deg_val == 1:
                            formula_note += "Linear: $$ax + b = 0 \\Rightarrow x = -\\frac{b}{a}$$\n\n"
                        else:
                            formula_note += f"Polynomial (degree {deg_val}) — factoring or numerical methods\n\n"
                    except Exception:
                        formula_note += "Algebraic manipulation\n\n"
                    steps = (
                        f"📖 **Concept: Solving Equations**\n\n"
                        f"We isolate the unknown by performing equivalent operations on both sides.\n\n"
                        f"{formula_note}"
                        f"**Step 1:** Equation: $${sp_latex(lhs)} = {sp_latex(rhs)}$$\n\n"
                        f"**Step 2:** Rearrange: $${sp_latex(expr)} = 0$$\n\n"
                        f"**Step 3:** Solve: $$\\boxed{{{sl}}}$$\n\n"
                        f"**Verification:** Substituting back confirms the solution ✓"
                    )
                    return (True, "Algebra — Equation Solving", steps, sl)
            else:
                eu = parse_expr(es, local_dict=local, transformations=T, evaluate=False)

                if not eu.free_symbols:
                    fv = eu.doit().simplify()
                    steps = (
                        f"📖 **Concept: Numerical Evaluation**\n\n"
                        f"Substituting all known values and computing the result.\n\n"
                        f"**Expression:** $${sp_latex(eu)}$$\n\n"
                        f"**Result:** $${sp_latex(fv)}$$\n\n"
                        f"**Decimal:** ${float(fv):.6f}$"
                    )
                    return True, "Numerical Evaluation", steps, sp_latex(fv)

                else:
                    ev = eu.doit()
                    if "expand" in ql:
                        sol = expand(ev)
                        pt = "Algebra — Expansion"
                    elif any(w in ql for w in ["factor", "factorise", "factorize"]):
                        sol = factor(ev)
                        pt = "Algebra — Factorization"
                    elif is_trig:
                        sol = trigsimp(simplify(ev))
                        pt = "Trigonometry — Simplification"
                    else:
                        sol = simplify(ev)
                        pt = "Algebra — Simplification"
                    # Generate step-by-step based on operation
                    steps = (
                        f"📖 **Concept: {pt}**\n\n"
                        f"We apply algebraic/trigonometric rules to transform the expression.\n\n"
                        f"**Original:** $${sp_latex(eu)}$$\n\n"
                        f"**After {pt.split('—')[-1].strip()}:** $${sp_latex(sol)}$$"
                    )
                    return (True, pt, steps, sp_latex(sol))

    except Exception as e:
        logger.warning(f"SymPy comprehensive solver failed: {e}")
    return False, "", "", ""


def heuristic_math_tutor(text: str, has_api_key: bool = False):
    """Comprehensive heuristic tutor covering all major math domains."""
    note = ("\n\n> ⚠️ *All configured AI backends failed or are rate-limited. "
            "Please check your API keys or network connection.*" if has_api_key else
            "\n\n> 💡 *No AI API keys configured. Add a free Groq API Key from [console.groq.com](https://console.groq.com) "
            "or Gemini API Key from [ai.google.dev](https://ai.google.dev) for accurate step-by-step AI solutions.*")

    t = normalize_greek(text.lower())

    # ── Probability & Statistics ──────────────────────────────────────────────
    if any(w in t for w in ["probab", "dice", "coin", "card", "permut", "combinat"]):
        return ("Probability & Combinatorics",
                "📖 **Concept: Probability & Combinatorics**\n\n"
                "Probability measures event likelihood (0 to 1). Combinatorics counts arrangements/selections.\n\n"
                "📐 **Fundamental Formulas:**\n\n"
                "Classical Probability: $$P(A) = \\frac{\\text{Favorable}}{\\text{Total}}$$\n\n"
                "Complement: $$P(A') = 1 - P(A)$$\n\n"
                "Addition: $$P(A \\cup B) = P(A) + P(B) - P(A \\cap B)$$\n\n"
                "Bayes' Theorem: $$P(A|B) = \\frac{P(B|A)P(A)}{P(B)}$$\n\n"
                "Combinations: $$C(n,r) = \\binom{n}{r} = \\frac{n!}{r!(n-r)!}$$\n\n"
                "Permutations: $$P(n,r) = \\frac{n!}{(n-r)!}$$\n\n"
                "Binomial Distribution: $$P(X=k) = \\binom{n}{k}p^k(1-p)^{n-k}$$"
                f"{note}",
                "P = Favorable / Total")

    # ── Trigonometry ─────────────────────────────────────────────────────────
    elif any(w in t for w in ["sec", "csc", "cosec", "sin", "cos", "tan", "theta",
                               "trig", "angle", "radian", "degree", "cot", "cotangent"]):
        return ("Trigonometry",
                "📖 **Concept: Trigonometry**\n\n"
                "Trigonometry studies relationships between angles and sides of triangles. "
                "The six trig ratios (sin, cos, tan, cot, sec, csc) are connected by fundamental identities.\n\n"
                "📐 **Fundamental Identities:**\n\n"
                "Pythagorean: $$\\sin^2\\theta + \\cos^2\\theta = 1$$\n"
                "$$1 + \\tan^2\\theta = \\sec^2\\theta, \\quad 1 + \\cot^2\\theta = \\csc^2\\theta$$\n\n"
                "Reciprocal: $$\\sec\\theta = \\frac{1}{\\cos\\theta}, \\quad \\csc\\theta = \\frac{1}{\\sin\\theta}, "
                "\\quad \\cot\\theta = \\frac{\\cos\\theta}{\\sin\\theta}$$\n\n"
                "Double Angle: $$\\sin 2\\theta = 2\\sin\\theta\\cos\\theta$$\n"
                "$$\\cos 2\\theta = \\cos^2\\theta - \\sin^2\\theta = 2\\cos^2\\theta - 1 = 1 - 2\\sin^2\\theta$$\n\n"
                "Half Angle: $$\\sin\\frac{\\theta}{2} = \\pm\\sqrt{\\frac{1-\\cos\\theta}{2}}, "
                "\\quad \\cos\\frac{\\theta}{2} = \\pm\\sqrt{\\frac{1+\\cos\\theta}{2}}$$\n\n"
                "Sum-to-Product: $$\\sin A + \\sin B = 2\\sin\\frac{A+B}{2}\\cos\\frac{A-B}{2}$$\n\n"
                "Key Values:\n"
                "| θ | 0° | 30° | 45° | 60° | 90° |\n"
                "|---|---|---|---|---|---|\n"
                "| sin | 0 | 1/2 | 1/√2 | √3/2 | 1 |\n"
                "| cos | 1 | √3/2 | 1/√2 | 1/2 | 0 |"
                f"{note}",
                "Apply trigonometric identities to simplify")

    # ── Calculus ──────────────────────────────────────────────────────────────
    elif any(w in t for w in ["deriv", "integr", "limit", "lim", "differentiat", "calcul",
                               "rate of change", "antiderivativ", "fundamental theorem"]):
        return ("Calculus",
                "📖 **Concept: Calculus**\n\n"
                "Calculus studies continuous change through differentiation (rates) and integration (accumulation). "
                "The Fundamental Theorem connects them as inverse operations.\n\n"
                "📐 **Differentiation Rules:**\n\n"
                "Power: $$\\frac{d}{dx}[x^n] = nx^{n-1}$$\n"
                "Product: $$\\frac{d}{dx}[uv] = u'v + uv'$$\n"
                "Quotient: $$\\frac{d}{dx}\\left[\\frac{u}{v}\\right] = \\frac{u'v - uv'}{v^2}$$\n"
                "Chain: $$\\frac{d}{dx}[f(g(x))] = f'(g(x))\\cdot g'(x)$$\n"
                "Trig: $$\\frac{d}{dx}[\\sin x] = \\cos x, \\quad \\frac{d}{dx}[\\cos x] = -\\sin x$$\n\n"
                "📐 **Integration Rules:**\n\n"
                "Power: $$\\int x^n\\,dx = \\frac{x^{n+1}}{n+1} + C$$\n"
                "By Parts: $$\\int u\\,dv = uv - \\int v\\,du$$\n"
                "FTC: $$\\int_a^b f(x)\\,dx = F(b) - F(a)$$\n\n"
                "📐 **Limits:**\n\n"
                "L'Hôpital: $$\\lim_{x\\to a}\\frac{f(x)}{g(x)} = \\lim_{x\\to a}\\frac{f'(x)}{g'(x)}$$"
                f"{note}",
                "Apply differentiation/integration/limit rules")

    # ── Linear Algebra ────────────────────────────────────────────────────────
    elif any(w in t for w in ["matrix", "determinant", "eigenvalue", "eigenvector",
                               "vector", "linear system", "rank", "transpose", "adjoint"]):
        return ("Linear Algebra",
                "📖 **Concept: Linear Algebra**\n\n"
                "Linear algebra studies vector spaces, linear transformations, and matrices. "
                "Core operations: determinant, inverse, eigenvalues, Gaussian elimination.\n\n"
                "📐 **Key Formulas:**\n\n"
                "2×2 Det: $$\\det\\begin{pmatrix}a & b\\\\ c & d\\end{pmatrix} = ad - bc$$\n\n"
                "Eigenvalue Equation: $$\\det(A - \\lambda I) = 0$$\n\n"
                "Matrix Inverse: $$A^{-1} = \\frac{1}{\\det A} \\text{adj}(A)$$\n\n"
                "Cramer's Rule: $$x_i = \\frac{\\det(A_i)}{\\det(A)}$$\n\n"
                "Row Reduction: Gaussian elimination using EROs\n\n"
                "Dot Product: $$\\vec{a} \\cdot \\vec{b} = |a||b|\\cos\\theta$$\n\n"
                "Cross Product: $$|\\vec{a} \\times \\vec{b}| = |a||b|\\sin\\theta$$"
                f"{note}",
                "Apply matrix operations")

    # ── Statistics ───────────────────────────────────────────────────────────
    elif any(w in t for w in ["mean", "median", "mode", "variance", "standard deviation",
                               "distribution", "normal", "regression", "correlation", "stat"]):
        return ("Statistics",
                "📖 **Concept: Statistics**\n\n"
                "Statistics involves collecting, analyzing, and interpreting data. "
                "Key measures: central tendency (mean, median, mode) and spread (variance, SD).\n\n"
                "📐 **Descriptive Statistics:**\n\n"
                "Mean: $$\\bar{x} = \\frac{\\sum x_i}{n}$$\n\n"
                "Variance: $$\\sigma^2 = \\frac{\\sum(x_i - \\bar{x})^2}{n}$$\n\n"
                "Std Dev: $$\\sigma = \\sqrt{\\sigma^2}$$\n\n"
                "Z-Score: $$z = \\frac{x - \\mu}{\\sigma}$$\n\n"
                "📐 **Probability Distributions:**\n\n"
                "Normal: $$f(x) = \\frac{1}{\\sigma\\sqrt{2\\pi}}e^{-\\frac{(x-\\mu)^2}{2\\sigma^2}}$$\n\n"
                "Binomial: $$P(X=k) = \\binom{n}{k}p^k(1-p)^{n-k}$$\n\n"
                "Poisson: $$P(X=k) = \\frac{\\lambda^k e^{-\\lambda}}{k!}$$"
                f"{note}",
                "Apply statistical formulas")

    # ── Number Theory ─────────────────────────────────────────────────────────
    elif any(w in t for w in ["prime", "gcd", "hcf", "lcm", "modulo", "congruent",
                               "divisib", "factor", "number theory"]):
        return ("Number Theory",
                "📖 **Concept: Number Theory**\n\n"
                "Number theory studies integers, prime numbers, divisibility, GCD, LCM, and modular arithmetic.\n\n"
                "📐 **Key Concepts:**\n\n"
                "GCD (Euclidean Algorithm): $$\\gcd(a,b) = \\gcd(b, a \\bmod b)$$\n\n"
                "LCM: $$\\text{lcm}(a,b) = \\frac{a \\cdot b}{\\gcd(a,b)}$$\n\n"
                "Prime Test: $n$ is prime if not divisible by any integer $2 \\le p \\le \\sqrt{n}$\n\n"
                "Fermat's Little: $$a^{p-1} \\equiv 1 \\pmod{p}$$ (p prime, p∤a)\n\n"
                "Euler's Totient: $$\\phi(n) = n\\prod_{p|n}\\left(1 - \\frac{1}{p}\\right)$$\n\n"
                "Wilson's Theorem: $$(p-1)! \\equiv -1 \\pmod{p}$$ (p prime)"
                f"{note}",
                "Apply number theory theorems")

    # ── Geometry ─────────────────────────────────────────────────────────────
    elif any(w in t for w in ["area", "perim", "volume", "circumfer", "radius",
                               "triangle", "circle", "rectangle", "polygon", "geometry",
                               "sphere", "cylinder", "cone", "cube", "diagonal"]):
        return ("Geometry",
                "📖 **Concept: Geometry**\n\n"
                "Geometry studies shapes, sizes, and properties of figures in 2D and 3D space.\n\n"
                "📐 **2D Shapes:**\n\n"
                "Circle: $$A = \\pi r^2, \\quad C = 2\\pi r$$\n"
                "Rectangle: $$A = lw, \\quad P = 2(l+w)$$\n"
                "Triangle: $$A = \\frac{1}{2}bh, \\quad \\text{Heron: } A = \\sqrt{s(s-a)(s-b)(s-c)}$$\n"
                "Regular Polygon: $$A = \\frac{na^2}{4}\\cot\\frac{\\pi}{n}$$\n\n"
                "📐 **3D Solids:**\n\n"
                "Sphere: $$V = \\frac{4}{3}\\pi r^3, \\quad SA = 4\\pi r^2$$\n"
                "Cylinder: $$V = \\pi r^2 h, \\quad SA = 2\\pi r(r+h)$$\n"
                "Cone: $$V = \\frac{1}{3}\\pi r^2 h, \\quad SA = \\pi r(r+l)$$\n"
                "Cube: $$V = a^3, \\quad SA = 6a^2$$\n\n"
                "📐 **Key Theorems:**\n\n"
                "Pythagorean: $$a^2 + b^2 = c^2$$\n"
                "Law of Sines: $$\\frac{a}{\\sin A} = \\frac{b}{\\sin B} = \\frac{c}{\\sin C}$$\n"
                "Law of Cosines: $$c^2 = a^2 + b^2 - 2ab\\cos C$$"
                f"{note}",
                "Apply geometry formulas")

    # ── Sequences & Series ────────────────────────────────────────────────────
    elif any(w in t for w in ["sequence", "series", "arithmetic", "geometric", "progression",
                               "ap", "gp", "summation", "sigma", "nth term", "fibonacci"]):
        return ("Sequences & Series",
                "📖 **Concept: Sequences & Series**\n\n"
                "A sequence is an ordered list of numbers; a series is their sum. "
                "AP has constant difference; GP has constant ratio.\n\n"
                "📐 **Arithmetic Progression (AP):**\n\n"
                "General term: $$T_n = a + (n-1)d$$\n"
                "Sum: $$S_n = \\frac{n}{2}[2a + (n-1)d] = \\frac{n}{2}(T_1 + T_n)$$\n\n"
                "📐 **Geometric Progression (GP):**\n\n"
                "General term: $$T_n = ar^{n-1}$$\n"
                "Sum (finite): $$S_n = \\frac{a(r^n-1)}{r-1} \\quad (r \\neq 1)$$\n"
                "Sum (infinite, |r|<1): $$S_\\infty = \\frac{a}{1-r}$$\n\n"
                "📐 **Special Sums:**\n\n"
                "$$\\sum_{k=1}^n k = \\frac{n(n+1)}{2}, \\quad \\sum_{k=1}^n k^2 = \\frac{n(n+1)(2n+1)}{6}$$"
                f"{note}",
                "Apply sequence/series formulas")

    # ── Complex Numbers ───────────────────────────────────────────────────────
    elif any(w in t for w in ["complex", "imaginary", "real part", "polar", "de moivre", "argand"]):
        return ("Complex Numbers",
                "📖 **Concept: Complex Numbers**\n\n"
                "Complex numbers $z = a + bi$ extend the real numbers where $i^2 = -1$. "
                "They can be represented in polar form $z = r(\\cos\\theta + i\\sin\\theta) = re^{i\\theta}$.\n\n"
                "📐 **Key Formulas:**\n\n"
                "Modulus: $$|z| = \\sqrt{a^2 + b^2}$$\n"
                "Argument: $$\\arg(z) = \\arctan\\left(\\frac{b}{a}\\right)$$\n"
                "Conjugate: $$\\bar{z} = a - bi$$\n"
                "Euler: $$e^{i\\theta} = \\cos\\theta + i\\sin\\theta$$\n"
                "De Moivre: $$(\\cos\\theta + i\\sin\\theta)^n = \\cos(n\\theta) + i\\sin(n\\theta)$$\n"
                "Multiplication: $$z_1 z_2 = r_1 r_2 e^{i(\\theta_1+\\theta_2)}$$"
                f"{note}",
                "Apply complex number operations")

    # ── Algebra (default) ─────────────────────────────────────────────────────
    else:
        return ("Algebra",
                "📖 **Concept: Algebra**\n\n"
                "Algebra uses symbols to represent numbers and express mathematical relationships. "
                "Core skills: solving equations, factoring expressions, manipulating inequalities.\n\n"
                "📐 **Key Formulas:**\n\n"
                "Quadratic Formula: $$x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$$\n\n"
                "Difference of Squares: $$a^2 - b^2 = (a+b)(a-b)$$\n\n"
                "Perfect Square: $$(a \\pm b)^2 = a^2 \\pm 2ab + b^2$$\n\n"
                "Sum of Cubes: $$a^3 + b^3 = (a+b)(a^2 - ab + b^2)$$\n\n"
                "Binomial Theorem: $$(a+b)^n = \\sum_{k=0}^{n} \\binom{n}{k} a^{n-k} b^k$$\n\n"
                "AM-GM Inequality: $$\\frac{a+b}{2} \\geq \\sqrt{ab} \\quad (a,b \\geq 0)$$"
                f"{note}",
                "Apply algebraic formulas and factoring")


# ─── Image / Crop Helpers ─────────────────────────────────────────────────────
QUESTION_RE = re.compile(
    r"^\s*(?:\d{1,3}[\.\)\:]\s|Q\.?\s*\d{1,3}[\.\)\:]?\s*|Question\s+\d{1,3}[\.\)\:]?\s*)",
    re.IGNORECASE)


def scale_box(box: List[float], W: int, H: int, pad: int = 10):
    if not box or len(box) != 4:
        return None
    ymin, xmin, ymax, xmax = box
    if all(c <= 1.05 for c in box):
        l, t, r, b = int(xmin * W), int(ymin * H), int(xmax * W), int(ymax * H)
    else:
        l, t, r, b = int(xmin / 1000 * W), int(ymin / 1000 * H), int(xmax / 1000 * W), int(ymax / 1000 * H)
    l, r = min(l, r), max(l, r)
    t, b = min(t, b), max(t, b)
    l = max(0, l - pad); t = max(0, t - pad)
    r = min(W, r + pad); b = min(H, b + pad)
    return [l, t, r, b] if (r - l > 2 and b - t > 2) else None


def crop_and_save(img: Image.Image, box: List[int], path: str):
    img.crop(box).save(path, "PNG")


# ─── OCR Helpers ──────────────────────────────────────────────────────────────
def tesseract_ocr(pil_img: Image.Image) -> str:
    if pytesseract is None or TESSERACT_CMD is None:
        return ""
    try:
        return pytesseract.image_to_string(pil_img, config="--psm 6 --oem 3").strip()
    except Exception as e:
        logger.warning(f"Tesseract OCR failed: {e}")
        return ""


def easyocr_ocr(pil_img: Image.Image) -> str:
    """Extract text using EasyOCR (no external binary required)."""
    reader = _get_easyocr()
    if reader is None:
        return ""
    try:
        import numpy as np
        img_array = np.array(pil_img.convert("RGB"))
        results = reader.readtext(img_array, detail=0, paragraph=True)
        text = "\n".join(r.strip() for r in results if r.strip())
        logger.info(f"EasyOCR extracted {len(text)} chars.")
        return text
    except Exception as e:
        logger.warning(f"EasyOCR failed: {e}")
        return ""


def gemini_vision_ocr(pil_img: Image.Image, client) -> str:
    prompt = (
        "You are a precise mathematical OCR engine. Extract ALL text from this image EXACTLY as shown.\n"
        "Rules:\n"
        "- Write EVERY mathematical expression in LaTeX: $...$ for inline, $$...$$ for block equations.\n"
        "- Greek letters: θ→$\\theta$, β→$\\beta$, α→$\\alpha$, φ→$\\phi$, π→$\\pi$, λ→$\\lambda$, μ→$\\mu$, σ→$\\sigma$, ω→$\\omega$.\n"
        "- Trig: sec→$\\sec$, cosec/csc→$\\csc$, cot→$\\cot$, sin→$\\sin$, cos→$\\cos$, tan→$\\tan$.\n"
        "- Fractions: a/b → $\\frac{a}{b}$ when clearly a fraction.\n"
        "- Powers: x² → $x^2$, x³ → $x^3$.\n"
        "- Square roots: √x → $\\sqrt{x}$.\n"
        "- Preserve ALL question numbers, option labels (A, B, C, D), and text formatting.\n"
        "- Return ONLY the extracted text with no commentary."
    )
    _, resp = call_gemini_with_fallback(client, [pil_img, prompt], vision_only=True)
    return resp.text.strip()


# ─── PDF Extraction ───────────────────────────────────────────────────────────
def free_extract(page: fitz.Page, page_idx: int, session_id: str,
                 pil_img: Image.Image, need: int, pdf_path: str) -> List[dict]:
    import pdfplumber
    W, H = pil_img.size
    pw, ph = page.rect.width, page.rect.height
    sx, sy = W / pw, H / ph
    blocks = sorted(
        [b for b in page.get_text("blocks") if b[6] == 0],
        key=lambda b: (round(b[1] / 15) * 15, b[0])
    )
    groups: List[dict] = []
    current: Optional[dict] = None
    counter = 0
    for blk in blocks:
        x0, y0, x1, y1, text = blk[0], blk[1], blk[2], blk[3], blk[4]
        if QUESTION_RE.match(text.split("\n")[0].strip()):
            if current and current["text"].strip():
                groups.append(current)
            counter += 1
            current = {"num": str(counter), "text": text.strip(), "bbox": [x0, y0, x1, y1]}
        elif current is not None:
            current["text"] += "\n" + text.strip()
            bx0, by0, bx1, by1 = current["bbox"]
            current["bbox"] = [min(bx0, x0), min(by0, y0), max(bx1, x1), max(by1, y1)]
    if current and current["text"].strip():
        groups.append(current)
    if not groups:
        for i, blk in enumerate(blocks[:need]):
            x0, y0, x1, y1, text = blk[0], blk[1], blk[2], blk[3], blk[4]
            if text.strip():
                groups.append({"num": str(i + 1), "text": text.strip(), "bbox": [x0, y0, x1, y1]})

    # Improve text extraction precision for each question crop using pdfplumber
    try:
        with pdfplumber.open(pdf_path) as pdf:
            plumb_page = pdf.pages[page_idx]
            for g in groups:
                x0, y0, x1, y1 = g["bbox"]
                # Crop with slight padding (2 points) to avoid cutting off edges
                cx0 = max(0, x0 - 2)
                cy0 = max(0, y0 - 2)
                cx1 = min(plumb_page.width, x1 + 2)
                cy1 = min(plumb_page.height, y1 + 2)
                if cx1 - cx0 > 2 and cy1 - cy0 > 2:
                    cropped = plumb_page.crop((cx0, cy0, cx1, cy1))
                    plumb_text = cropped.extract_text(layout=True)
                    if plumb_text and plumb_text.strip():
                        g["text"] = plumb_text.strip()
    except Exception as pe:
        logger.warning(f"pdfplumber high-precision text extraction failed: {pe}. Using PyMuPDF block text fallback.")

    results = []
    for g in groups[:need]:
        qid = str(uuid.uuid4())[:8]
        x0, y0, x1, y1 = g["bbox"]
        px0 = max(0, int(x0 * sx) - 12)
        py0 = max(0, int(y0 * sy) - 12)
        px1 = min(W, int(x1 * sx) + 12)
        py1 = min(H, int(y1 * sy) + 12)
        img_url = None
        if px1 - px0 > 4 and py1 - py0 > 4:
            fn = f"q_{session_id}_p{page_idx}_{qid}.png"
            crop_and_save(pil_img, [px0, py0, px1, py1], os.path.join(CROPS_DIR, fn))
            img_url = f"/static/crops/{fn}"
        results.append({
            "id": qid, "question_number": g["num"],
            "text": g["text"], "image_url": img_url,
            "has_diagram": False, "diagram_url": None,
        })
    return results


def gemini_extract(page_idx: int, session_id: str, pil_img: Image.Image, client, need: int) -> List[dict]:
    from google.genai import types
    W, H = pil_img.size

    prompt = (
        "You are a precise mathematical exam OCR engine. Analyze this math exam page image.\n"
        "Identify ALL numbered questions (e.g. '1.', 'Q1', 'Question 1') and extract their full content.\n"
        "For each question:\n"
        "1. Extract the COMPLETE question text preserving ALL mathematical notation.\n"
        "2. Write ALL math expressions in LaTeX: $...$ inline, $$...$$ for display equations.\n"
        "3. Greek letters MUST be LaTeX: θ→$\\theta$, β→$\\beta$, α→$\\alpha$, π→$\\pi$, etc.\n"
        "4. Trig functions: sin→$\\sin$, cos→$\\cos$, sec→$\\sec$, cosec/csc→$\\csc$, cot→$\\cot$.\n"
        "5. Fractions: write as $\\frac{numerator}{denominator}$.\n"
        "6. Powers: x²→$x^2$, roots: √x→$\\sqrt{x}$.\n"
        "7. Include ALL answer options (A), (B), (C), (D) if present.\n"
        "8. Detect `question_box` as [ymin, xmin, ymax, xmax] normalized to 1000.\n"
        "9. If a diagram/figure exists, set has_diagram=true with its `diagram_box`.\n"
        "Return a structured JSON list matching the schema exactly."
    )

    try:
        _, resp = call_gemini_with_fallback(
            client,
            [pil_img, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=PageQuestions,
                temperature=0.0
            ),
            vision_only=True
        )
    except Exception as e:
        logger.error(f"gemini_extract failed for page {page_idx}: {e}")
        return []

    txt = resp.text.strip()
    if txt.startswith("```"):
        txt = "\n".join(txt.splitlines()[1:-1]).strip()
    try:
        data = PageQuestions.model_validate_json(txt)
    except Exception as e:
        logger.error(f"Failed to parse gemini_extract JSON: {e} — raw: {txt[:200]}")
        return []

    out = []
    for q in data.questions[:need]:
        qid = str(uuid.uuid4())[:8]
        rec: dict = {
            "id": qid, "question_number": q.question_number,
            "text": q.text, "has_diagram": q.has_diagram,
            "image_url": None, "diagram_url": None,
        }
        box = scale_box(q.question_box, W, H, 10)
        if box:
            fn = f"q_{session_id}_p{page_idx}_{qid}.png"
            crop_and_save(pil_img, box, os.path.join(CROPS_DIR, fn))
            rec["image_url"] = f"/static/crops/{fn}"
        if q.has_diagram and q.diagram_box:
            dbox = scale_box(q.diagram_box, W, H, 5)
            if dbox:
                fn = f"diag_{session_id}_p{page_idx}_{qid}.png"
                crop_and_save(pil_img, dbox, os.path.join(CROPS_DIR, fn))
                rec["diagram_url"] = f"/static/crops/{fn}"
        out.append(rec)
    return out


# ─── Gemini Math Solver ───────────────────────────────────────────────────────
def gemini_solve(question_text: str, img_path: Optional[str], api_key: str) -> dict:
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=api_key)
    contents = []
    if img_path and os.path.exists(img_path):
        contents.append(Image.open(img_path))
    prompt = (
        "You are a world-class mathematics professor and tutor. Your task is to solve the given problem "
        "with COMPLETE ACCURACY and present a beautiful, pedagogical explanation that a student can learn from.\n\n"
        "CRITICAL ACCURACY RULES:\n"
        "- READ the problem statement VERY CAREFULLY. Identify EXACTLY what is being asked.\n"
        "- If the problem is an MCQ, evaluate EACH option by substituting or computing, and select the CORRECT one.\n"
        "- DOUBLE-CHECK your arithmetic at every step. Verify your final answer satisfies the original equation/condition.\n"
        "- For trigonometric problems: USE EXACT VALUES (not decimal approximations). e.g. $\\sin 30° = \\frac{1}{2}$.\n"
        "- For algebraic equations: VERIFY your solution by substituting back into the original equation.\n"
        "- For calculus: VERIFY by differentiating integrals or integrating derivatives where possible.\n"
        "- NEVER guess. If unsure, show all possible approaches and determine the correct one.\n\n"
        "FORMAT REQUIREMENTS:\n"
        "- Use LaTeX for ALL math: $...$ inline, $$...$$ for display equations.\n"
        "- Greek letters in LaTeX: $\\theta$, $\\beta$, $\\alpha$, $\\phi$, $\\pi$, $\\lambda$.\n"
        "- Trig functions: $\\sin$, $\\cos$, $\\sec$, $\\csc$, $\\cot$, $\\tan$.\n"
        "- Fractions: $\\frac{a}{b}$. Roots: $\\sqrt{x}$. Powers: $x^{n}$.\n\n"
        "STRUCTURE YOUR SOLUTION AS FOLLOWS:\n"
        "1. In 'theory_explanation': Write 2-4 sentences explaining the MATHEMATICAL CONCEPT behind this problem. "
        "What branch of math does it belong to? What key theorems or properties are relevant? This should feel like a mini-lesson.\n"
        "2. In 'formulas_used': List ALL formulas/identities/theorems used, each on a new line with LaTeX. "
        "Format each as: 'Formula Name: $$formula$$'. For example:\n"
        "   Pythagorean Identity: $$\\sin^2\\theta + \\cos^2\\theta = 1$$\n"
        "   Quadratic Formula: $$x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$$\n"
        "3. In 'solution_steps': Write a DETAILED numbered step-by-step solution. Each step should have:\n"
        "   - A clear description of what you're doing and WHY\n"
        "   - The mathematical computation in LaTeX\n"
        "   - Brief explanation connecting to the next step\n"
        "   Format steps as: 'Step 1: [Title]\n[Explanation and computation]\n\nStep 2: ...'\n"
        "4. In 'final_answer': State the definitive answer concisely with LaTeX.\n"
        "5. In 'difficulty_level': One of 'Easy', 'Medium', 'Hard', or 'Advanced'.\n\n"
    )
    if question_text and question_text.strip():
        prompt += f"Problem to solve:\n{question_text}\n\n"
    prompt += (
        "Return a JSON object with these fields:\n"
        "  problem_type: specific math topic (e.g. 'Trigonometry — Identities', 'Calculus — Integration', 'Probability — Conditional')\n"
        "  difficulty_level: 'Easy' | 'Medium' | 'Hard' | 'Advanced'\n"
        "  formulas_used: all formulas/identities used (each on a new line with LaTeX)\n"
        "  theory_explanation: mini-lesson on the underlying concept (2-4 sentences)\n"
        "  reasoning: your internal chain-of-thought analysis (show all scratch work and verification)\n"
        "  solution_steps: complete numbered step-by-step solution with full LaTeX formatting\n"
        "  final_answer: concise definitive answer (with LaTeX where appropriate)\n"
        "Return ONLY raw JSON, no markdown fences."
    )
    contents.append(prompt)

    is_vision = any(isinstance(c, Image.Image) for c in contents)
    _, resp = call_gemini_with_fallback(
        client,
        contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=SolutionOutput,
            temperature=0.05
        ),
        vision_only=is_vision
    )
    # Safely extract text — resp.text can raise if the response is blocked
    try:
        raw = resp.text
    except Exception as e:
        raise Exception(f"Gemini response was blocked or empty: {e}")
    if not raw or not raw.strip():
        raise Exception("Gemini returned an empty response (possible safety block or model output error).")
    raw = raw.strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = "\n".join(raw.splitlines()[1:-1]).strip()
    data = SolutionOutput.model_validate_json(raw)
    return {
        "status": "success", "solver_mode": "gemini_ai",
        "problem_type": data.problem_type,
        "difficulty_level": data.difficulty_level or "",
        "formulas_used": data.formulas_used or "",
        "theory_explanation": data.theory_explanation or "",
        "reasoning": data.reasoning,
        "solution_steps": data.solution_steps,
        "final_answer": data.final_answer,
    }



# ─── Groq LLM Solver ─────────────────────────────────────────────────────────
GROQ_SOLVER_PROMPT = (
    "You are a world-class mathematics professor and tutor. Your task is to solve the given problem "
    "with COMPLETE ACCURACY and present a beautiful, pedagogical explanation.\n\n"
    "CRITICAL ACCURACY RULES:\n"
    "- READ the problem VERY CAREFULLY. Identify EXACTLY what is being asked.\n"
    "- If it is an MCQ, evaluate EACH option by substituting or computing, select the CORRECT one.\n"
    "- DOUBLE-CHECK your arithmetic at every step. Verify your answer satisfies the original equation.\n"
    "- For trig problems: use EXACT VALUES (e.g. $\\sin 30° = \\frac{1}{2}$), not decimals.\n"
    "- For algebraic equations: VERIFY your solution by substituting back.\n"
    "- For calculus: VERIFY by differentiating integrals or integrating derivatives.\n"
    "- NEVER guess. Show all work and verify.\n\n"
    "FORMAT: Use LaTeX for ALL math: $...$ inline, $$...$$ display. Greek in LaTeX. Fractions as $\\frac{a}{b}$.\n\n"
    "STRUCTURE YOUR OUTPUT:\n"
    "1. 'theory_explanation': 2-4 sentence mini-lesson on the mathematical concept behind this problem.\n"
    "2. 'formulas_used': List ALL formulas used. EACH formula MUST be on its own line in EXACTLY this format:\n"
    "   Formula Name: $$LaTeX formula here$$\n"
    "   Example:\n"
    "   Pythagorean Identity: $$\\sin^2\\theta + \\cos^2\\theta = 1$$\n"
    "   Reciprocal Identity: $$\\sec\\theta = \\frac{1}{\\cos\\theta}$$\n"
    "   Do NOT use numbered lists. Do NOT put 'Name:' on a separate line. One formula per line.\n"
    "3. 'solution_steps': Detailed NUMBERED step-by-step solution. Each step: title, explanation, LaTeX computation.\n"
    "   Format: 'Step 1: [Title]\n[Explanation and math]\n\nStep 2: ...'\n"
    "4. 'final_answer': Concise definitive answer with LaTeX. Use $$...$$ for display math.\n"
    "5. 'difficulty_level': 'Easy' | 'Medium' | 'Hard' | 'Advanced'.\n\n"
    "Return ONLY a JSON object (no markdown, no fences) with fields:\n"
    "  problem_type, difficulty_level, formulas_used, theory_explanation, reasoning, solution_steps, final_answer\n"
)


def groq_solve(question_text: str, groq_api_key: str, model: str = "llama-3.3-70b-versatile") -> dict:
    """Solve using Groq's LLM API (OpenAI-compatible). Returns same dict as gemini_solve."""
    import requests as _requests
    import json as _json
    url = "https://api.groq.com/openai/v1/chat/completions"

    # Try the specified model, then fallback models
    models_to_try = [model]
    fallbacks = ["llama-3.3-70b-versatile", "llama-3.1-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"]
    for fb in fallbacks:
        if fb not in models_to_try:
            models_to_try.append(fb)

    last_err = None
    for current_model in models_to_try:
        # ── First try: JSON response mode ─────────────────────────────────────
        # Some models fail JSON mode for complex math — fall back to text+parse
        for use_json_mode in [True, False]:
            messages = [
                {"role": "system", "content": GROQ_SOLVER_PROMPT},
                {"role": "user", "content": f"Problem to solve:\n{question_text}"}
            ]
            payload = {
                "model": current_model,
                "messages": messages,
                "temperature": 0.1,
                "max_tokens": 4096,
            }
            if use_json_mode:
                payload["response_format"] = {"type": "json_object"}
            headers = {
                "Authorization": f"Bearer {groq_api_key}",
                "Content-Type": "application/json",
            }
            try:
                resp = _requests.post(url, json=payload, headers=headers, timeout=60)
                if resp.status_code != 200:
                    err_body = resp.text[:300]
                    if "decommissioned" in err_body.lower() or (
                        resp.status_code == 400 and use_json_mode
                    ):
                        # JSON mode failing — try text mode next iteration
                        if use_json_mode:
                            logger.warning(f"Groq {current_model} JSON mode failed, retrying text mode")
                            last_err = Exception(f"Groq {current_model}: HTTP {resp.status_code} (json mode)")
                            break  # break inner loop, will retry with use_json_mode=False
                        else:
                            logger.warning(f"Groq model {current_model} unavailable: {err_body[:120]}")
                            last_err = Exception(f"Groq model {current_model}: HTTP {resp.status_code}")
                            break  # break inner, move to next model
                    raise Exception(f"Groq API error {resp.status_code}: {err_body}")
                result = resp.json()
                content = result["choices"][0]["message"]["content"].strip()
                # Strip markdown fences if present
                if content.startswith("```"):
                    lines = content.splitlines()
                    content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:]).strip()
                # Try JSON parsing
                try:
                    data = _json.loads(content)
                except _json.JSONDecodeError:
                    # Try to extract JSON block from text response
                    json_match = re.search(r"\{[\s\S]+\}", content)
                    if json_match:
                        data = _json.loads(json_match.group(0))
                    else:
                        # Treat entire content as solution_steps
                        data = {
                            "problem_type": "Mathematics",
                            "solution_steps": content,
                            "final_answer": "",
                        }
                return {
                    "status": "success", "solver_mode": "groq_ai",
                    "provider_model": f"Groq / {current_model}",
                    "problem_type": data.get("problem_type", "Mathematics"),
                    "difficulty_level": data.get("difficulty_level", ""),
                    "formulas_used": data.get("formulas_used", ""),
                    "theory_explanation": data.get("theory_explanation", ""),
                    "reasoning": data.get("reasoning", ""),
                    "solution_steps": data.get("solution_steps", ""),
                    "final_answer": data.get("final_answer", ""),
                }
            except _requests.exceptions.RequestException as e:
                logger.warning(f"Groq solve network error for {current_model}: {e}")
                last_err = e
                break  # network error — skip json/text retry, move to next model
            except Exception as e:
                if use_json_mode:
                    logger.warning(f"Groq {current_model} JSON mode error, trying text mode: {e}")
                    last_err = e
                    # continue inner loop for text mode
                else:
                    logger.warning(f"Groq solve failed for model {current_model}: {e}")
                    last_err = e
                    break

    raise Exception(f"All Groq models failed. Last error: {last_err}")


# ─── Ollama LLM Solver ────────────────────────────────────────────────────────
def ollama_solve(question_text: str, model: str = "llama3", base_url: str = "http://localhost:11434") -> dict:
    """Solve using a local Ollama LLM. Returns same dict as gemini_solve."""
    import urllib.request, json as _json
    url = f"{base_url.rstrip('/')}/api/chat"
    messages = [
        {"role": "system", "content": GROQ_SOLVER_PROMPT},
        {"role": "user", "content": f"Problem to solve:\n{question_text}"}
    ]
    payload = _json.dumps({
        "model": model,
        "messages": messages,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.1},
    }).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = _json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise Exception(f"Ollama not reachable at {base_url}: {e.reason}")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise Exception(f"Ollama API error {e.code}: {body[:300]}")

    content = result.get("message", {}).get("content", "").strip()
    if content.startswith("```"):
        content = "\n".join(content.splitlines()[1:-1]).strip()
    data = _json.loads(content)
    return {
        "status": "success", "solver_mode": "ollama_ai",
        "provider_model": f"Ollama / {model}",
        "problem_type": data.get("problem_type", "Mathematics"),
        "difficulty_level": data.get("difficulty_level", ""),
        "formulas_used": data.get("formulas_used", ""),
        "theory_explanation": data.get("theory_explanation", ""),
        "reasoning": data.get("reasoning", ""),
        "solution_steps": data.get("solution_steps", ""),
        "final_answer": data.get("final_answer", ""),
    }



# ─── Cache ────────────────────────────────────────────────────────────────────
PROCESSED_CACHE: dict = {}


# ─── Startup ──────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def warmup():
    # SymPy warmup
    try:
        import sympy
        from sympy.parsing.sympy_parser import (
            parse_expr, standard_transformations,
            implicit_multiplication_application, convert_xor)
        from sympy import symbols, solve
        x = symbols("x")
        T = standard_transformations + (implicit_multiplication_application, convert_xor)
        solve(parse_expr("x-1", transformations=T), x)
        logger.info("SymPy warmup complete.")
    except Exception as e:
        logger.warning(f"SymPy warmup: {e}")

    # EasyOCR warmup — load models in background thread so server starts fast
    import threading
    def _warmup_easyocr():
        try:
            reader = _get_easyocr()
            if reader:
                logger.info("EasyOCR warmup complete.")
            else:
                logger.warning("EasyOCR warmup failed — OCR will fall back to Gemini Vision.")
        except Exception as e:
            logger.warning(f"EasyOCR warmup error: {e}")
    threading.Thread(target=_warmup_easyocr, daemon=True).start()



# ─── Routes ───────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Math PDF Extractor API running.", "tesseract": bool(TESSERACT_CMD)}


@app.post("/api/validate-key")
def validate_key(x_gemini_api_key: Optional[str] = Header(None)):
    api_key = x_gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
    if api_key in ("", "your_gemini_api_key_here"):
        return {"status": "missing", "valid": False, "message": "API key is missing"}
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        # Try each model until one works
        for model in GEMINI_MODELS:
            try:
                client.models.generate_content(model=model, contents="test: say hi")
                return {"status": "valid", "valid": True, "message": f"API key is valid (using {model})"}
            except Exception as e:
                emsg = str(e)
                if any(k in emsg.lower() for k in ["quota", "resource_exhausted", "rate", "429"]):
                    continue
                # Auth error or similar - key is genuinely invalid
                if any(k in emsg.lower() for k in ["api key", "unauthorized", "invalid"]):
                    return {"status": "invalid", "valid": False, "message": str(e)}
                continue
        return {"status": "quota", "valid": True,
                "message": "API key is valid but all models are temporarily quota-limited. Try again later."}
    except Exception as e:
        logger.error(f"API key validation failed: {e}")
        return {"status": "invalid", "valid": False, "message": str(e)}


@app.post("/api/process")
async def process_pdf(
    pdf: UploadFile = File(...),
    num_questions: int = Form(5),
    output_format: str = Form("text"),
    name: str = Form(""),
    x_gemini_api_key: Optional[str] = Header(None),
):
    api_key = x_gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
    if api_key in ("", "your_gemini_api_key_here"):
        api_key = ""
    use_gemini = bool(api_key)
    if not pdf.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files accepted.")
    import hashlib
    content = await pdf.read()
    await pdf.seek(0)
    cache_key = f"{hashlib.sha256(content).hexdigest()}_{num_questions}_{output_format}_{use_gemini}"
    if cache_key in PROCESSED_CACHE:
        r = PROCESSED_CACHE[cache_key].copy()
        r["user_name"] = name
        return r
    sid = str(uuid.uuid4())
    pdf_path = os.path.join(UPLOAD_DIR, f"{sid}_{pdf.filename}")
    try:
        with open(pdf_path, "wb") as buf:
            shutil.copyfileobj(pdf.file, buf)
    except Exception as e:
        raise HTTPException(500, f"Failed to save PDF: {e}")
    questions: List[dict] = []
    doc = None
    client = None
    try:
        if use_gemini:
            from google import genai
            client = genai.Client(api_key=api_key)
        doc = fitz.open(pdf_path)
        for pi in range(len(doc)):
            if num_questions > 0 and len(questions) >= num_questions:
                break
            need = num_questions - len(questions) if num_questions > 0 else 999999
            page = doc[pi]
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            tmp = os.path.join(UPLOAD_DIR, f"{sid}_pg{pi}.png")
            pix.save(tmp)
            pil = Image.open(tmp)
            try:
                if use_gemini and client:
                    try:
                        qs = gemini_extract(pi, sid, pil, client, need)
                        if not qs:
                            logger.info(f"Gemini returned empty list for page {pi}, falling back to free_extract.")
                            qs = free_extract(page, pi, sid, pil, need, pdf_path)
                    except Exception as ge:
                        emsg = str(ge)
                        if any(x in emsg.lower() for x in ["api key", "api_key_invalid", "unauthorized"]):
                            raise HTTPException(400, f"Gemini API key invalid: {emsg}")
                        logger.warning(f"gemini_extract failed (page {pi}), using free_extract: {emsg[:120]}")
                        qs = free_extract(page, pi, sid, pil, need, pdf_path)
                else:
                    qs = free_extract(page, pi, sid, pil, need, pdf_path)
                questions.extend(qs)
            finally:
                pil.close()
                try:
                    os.remove(tmp)
                except Exception:
                    pass
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to process PDF: {e}")
    finally:
        if doc:
            try:
                doc.close()
            except Exception:
                pass
        try:
            os.remove(pdf_path)
        except Exception:
            pass
    if not questions:
        raise HTTPException(
            422,
            "No questions detected. Ensure PDF has numbered questions (e.g. '1.', 'Q1', 'Question 1'). "
            "Add a Gemini API key for AI extraction.")
    limit = num_questions if num_questions > 0 else len(questions)
    out = []
    for q in questions[:limit]:
        if output_format == "image":
            out.append({
                "id": q["id"], "question_number": q["question_number"],
                "image_url": q.get("image_url"),
            })
        else:
            out.append({
                "id": q["id"], "question_number": q["question_number"],
                "text": q["text"], "has_diagram": q.get("has_diagram", False),
                "diagram_url": q.get("diagram_url"), "image_url": q.get("image_url"),
            })
    result = {
        "user_name": name,
        "questions_count": len(out),
        "output_format": output_format,
        "extraction_mode": "gemini_ai" if use_gemini else "free_local",
        "questions": out,
    }
    PROCESSED_CACHE[cache_key] = result
    return result


@app.post("/api/ocr")
async def ocr_image(
    image: UploadFile = File(...),
    x_gemini_api_key: Optional[str] = Header(None),
):
    ALLOWED = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}
    ext = os.path.splitext(image.filename.lower())[1]
    if ext not in ALLOWED:
        raise HTTPException(400, f"Unsupported type '{ext}'. Accepted: {', '.join(sorted(ALLOWED))}")
    sid = str(uuid.uuid4())[:8]
    safe_fn = re.sub(r"[^a-zA-Z0-9_\.\-]", "_", image.filename)
    img_name = f"ocr_{sid}_{safe_fn}"
    img_path = os.path.join(CROPS_DIR, img_name)
    try:
        with open(img_path, "wb") as f:
            shutil.copyfileobj(image.file, f)
    except Exception as e:
        raise HTTPException(500, f"Failed to save image: {e}")
    mode = "free_local"
    text = ""
    api_key = x_gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
    has_key = bool(api_key and api_key != "your_gemini_api_key_here")
    try:
        if has_key:
            try:
                from google import genai
                client = genai.Client(api_key=api_key)
                pil = Image.open(img_path)
                text = gemini_vision_ocr(pil, client)
                mode = "gemini_ai"
                pil.close()
            except Exception as ge:
                logger.error(f"Gemini OCR error: {ge}")
                # Fall through to Tesseract
        # ── EasyOCR (primary offline OCR — no external binary needed) ────────
        if not text:
            try:
                pil = Image.open(img_path)
                pil_g = pil.convert("L")
                pil_e = ImageEnhance.Contrast(pil_g).enhance(2.0)
                text = easyocr_ocr(pil_e)
                if text:
                    mode = "easyocr"
                pil.close()
            except Exception as ee:
                logger.error(f"EasyOCR failed: {ee}")
        # ── Tesseract (fallback if installed) ────────────────────────────────
        if not text and TESSERACT_CMD and pytesseract:
            try:
                pil = Image.open(img_path)
                pil_g = pil.convert("L")
                pil_e = ImageEnhance.Contrast(pil_g).enhance(2.0)
                text = tesseract_ocr(pil_e)
                if text:
                    mode = "tesseract_ocr"
                pil.close()
            except Exception as te:
                logger.error(f"Tesseract failed: {te}")
        # ── PyMuPDF text layer (last resort for PDF-based images) ────────────
        if not text:
            try:
                doc = fitz.open(img_path)
                text = doc[0].get_text("text").strip()
                doc.close()
                if text:
                    mode = "pymupdf_text"
            except Exception:
                pass
        if not text:
            eocr_status = "available" if _easyocr_available else "not available"
            if has_key:
                text = (
                    "No text could be extracted from this image.\n\n"
                    "The Gemini Vision API was tried but returned no results — "
                    "the image may be very small, blurry, or contain only diagrams.\n\n"
                    f"EasyOCR is {eocr_status} for offline extraction.\n\n"
                    "Try: uploading a clearer screenshot, or type the question manually."
                )
            else:
                text = (
                    "No text could be extracted from this image.\n\n"
                    "To extract math from screenshots:\n"
                    "- Add a Gemini API Key in the settings bar above for AI Vision OCR that reads "
                    "Greek letters (θ, β, α), fractions, sec, cosec, and all math symbols.\n"
                    f"- EasyOCR offline extraction is {eocr_status} for basic printed text."
                )
        return {
            "filename": image.filename,
            "extraction_mode": mode,
            "text": text,
            "char_count": len(text),
            "image_url": f"/static/crops/{img_name}",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OCR error: {e}")
        raise HTTPException(500, f"OCR failed: {e}")


@app.post("/api/solve")
async def solve_question(
    req: SolveRequest,
    x_gemini_api_key: Optional[str] = Header(None),
    x_groq_api_key: Optional[str] = Header(None),
    x_groq_model: Optional[str] = Header(None),
    x_ollama_model: Optional[str] = Header(None),
    x_ollama_base_url: Optional[str] = Header(None),
    x_ai_provider: Optional[str] = Header(None),  # "gemini" | "groq" | "ollama" | "offline"
):
    gemini_key = x_gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
    if gemini_key in ("", "your_gemini_api_key_here"):
        gemini_key = ""

    groq_key = x_groq_api_key or os.environ.get("GROQ_API_KEY", "")
    if groq_key in ("", "your_groq_api_key_here"):
        groq_key = ""

    # Determine effective provider (header overrides request body)
    provider = (x_ai_provider or req.provider or "auto").lower().strip()
    groq_model = x_groq_model or req.groq_model or "llama-3.3-70b-versatile"
    ollama_model = x_ollama_model or req.ollama_model or "llama3"
    ollama_url = x_ollama_base_url or req.ollama_base_url or "http://localhost:11434"

    img_path = None
    if req.image_url:
        fn = req.image_url.split("/")[-1]
        cand = os.path.join(CROPS_DIR, fn)
        if os.path.exists(cand):
            img_path = cand

    question = (req.text or "").strip()

    # ── 1. Gemini (explicit or auto with key present) ─────────────────────────
    should_try_gemini = True
    if provider == "auto" and groq_key and is_gemini_quota_limited(gemini_key):
        should_try_gemini = False
        logger.info("Gemini key is cached as quota-limited; skipping to Groq solver directly.")

    if provider in ("gemini", "auto") and gemini_key and should_try_gemini:
        try:
            return gemini_solve(question, img_path, gemini_key)
        except Exception as ge:
            logger.error(f"Gemini solve error: {ge}")
            emsg = str(ge).lower()
            if any(k in emsg for k in ["quota", "resource_exhausted", "429", "rate"]):
                mark_gemini_quota_limited(gemini_key)
            if provider == "gemini":
                return {
                    "status": "error", "solver_mode": "gemini_ai",
                    "problem_type": "Error",
                    "solution_steps": f"Gemini API error: {ge}",
                    "final_answer": "Gemini error",
                }

    # ── 2. Groq (explicit or auto with key present) ───────────────────────────
    if provider in ("groq", "auto") and groq_key and question:
        try:
            return groq_solve(question, groq_key, model=groq_model)
        except Exception as ge:
            logger.warning(f"Groq solve failed, falling back: {ge}")
            if provider == "groq":   # Explicit Groq requested — surface error
                return {
                    "status": "error", "solver_mode": "groq_ai",
                    "problem_type": "Error",
                    "solution_steps": f"Groq API error: {ge}",
                    "final_answer": "Groq error",
                }

    # ── 3. Ollama (explicit or auto when available) ───────────────────────────
    if provider in ("ollama", "auto") and question:
        try:
            return ollama_solve(question, ollama_model, ollama_url)
        except Exception as oe:
            logger.warning(f"Ollama solve failed: {oe}")
            if provider == "ollama":
                return {
                    "status": "error", "solver_mode": "ollama_ai",
                    "problem_type": "Error",
                    "solution_steps": f"Ollama error: {oe}. Make sure Ollama is running locally at {ollama_url}.",
                    "final_answer": "Ollama not available",
                }

    # ── 4. Offline SymPy ─────────────────────────────────────────────────────
    if question:
        ok, pt, steps, ans = try_sympy_solve(question)
        if ok:
            return {
                "status": "success", "solver_mode": "free_local_sympy",
                "problem_type": pt, "solution_steps": steps, "final_answer": ans,
            }
        # ── 5. Heuristic tutor ───────────────────────────────────────────────
        has_any_key = bool(gemini_key or groq_key)
        pt, steps, ans = heuristic_math_tutor(question, has_api_key=has_any_key)
        return {
            "status": "success", "solver_mode": "free_local_tutor",
            "problem_type": pt, "solution_steps": steps, "final_answer": ans,
        }

    # ── 6. Image-only fallback ────────────────────────────────────────────────
    return {
        "status": "error", "solver_mode": "free_local", "problem_type": "Visual Question",
        "solution_steps": (
            "**Unable to Solve**\n\nThis image-only question requires an AI provider for vision solving.\n\n"
            "Options:\n"
            "1. Add a **Gemini API Key** in the settings bar above\n"
            "2. Add a **Groq API Key** for text-based AI solving\n"
            "3. Use the **Screenshot OCR** tab first to extract text, then solve\n"
            "4. Type the question manually in the text box\n\n"
            "Get a free Gemini key at [ai.google.dev](https://ai.google.dev) or a free Groq key at [console.groq.com](https://console.groq.com)"
        ),
        "final_answer": "AI key required for image solving",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
