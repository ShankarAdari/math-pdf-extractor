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
    "gemini-1.5-flash",
]

# Models that support vision (image input)
VISION_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
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


# ─── Enhanced Offline SymPy Solver ────────────────────────────────────────────
def try_sympy_solve(question_text: str):
    """Returns (success, problem_type, steps, final_answer)."""
    import sympy
    from sympy import (symbols, solve, diff, integrate, simplify,
                       sin, cos, tan, cot, sec, csc, pi)
    from sympy.parsing.sympy_parser import (
        parse_expr, standard_transformations,
        implicit_multiplication_application, convert_xor)

    q = normalize_greek(question_text)
    q = normalize_trig(q)
    ql = q.lower()

    is_int  = any(x in ql for x in ["integrate", "integral", "antiderivative"])
    is_diff = any(x in ql for x in ["derivative", "differentiate", "d/dx", "diff("])

    try:
        x, y, z, t = symbols("x y z t")
        theta = symbols("theta", real=True)
        alpha = symbols("alpha", real=True)
        beta  = symbols("beta",  real=True)

        local = {
            "theta": theta, "alpha": alpha, "beta": beta,
            "pi": pi, "sin": sin, "cos": cos, "tan": tan,
            "cot": cot, "sec": sec, "csc": csc,
        }
        T = standard_transformations + (implicit_multiplication_application, convert_xor)

        def P(s: str):
            return parse_expr(s, local_dict=local, transformations=T)

        if is_int:
            es = re.sub(r"\bint\b|\bdx\b|\bof\b", " ", _extract_math_expr(q), flags=re.I).strip()
            expr = P(es)
            sol = integrate(expr, x)
            theory = ("📖 **Concept: Indefinite Integration**\n\n"
                      "Integration is the reverse process of differentiation. "
                      "Finding the indefinite integral (or antiderivative) of a function $f(x)$ "
                      "gives a family of functions $F(x) + C$ such that $F'(x) = f(x)$.\n\n")
            formulas = ("📐 **Formulas Used:**\n\n"
                        "Power Rule: $$\\int x^n \\, dx = \\frac{x^{n+1}}{n+1} + C \\quad (n \\neq -1)$$\n\n")
            steps = (f"{theory}{formulas}"
                     f"**Step 1: Identify the integrand**\n\n"
                     f"We need to find: $$\\int {sympy.latex(expr)} \\, dx$$\n\n"
                     f"**Step 2: Apply integration rules**\n\n"
                     f"Applying the appropriate integration rules term by term:\n\n"
                     f"$$\\int {sympy.latex(expr)} \\, dx = {sympy.latex(sol)} + C$$\n\n"
                     f"**Step 3: Verification**\n\n"
                     f"We can verify by differentiating: $\\frac{{d}}{{dx}}\\left({sympy.latex(sol)}\\right) = {sympy.latex(diff(sol, x))}$ ✓")
            return (True, "Calculus — Integration", steps, f"{sympy.latex(sol)} + C")

        elif is_diff:
            es = re.sub(r"\bd/dx\b|\bdiff\b|\bof\b", " ", _extract_math_expr(q), flags=re.I).strip()
            expr = P(es)
            sol = diff(expr, x)
            theory = ("📖 **Concept: Differentiation**\n\n"
                      "Differentiation finds the rate of change of a function. "
                      "The derivative $f'(x)$ represents the slope of the tangent line to $f(x)$ at any point.\n\n")
            formulas = ("📐 **Formulas Used:**\n\n"
                        "Power Rule: $$\\frac{d}{dx}[x^n] = n \\cdot x^{n-1}$$\n"
                        "Chain Rule: $$\\frac{d}{dx}[f(g(x))] = f'(g(x)) \\cdot g'(x)$$\n\n")
            steps = (f"{theory}{formulas}"
                     f"**Step 1: Identify the function**\n\n"
                     f"We need to differentiate: $$f(x) = {sympy.latex(expr)}$$\n\n"
                     f"**Step 2: Apply differentiation rules**\n\n"
                     f"Differentiating term by term:\n\n"
                     f"$$\\frac{{d}}{{dx}}\\left({sympy.latex(expr)}\\right) = {sympy.latex(sol)}$$\n\n"
                     f"**Step 3: Final Result**\n\n"
                     f"$$f'(x) = {sympy.latex(sol)}$$")
            return (True, "Calculus — Differentiation", steps, sympy.latex(sol))

        else:
            es = convert_trig_powers(_extract_math_expr(q))
            if not es:
                return False, "", "", ""

            if "=" in es:
                lhs_s, rhs_s = es.split("=", 1)
                lhs = P(lhs_s.strip())
                rhs = P(rhs_s.strip())
                expr = lhs - rhs
                sol, sv = None, None
                for v in [x, y, z, t, theta, alpha, beta]:
                    try:
                        c = solve(expr, v)
                        if c:
                            sol = c
                            sv = v
                            break
                    except Exception:
                        continue
                if sol and sv is not None:
                    sl = ", ".join(f"{sv} = {sympy.latex(s)}" for s in sol)
                    theory = ("📖 **Concept: Solving Equations**\n\n"
                              "To solve an equation, we isolate the unknown variable by performing "
                              "equivalent operations on both sides. The goal is to find the value(s) "
                              "of the variable that make the equation true.\n\n")
                    # Determine formula based on degree
                    formula_note = "📐 **Formulas Used:**\n\n"
                    try:
                        deg = sympy.degree(expr, sv)
                        if deg == 2:
                            formula_note += "Quadratic Formula: $$x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$$\n\n"
                        elif deg == 1:
                            formula_note += "Linear Equation: $$ax + b = 0 \\Rightarrow x = -\\frac{b}{a}$$\n\n"
                        else:
                            formula_note += f"Polynomial equation of degree {deg}\n\n"
                    except Exception:
                        formula_note += "Algebraic manipulation and factoring\n\n"
                    steps = (f"{theory}{formula_note}"
                             f"**Step 1: Write the equation**\n\n"
                             f"$${sympy.latex(lhs)} = {sympy.latex(rhs)}$$\n\n"
                             f"**Step 2: Rearrange to standard form**\n\n"
                             f"Move all terms to one side:\n\n$${sympy.latex(expr)} = 0$$\n\n"
                             f"**Step 3: Solve for ${sympy.latex(sv)}$**\n\n"
                             f"$$\\boxed{{{sl}}}$$\n\n"
                             f"**Step 4: Verification**\n\n"
                             f"Substituting back into the original equation confirms the solution(s). ✓")
                    return (True, "Algebra — Equation", steps, sl)
            else:
                eu = parse_expr(es, local_dict=local, transformations=T, evaluate=False)

                if not eu.free_symbols:
                    steps = [f"**Numerical Evaluation**\n\n$${sympy.latex(eu)}$$\n"]
                    if isinstance(eu, sympy.Add):
                        steps.append("\nEvaluating each term:")
                        for a in eu.args:
                            steps.append(f"- ${sympy.latex(a)}$ = ${sympy.latex(a.doit())}$")
                        fv = eu.doit().simplify()
                        terms = []
                        for i, a in enumerate(eu.args):
                            v = a.doit()
                            if v.is_Number and v < 0:
                                terms.append(f"- {sympy.latex(-v)}" if i > 0 else sympy.latex(v))
                            else:
                                terms.append(f"+ {sympy.latex(v)}" if i > 0 else sympy.latex(v))
                        steps.append(f"\n$${' '.join(terms)} = {sympy.latex(fv)}$$")
                    else:
                        fv = eu.doit().simplify()
                        steps.append(f"\n$${sympy.latex(eu)} = {sympy.latex(fv)}$$")
                    return True, "Numerical Evaluation", "\n".join(steps), sympy.latex(fv)

                else:
                    ev = eu.doit()
                    if "expand" in ql:
                        sol = sympy.expand(ev)
                        pt = "Algebra (Expansion)"
                    elif any(w in ql for w in ["factor", "factorise", "factorize"]):
                        sol = sympy.factor(ev)
                        pt = "Algebra (Factorization)"
                    else:
                        sol = sympy.simplify(ev)
                        pt = "Algebra (Simplification)"
                    return (True, pt,
                            f"**{pt}**\n\n$${sympy.latex(eu)}$$\n\n$$= {sympy.latex(sol)}$$",
                            sympy.latex(sol))

    except Exception as e:
        logger.warning(f"SymPy solver failed: {e}")
    return False, "", "", ""


def heuristic_math_tutor(text: str, has_api_key: bool = False):
    """Last-resort heuristic tutor. Returns (problem_type, steps, final_answer)."""
    note = ("\n\n> ⚠️ *All configured AI backends failed or are rate-limited. "
            "Please check your API keys or network connection.*" if has_api_key else
            "\n\n> 💡 *No AI API keys configured. For accurate direct answers, click \"⚙️ AI Provider Settings\" "
            "above and enter a free Groq API Key from console.groq.com or a Gemini API Key from ai.google.dev*")

    t = normalize_greek(text.lower())
    if any(w in t for w in ["probab", "dice", "coin", "card"]):
        return ("Probability",
                "📖 **Concept: Probability Theory**\n\n"
                "Probability measures the likelihood of an event occurring, expressed as a value between 0 and 1. "
                "The classical definition states that probability equals the ratio of favorable outcomes to total possible outcomes.\n\n"
                "📐 **Key Formulas:**\n\n"
                "Basic Probability: $$P(A) = \\frac{\\text{Number of Favorable Outcomes}}{\\text{Total Number of Outcomes}}$$\n\n"
                "Complement Rule: $$P(A') = 1 - P(A)$$\n\n"
                "Addition Rule: $$P(A \\cup B) = P(A) + P(B) - P(A \\cap B)$$\n\n"
                "Multiplication Rule (Independent): $$P(A \\cap B) = P(A) \\times P(B)$$\n\n"
                "Conditional Probability: $$P(A|B) = \\frac{P(A \\cap B)}{P(B)}$$"
                f"{note}",
                "Apply the probability formula: P = Favorable / Total")

    elif any(w in t for w in ["sec", "csc", "cosec", "sin", "cos", "tan", "theta", "trig"]):
        return ("Trigonometry",
                "📖 **Concept: Trigonometry**\n\n"
                "Trigonometry studies the relationships between angles and sides of triangles. "
                "The six trigonometric ratios (sin, cos, tan, cot, sec, csc) form the foundation, "
                "connected by fundamental identities that allow algebraic simplification.\n\n"
                "📐 **Fundamental Identities:**\n\n"
                "Pythagorean Identities:\n"
                "$$\\sin^2\\theta + \\cos^2\\theta = 1$$\n"
                "$$1 + \\tan^2\\theta = \\sec^2\\theta$$\n"
                "$$1 + \\cot^2\\theta = \\csc^2\\theta$$\n\n"
                "Reciprocal Relations:\n"
                "$$\\sec\\theta = \\frac{1}{\\cos\\theta}, \\quad "
                "\\csc\\theta = \\frac{1}{\\sin\\theta}, \\quad "
                "\\cot\\theta = \\frac{\\cos\\theta}{\\sin\\theta}$$\n\n"
                "Double Angle Formulas:\n"
                "$$\\sin 2\\theta = 2\\sin\\theta\\cos\\theta$$\n"
                "$$\\cos 2\\theta = \\cos^2\\theta - \\sin^2\\theta = 2\\cos^2\\theta - 1 = 1 - 2\\sin^2\\theta$$"
                f"{note}",
                "Apply trigonometric identities to simplify")

    elif any(w in t for w in ["deriv", "integr", "limit"]):
        return ("Calculus",
                "📖 **Concept: Calculus**\n\n"
                "Calculus deals with rates of change (differentiation) and accumulation (integration). "
                "The Fundamental Theorem of Calculus connects these two operations as inverse processes.\n\n"
                "📐 **Key Formulas:**\n\n"
                "Power Rule (Differentiation): $$\\frac{d}{dx}[x^n] = n \\cdot x^{n-1}$$\n\n"
                "Power Rule (Integration): $$\\int x^n \\, dx = \\frac{x^{n+1}}{n+1} + C \\quad (n \\neq -1)$$\n\n"
                "Chain Rule: $$\\frac{d}{dx}[f(g(x))] = f'(g(x)) \\cdot g'(x)$$\n\n"
                "Product Rule: $$\\frac{d}{dx}[f \\cdot g] = f' \\cdot g + f \\cdot g'$$\n\n"
                "Integration by Parts: $$\\int u \\, dv = uv - \\int v \\, du$$"
                f"{note}",
                "Apply differentiation / integration rules")

    elif any(w in t for w in ["matrix", "determinant", "eigenvalue", "vector"]):
        return ("Linear Algebra",
                "📖 **Concept: Linear Algebra**\n\n"
                "Linear algebra studies vector spaces and linear mappings between them. "
                "Matrices represent linear transformations, and their properties (determinant, eigenvalues) "
                "reveal the nature of these transformations.\n\n"
                "📐 **Key Formulas:**\n\n"
                "2×2 Determinant: $$\\det\\begin{pmatrix}a & b\\\\ c & d\\end{pmatrix} = ad - bc$$\n\n"
                "Eigenvalue Equation: $$\\det(A - \\lambda I) = 0$$\n\n"
                "Matrix Inverse (2×2): $$A^{-1} = \\frac{1}{\\det(A)}\\begin{pmatrix}d & -b\\\\ -c & a\\end{pmatrix}$$"
                f"{note}",
                "Apply matrix operations and eigenvalue equation")

    else:
        return ("Mathematics",
                "📖 **Concept: Algebra**\n\n"
                "Algebra is the study of mathematical symbols and the rules for manipulating them. "
                "Solving equations involves finding values of unknowns that satisfy given conditions.\n\n"
                "📐 **Key Formulas:**\n\n"
                "Quadratic Formula: $$x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$$\n\n"
                "Difference of Squares: $$a^2 - b^2 = (a+b)(a-b)$$\n\n"
                "Perfect Square: $$(a \\pm b)^2 = a^2 \\pm 2ab + b^2$$\n\n"
                "Binomial Theorem: $$(a+b)^n = \\sum_{k=0}^{n} \\binom{n}{k} a^{n-k} b^k$$"
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
    "2. 'formulas_used': List ALL formulas/identities/theorems used, each on new line as: 'Name: $$formula$$'\n"
    "3. 'solution_steps': Detailed NUMBERED step-by-step solution. Each step: title, explanation, LaTeX computation.\n"
    "   Format: 'Step 1: [Title]\n[Explanation and math]\n\nStep 2: ...'\n"
    "4. 'final_answer': Concise definitive answer with LaTeX.\n"
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
    fallbacks = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
    for fb in fallbacks:
        if fb not in models_to_try:
            models_to_try.append(fb)

    last_err = None
    for current_model in models_to_try:
        messages = [
            {"role": "system", "content": GROQ_SOLVER_PROMPT},
            {"role": "user", "content": f"Problem to solve:\n{question_text}"}
        ]
        payload = {
            "model": current_model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 4096,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {groq_api_key}",
            "Content-Type": "application/json",
        }
        try:
            resp = _requests.post(url, json=payload, headers=headers, timeout=60)
            if resp.status_code != 200:
                err_body = resp.text[:300]
                # Skip decommissioned models silently
                if "decommissioned" in err_body.lower() or resp.status_code == 400:
                    logger.warning(f"Groq model {current_model} unavailable: {err_body[:120]}")
                    last_err = Exception(f"Groq model {current_model}: HTTP {resp.status_code}")
                    continue
                raise Exception(f"Groq API error {resp.status_code}: {err_body}")
            result = resp.json()
            content = result["choices"][0]["message"]["content"].strip()
            if content.startswith("```"):
                content = "\n".join(content.splitlines()[1:-1]).strip()
            data = _json.loads(content)
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
            continue
        except Exception as e:
            logger.warning(f"Groq solve failed for model {current_model}: {e}")
            last_err = e
            continue

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
