import os
import fitz  # PyMuPDF
import matplotlib.pyplot as plt
import numpy as np

def make_equation_image(latex_str, filepath, fontsize=16):
    """Renders a LaTeX math expression to a transparent PNG using matplotlib."""
    fig = plt.figure(figsize=(6, 0.8))
    plt.text(0.5, 0.5, latex_str, fontsize=fontsize, ha='center', va='center')
    plt.axis('off')
    plt.savefig(filepath, bbox_inches='tight', dpi=200, transparent=True)
    plt.close(fig)

def make_triangle_diagram(filepath):
    """Creates a geometry diagram of a right-angled triangle."""
    fig, ax = plt.subplots(figsize=(4, 3))
    # Triangle vertices
    x = [0, 4, 0, 0]
    y = [0, 0, 3, 0]
    ax.plot(x, y, 'b-', linewidth=3)
    ax.fill(x, y, '#e6f2ff')
    
    # Labels
    ax.text(2, -0.4, "a = 4 cm", fontsize=12, ha='center')
    ax.text(-0.5, 1.5, "b = 3 cm", fontsize=12, va='center', rotation=90)
    ax.text(2.2, 1.7, "c = ?", fontsize=12, ha='center')
    
    # Right angle marker
    ax.plot([0, 0.3, 0.3, 0], [0.3, 0.3, 0, 0], 'r-', linewidth=1.5)
    
    ax.set_xlim(-1, 5)
    ax.set_ylim(-1, 4.2)
    ax.axis('off')
    plt.savefig(filepath, bbox_inches='tight', dpi=200, transparent=True)
    plt.close(fig)

def make_parabola_diagram(filepath):
    """Creates a graph diagram of a parabola y = -x^2 + 4x - 3."""
    fig, ax = plt.subplots(figsize=(4, 3))
    x = np.linspace(0, 4, 100)
    y = -x**2 + 4*x - 3
    ax.plot(x, y, 'r-', linewidth=3, label=r'$y = -x^2 + 4x - 3$')
    
    # Draw axes
    ax.axhline(0, color='black', linewidth=1)
    ax.axvline(0, color='black', linewidth=1)
    
    # Highlight vertex (2, 1) and roots (1, 0), (3, 0)
    ax.plot(2, 1, 'go')
    ax.text(2, 1.2, "Vertex (2, 1)", fontsize=10, ha='center', color='green')
    
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-2, 2)
    ax.axis('on')
    plt.savefig(filepath, bbox_inches='tight', dpi=200, transparent=True)
    plt.close(fig)

def build_pdf(pdf_path):
    print("Generating PDF assets...")
    temp_files = []
    
    # Generate LaTeX equations
    eq1_path = "temp_eq1.png"
    make_equation_image(r"$x^2 - 5x + 6 = 0$", eq1_path)
    temp_files.append(eq1_path)
    
    eq2_path = "temp_eq2.png"
    make_equation_image(r"$\int_0^{\pi} \sin(x) \, dx$", eq2_path)
    temp_files.append(eq2_path)
    
    eq3_path = "temp_eq3.png"
    make_equation_image(r"$A = [1, 2; 3, 4]$", eq3_path)
    temp_files.append(eq3_path)
    
    # Generate diagrams
    tri_path = "temp_triangle.png"
    make_triangle_diagram(tri_path)
    temp_files.append(tri_path)
    
    para_path = "temp_parabola.png"
    make_parabola_diagram(para_path)
    temp_files.append(para_path)
    
    # Build PDF with PyMuPDF
    print("Creating PDF document...")
    doc = fitz.open()
    
    # PAGE 1: Algebra & Calculus
    page1 = doc.new_page(width=595, height=842) # A4 size
    # Draw Title
    page1.insert_text(fitz.Point(50, 60), "Mathematics Diagnostic Test", fontsize=24, fontname="hebo")
    page1.insert_text(fitz.Point(50, 90), "Section A: Algebra & Calculus", fontsize=16, fontname="hebo")
    
    # Question 1
    page1.insert_text(fitz.Point(50, 150), "Question 1. Find the roots of the quadratic equation:", fontsize=12, fontname="helv")
    # Insert equation 1
    eq1_rect = fitz.Rect(70, 170, 270, 210)
    page1.insert_image(eq1_rect, filename=eq1_path)
    page1.insert_text(fitz.Point(50, 240), "Show all your workings and state the final answer clearly.", fontsize=11, fontname="helv")
    
    # Draw a divider
    shape = page1.new_shape()
    shape.draw_line(fitz.Point(50, 280), fitz.Point(545, 280))
    shape.commit()
    
    # Question 2
    page1.insert_text(fitz.Point(50, 320), "Question 2. Evaluate the following definite integral:", fontsize=12, fontname="helv")
    # Insert equation 2
    eq2_rect = fitz.Rect(70, 340, 270, 390)
    page1.insert_image(eq2_rect, filename=eq2_path)
    page1.insert_text(fitz.Point(50, 420), "Use the fundamental theorem of calculus to evaluate the expression.", fontsize=11, fontname="helv")
    
    # Draw a divider
    shape = page1.new_shape()
    shape.draw_line(fitz.Point(50, 460), fitz.Point(545, 460))
    shape.commit()
    
    # Question 3 (Matrix)
    page1.insert_text(fitz.Point(50, 500), "Question 3. Let the 2x2 matrix A be defined as:", fontsize=12, fontname="helv")
    eq3_rect = fitz.Rect(70, 520, 270, 570)
    page1.insert_image(eq3_rect, filename=eq3_path)
    page1.insert_text(fitz.Point(50, 600), "Calculate the determinant of matrix A and find its inverse, A^-1.", fontsize=12, fontname="helv")
    
    # PAGE 2: Geometry & Graphs
    page2 = doc.new_page(width=595, height=842)
    page2.insert_text(fitz.Point(50, 60), "Section B: Geometry & Graphical Analysis", fontsize=16, fontname="hebo")
    
    # Question 4 (Triangle Geometry)
    page2.insert_text(fitz.Point(50, 120), "Question 4. Refer to the right-angled triangle shown below. Calculate the length of the", fontsize=12, fontname="helv")
    page2.insert_text(fitz.Point(50, 138), "hypotenuse 'c' using the Pythagorean Theorem.", fontsize=12, fontname="helv")
    # Insert triangle diagram
    tri_rect = fitz.Rect(70, 160, 270, 310)
    page2.insert_image(tri_rect, filename=tri_path)
    
    # Draw a divider
    shape = page2.new_shape()
    shape.draw_line(fitz.Point(50, 340), fitz.Point(545, 340))
    shape.commit()
    
    # Question 5 (Parabola Graph)
    page2.insert_text(fitz.Point(50, 380), "Question 5. The graph below displays a quadratic parabola.", fontsize=12, fontname="helv")
    page2.insert_text(fitz.Point(50, 398), "Identify the x-intercepts and write the function in intercept form.", fontsize=12, fontname="helv")
    # Insert parabola graph
    para_rect = fitz.Rect(70, 420, 320, 610)
    page2.insert_image(para_rect, filename=para_path)
    
    doc.save(pdf_path)
    doc.close()
    
    # Clean up temporary image assets
    print("Cleaning up temporary assets...")
    for f in temp_files:
        if os.path.exists(f):
            os.remove(f)
            
    print(f"Test PDF generated successfully at: {pdf_path}")

if __name__ == "__main__":
    build_pdf("test_math.pdf")
