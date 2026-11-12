"""Gera docs/relatorio.pdf a partir de docs/relatorio.md (com as figuras de resultados/).

Uso:
    python docs/gerar_pdf.py
"""

from pathlib import Path

from markdown_pdf import MarkdownPdf, Section

RAIZ = Path(__file__).resolve().parents[1]
CSS = """
body { font-family: sans-serif; font-size: 10.5pt; line-height: 1.4; }
h1 { font-size: 20pt; } h2 { font-size: 15pt; margin-top: 18pt; } h3 { font-size: 12pt; }
table { border-collapse: collapse; margin: 6pt 0; }
th, td { border: 1px solid #999; padding: 3pt 6pt; font-size: 9.5pt; }
th { background: #eee; }
img { max-width: 100%; }
code { font-size: 9pt; }
"""


def main() -> None:
    texto = (RAIZ / "docs" / "relatorio.md").read_text(encoding="utf-8")
    pdf = MarkdownPdf(toc_level=2, optimize=True)
    pdf.meta["title"] = "Navegação de pouso lunar por visão computacional"
    pdf.meta["author"] = "GS 2026/2 - FIAP"
    # No .md as figuras são ../resultados/x.png (relativas a docs/, para o GitHub
    # mostrar). O markdown-pdf não sobe de pasta, então resolvemos pela raiz.
    texto = texto.replace("](../resultados/", "](resultados/")
    pdf.add_section(Section(texto, root=str(RAIZ), paper_size="A4"), user_css=CSS)
    destino = RAIZ / "docs" / "relatorio.pdf"
    pdf.save(str(destino))
    print(f"gerado: {destino}")


if __name__ == "__main__":
    main()
