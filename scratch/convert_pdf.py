import sys
import subprocess

try:
    from markdown_pdf import Section, MarkdownPdf
    pdf = MarkdownPdf(toc_level=2)
    pdf.add_section(Section("README.md", toc=False))
    pdf.save("README.pdf")
    print("PDF generated successfully via markdown_pdf!")
except ImportError:
    print("markdown-pdf not installed. Trying to install...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "markdown-pdf"])
    from markdown_pdf import Section, MarkdownPdf
    pdf = MarkdownPdf(toc_level=2)
    pdf.add_section(Section("README.md", toc=False))
    pdf.save("README.pdf")
    print("PDF generated successfully via markdown_pdf!")
except Exception as e:
    print(f"Failed to generate PDF: {e}")
