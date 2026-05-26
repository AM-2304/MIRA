import os
import re
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

def parse_markdown_to_flowables(md_path, styles):
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()

    flowables = []
    
    # Pre-process bold and italic formatting using HTML-like tags for ReportLab
    # Bold **text** -> <b>text</b>
    content = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', content)
    # Italic *text* -> <i>text</i>
    content = re.sub(r'\*(.*?)\*', r'<i>\1</i>', content)

    lines = content.split('\n')
    in_code_block = False
    code_lines = []
    in_table = False
    table_rows = []

    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Handle Code Blocks
        if line.strip().startswith('```'):
            if in_code_block:
                # End of code block
                code_text = '\n'.join(code_lines)
                code_style = ParagraphStyle(
                    'CodeStyle',
                    parent=styles['Normal'],
                    fontName='Courier',
                    fontSize=9,
                    leading=11,
                    textColor=colors.HexColor('#222222'),
                    backColor=colors.HexColor('#F4F4F5'),
                    borderPadding=8,
                    borderRadius=4,
                    spaceAfter=12
                )
                # Escape HTML chars in code block to avoid ReportLab parsing error
                code_text = code_text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                flowables.append(Paragraph(code_text.replace('\n', '<br/>'), code_style))
                in_code_block = False
                code_lines = []
            else:
                in_code_block = True
            i += 1
            continue

        if in_code_block:
            code_lines.append(line)
            i += 1
            continue

        # Handle Tables
        if line.strip().startswith('|'):
            # If it's a table alignment row (e.g. | :--- | :--- |), skip it
            if '---' in line:
                i += 1
                continue
            
            in_table = True
            # Parse row columns
            cols = [col.strip() for col in line.split('|')[1:-1]]
            table_rows.append(cols)
            i += 1
            continue
        elif in_table:
            # Table ended, render it
            if table_rows:
                # Compile ReportLab table
                table_data = []
                for row_idx, row in enumerate(table_rows):
                    processed_row = []
                    for col in row:
                        if row_idx == 0:
                            style = styles['TableHeaderStyle']
                        else:
                            style = styles['TableBodyStyle']
                        processed_row.append(Paragraph(col, style))
                    table_data.append(processed_row)

                # 4 columns with explicit width proportions
                col_widths = [160, 110, 110, 80]
                t = Table(table_data, colWidths=col_widths)
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E293B')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('TOPPADDING', (0, 0), (-1, -1), 8),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                    ('LEFTPADDING', (0, 0), (-1, -1), 8),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 8),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')]),
                    ('LINEBELOW', (0, 0), (-1, 0), 2, colors.HexColor('#0F172A')),
                    ('LINEBELOW', (0, 1), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
                ]))
                flowables.append(t)
                flowables.append(Spacer(1, 12))
            
            in_table = False
            table_rows = []
            # Do not continue, process the current line as normal

        # Handle Horizontal Rule
        if line.strip() == '---':
            flowables.append(Spacer(1, 8))
            flowables.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#CBD5E1'), spaceAfter=15))
            i += 1
            continue

        # Handle Headers
        if line.startswith('# '):
            text = line[2:].strip()
            flowables.append(Paragraph(text, styles['CustomTitle']))
            flowables.append(Spacer(1, 15))
            i += 1
            continue
        elif line.startswith('## '):
            text = line[3:].strip()
            flowables.append(Paragraph(text, styles['CustomH1']))
            flowables.append(Spacer(1, 10))
            i += 1
            continue
        elif line.startswith('### '):
            text = line[4:].strip()
            flowables.append(Paragraph(text, styles['CustomH2']))
            flowables.append(Spacer(1, 8))
            i += 1
            continue

        # Handle Bullet Points
        if line.strip().startswith('*') or line.strip().startswith('-'):
            # Check if bullet matches a numbered list pattern or bullet
            bullet_text = line.strip()[1:].strip()
            # If it starts with a strong label like **Quality:** (which is now <b>Quality:</b>)
            flowables.append(Paragraph(bullet_text, styles['BulletStyle']))
            i += 1
            continue
        
        # Handle Numbered Lists
        match_num = re.match(r'^\d+\.\s+(.*)', line.strip())
        if match_num:
            num_text = match_num.group(1)
            # Prefix with bold number
            prefix = line.strip().split('.')[0]
            flowables.append(Paragraph(f"<b>{prefix}.</b> {num_text}", styles['BulletStyle']))
            i += 1
            continue

        # Handle Normal Paragraphs
        if line.strip():
            flowables.append(Paragraph(line.strip(), styles['CustomBody']))
            flowables.append(Spacer(1, 8))
            
        i += 1

    return flowables

def generate_pdf(md_path, pdf_path):
    # Base layout configurations
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        rightMargin=54,  # 0.75 in
        leftMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()
    
    # Custom, premium typographic styles
    custom_styles = {
        'Normal': styles['Normal'],
        'CustomTitle': ParagraphStyle(
            'CustomTitle',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=22,
            leading=26,
            textColor=colors.HexColor('#0F172A'),
            spaceAfter=15,
            alignment=TA_CENTER
        ),
        'CustomH1': ParagraphStyle(
            'CustomH1',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=14,
            leading=18,
            textColor=colors.HexColor('#1E293B'),
            spaceBefore=14,
            spaceAfter=8,
            keepWithNext=True
        ),
        'CustomH2': ParagraphStyle(
            'CustomH2',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=11,
            leading=14,
            textColor=colors.HexColor('#334155'),
            spaceBefore=10,
            spaceAfter=6,
            keepWithNext=True
        ),
        'CustomBody': ParagraphStyle(
            'CustomBody',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=10,
            leading=14,
            textColor=colors.HexColor('#334155'),
            spaceAfter=8,
            alignment=TA_LEFT
        ),
        'BulletStyle': ParagraphStyle(
            'BulletStyle',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=9.5,
            leading=13.5,
            textColor=colors.HexColor('#334155'),
            leftIndent=15,
            firstLineIndent=-10,
            spaceAfter=6
        ),
        'TableHeaderStyle': ParagraphStyle(
            'TableHeaderStyle',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=9,
            leading=11,
            textColor=colors.white
        ),
        'TableBodyStyle': ParagraphStyle(
            'TableBodyStyle',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=8.5,
            leading=11,
            textColor=colors.HexColor('#334155')
        )
    }

    # Parse and compile Markdown flowables
    flowables = parse_markdown_to_flowables(md_path, custom_styles)

    # Footer layout function
    def add_page_decorations(canvas, doc):
        canvas.saveState()
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(colors.HexColor('#64748B'))
        
        # Header rule and text
        canvas.drawString(54, 750, "Technical Report: Gemma 4 Multimodal Companion")
        canvas.setStrokeColor(colors.HexColor('#E2E8F0'))
        canvas.setLineWidth(0.5)
        canvas.line(54, 742, 558, 742)
        
        # Footer rule and page numbers
        canvas.line(54, 54, 558, 54)
        canvas.drawString(54, 42, "CONFIDENTIAL - PROPRIETARY INTERNAL STRATEGY")
        canvas.drawRightString(558, 42, f"Page {doc.page}")
        canvas.restoreState()

    # Build the document beautifully
    doc.build(flowables, onFirstPage=add_page_decorations, onLaterPages=add_page_decorations)
    print(f"Successfully generated executive PDF report at: {pdf_path}")

if __name__ == "__main__":
    generate_pdf(
        "/Users/vasu/Documents/GitHub/gemma4-ira-companion/final_report.md",
        "/Users/vasu/Documents/GitHub/gemma4-ira-companion/final_report.pdf"
    )
