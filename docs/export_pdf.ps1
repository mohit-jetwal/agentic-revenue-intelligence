# Export a generated .docx to PDF, updating the table of contents first.
#
# Word rather than a pure-Python converter because the TOC is a *field*: only a
# Word engine can resolve it to real page numbers. A library export would ship a
# PDF whose contents page was blank, which is worse than no contents page - it
# looks like a defect rather than an omission.
#
# Usage:
#   .\docs\export_pdf.ps1 docs\Agentic_Revenue_Intelligence_Project_Bible.docx

param(
    [Parameter(Mandatory = $true)]
    [string]$Path
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path $Path)) {
    Write-Error "not found: $Path"
    exit 1
}

$source = (Resolve-Path $Path).Path
$target = $source -replace '\.docx$', '.pdf'

$word = New-Object -ComObject Word.Application
$word.Visible = $false

try {
    # ReadOnly, no add-ins prompt.
    $document = $word.Documents.Open($source, $false, $true)

    # Fields first, then the TOC objects: a TOC that has never been updated has
    # no entries for Fields.Update() to refresh.
    try { $document.Fields.Update() | Out-Null } catch { }
    foreach ($toc in $document.TablesOfContents) { $toc.Update() }

    # 17 = wdExportFormatPDF
    $document.ExportAsFixedFormat($target, 17)
    $document.Close($false)
    Write-Host "wrote $target"
}
finally {
    $word.Quit()
}
