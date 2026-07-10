"""
symbols.py — Default PSX universes for the scanner.

KSE100 below is a reasonably current constituent list; index composition is
rebalanced periodically, so the app exposes it as an editable text area and
can also pull the full live symbol directory from dps.psx.com.pk/symbols.
"""

KSE100 = [
    "AGP", "AICL", "AIRLINK", "AKBL", "APL", "ATRL", "AVN", "BAFL", "BAHL",
    "BNWM", "BOP", "CEPB", "CHCC", "CNERGY", "COLG", "DAWH", "DCR", "DGKC",
    "EFERT", "EFUG", "ENGRO", "EPCL", "FABL", "FATIMA", "FCCL", "FCEPL",
    "FFC", "FHAM", "GADT", "GAL", "GHGL", "GLAXO", "HBL", "HCAR",
    "HGFA", "HMB", "HUBC", "HUMNL", "IBFL", "ILP", "INDU", "INIL", "ISL",
    "JDWS", "JVDC", "KAPCO", "KEL", "KOHC", "KTML", "LCI", "LOTCHEM", "LUCK",
    "MARI", "MCB", "MEBL", "MLCF", "MTL", "MUGHAL", "MUREB", "NBP", "NESTLE",
    "NML", "NRL", "OGDC", "PABC", "PAEL", "PAKT", "PGLC", "PIBTL", "PIOC",
    "PKGP", "PKGS", "POL", "PPL", "PSEL", "PSO", "PSX", "PTC", "RMPL",
    "SAZEW", "SCBPL", "SEARL", "SHEL", "SHFA", "SNGP", "SRVI", "SSGC",
    "SYS", "TGL", "THALL", "TRG", "UBL", "UNITY", "YOUW",
]

# Compact liquid subset for quick scans
QUICK25 = [
    "HBL", "UBL", "MCB", "MEBL", "BAFL", "BAHL", "NBP", "BOP",
    "OGDC", "PPL", "POL", "MARI", "PSO", "SNGP", "SSGC", "SHEL",
    "ENGRO", "FFC", "EFERT", "LUCK", "DGKC", "MLCF", "SYS", "TRG", "HUBC",
]
