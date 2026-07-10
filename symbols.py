"""
symbols.py — PSX universes for the scanner.

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


# Full PSX main-board universe (~435 tickers). Compiled statically
# because the live symbol directory (dps.psx.com.pk/symbols) blocks cloud
# hosts. Delisted / renamed symbols fail gracefully and are skipped. Missing
# ones can always be added via the Custom universe.
ALL_PSX = ['AABS', 'ABL', 'ABOT', 'ACPL', 'ADAMS', 'ADMM', 'AGHA', 'AGIC', 'AGIL', 'AGL', 'AGP', 'AGSML', 'AGTL', 'AHCL', 'AICL', 'AIRLINK', 'AKBL', 'AKDHL', 'AKGL', 'ALAC', 'ALIFE', 'ALNRS', 'ALTN', 'AMBL', 'ANL', 'APL', 'ARPAK', 'ARPL', 'ASC', 'ASHT', 'ASIC', 'ASL', 'ASTL', 'ATBA', 'ATIL', 'ATLH', 'ATRL', 'AVN', 'AWTX', 'BAFL', 'BAHL', 'BATA', 'BCL', 'BECO', 'BELA', 'BERG', 'BFMOD', 'BGL', 'BHAT', 'BIFO', 'BILF', 'BIPL', 'BNL', 'BNWM', 'BOK', 'BOP', 'BPL', 'BTL', 'BUXL', 'BWCL', 'BWHL', 'CENI', 'CEPB', 'CHAS', 'CHCC', 'CJPL', 'CLOV', 'CNERGY', 'COLG', 'CPHL', 'CRTM', 'CSAP', 'CSIL', 'CTM', 'CWSM', 'CYAN', 'DAAG', 'DADX', 'DAWH', 'DBCI', 'DCL', 'DCR', 'DEL', 'DFML', 'DGKC', 'DIIL', 'DINT', 'DKL', 'DLL', 'DNCC', 'DOL', 'DSFL', 'DSIL', 'DSL', 'DWSM', 'DYNO', 'ECOP', 'EFERT', 'EFUG', 'EFUL', 'ELCM', 'ELSM', 'EMCO', 'ENGRO', 'EPCL', 'EPQL', 'ESBL', 'EWIC', 'EXIDE', 'FABL', 'FANM', 'FASM', 'FATIMA', 'FCCL', 'FCEL', 'FCEPL', 'FCIBL', 'FCONM', 'FCSC', 'FDPL', 'FECM', 'FECTC', 'FEM', 'FEROZ', 'FFC', 'FFL', 'FFLM', 'FHAM', 'FIBLM', 'FIL', 'FIMM', 'FLYNG', 'FML', 'FNEL', 'FPJM', 'FPRM', 'FRCL', 'FRSM', 'FTMM', 'FTSM', 'FZCM', 'GADT', 'GAL', 'GAMON', 'GATM', 'GEMBLUE', 'GEMPAPL', 'GGGL', 'GGL', 'GHGL', 'GHNI', 'GHNL', 'GLAXO', 'GLOT', 'GLPL', 'GOC', 'GRR', 'GRYL', 'GSPM', 'GTYR', 'GUSM', 'GUTM', 'GVGL', 'GWLC', 'HABSM', 'HADC', 'HAEL', 'HAFL', 'HALEON', 'HASCOL', 'HBL', 'HCAR', 'HGFA', 'HICL', 'HIFA', 'HINO', 'HINOON', 'HIRAT', 'HMB', 'HMIM', 'HPL', 'HRPL', 'HTL', 'HUBC', 'HUMNL', 'HWQS', 'IBFL', 'IBLHL', 'ICCI', 'ICIBL', 'ICL', 'IDRT', 'IDSM', 'IDYM', 'IGIHL', 'IGIL', 'ILP', 'IMAGE', 'IML', 'INDU', 'INIL', 'INKL', 'IPAK', 'ISIL', 'ISL', 'ITTEFAQ', 'JATM', 'JDMT', 'JDWS', 'JGICL', 'JKSM', 'JLICL', 'JOPP', 'JSBL', 'JSCL', 'JSGCL', 'JSIL', 'JSML', 'JUBS', 'JVDC', 'KAPCO', 'KCL', 'KEL', 'KHTC', 'KML', 'KOHC', 'KOHE', 'KOHP', 'KOHTM', 'KOIL', 'KOSM', 'KPUS', 'KSBP', 'KSTM', 'KTML', 'LCI', 'LEUL', 'LOADS', 'LOTCHEM', 'LPGL', 'LPL', 'LSECL', 'LSEFSL', 'LSEVL', 'LUCK', 'MACFL', 'MACTER', 'MARI', 'MCB', 'MCBIM', 'MDTL', 'MEBL', 'MEHT', 'MERIT', 'MFFL', 'MFL', 'MIRKS', 'MLCF', 'MODAM', 'MOHE', 'MQTM', 'MRNS', 'MSCL', 'MSOT', 'MTL', 'MUGHAL', 'MUREB', 'MWMP', 'MZNPETF', 'NAGC', 'NATF', 'NBP', 'NCL', 'NCML', 'NESTLE', 'NETSOL', 'NEXT', 'NICL', 'NITGETF', 'NML', 'NONS', 'NPL', 'NRL', 'NRSL', 'OBOY', 'OCTOPUS', 'OGDC', 'OLPL', 'OLPM', 'OML', 'ORM', 'OTSU', 'PABC', 'PACE', 'PAEL', 'PAKD', 'PAKL', 'PAKOXY', 'PAKRI', 'PAKT', 'PASL', 'PASM', 'PCAL', 'PECO', 'PGLC', 'PHDL', 'PIAHCLA', 'PIBTL', 'PICT', 'PIL', 'PIM', 'PINL', 'PIOC', 'PKGI', 'PKGP', 'PKGS', 'PMPK', 'PMRS', 'PNSC', 'POL', 'POML', 'POWER', 'PPL', 'PPP', 'PPVC', 'PREMA', 'PRL', 'PRWM', 'PSEL', 'PSO', 'PSX', 'PSYL', 'PTC', 'PTL', 'QUET', 'QUICE', 'RANI', 'RCML', 'REDCO', 'REWM', 'RICL', 'RMPL', 'RPL', 'RUBY', 'RUPL', 'SAIF', 'SANSM', 'SAPT', 'SARC', 'SASML', 'SAZEW', 'SBL', 'SCBPL', 'SCL', 'SEARL', 'SEL', 'SEPL', 'SERT', 'SFL', 'SGABL', 'SGF', 'SGPL', 'SHCM', 'SHDT', 'SHEL', 'SHEZ', 'SHFA', 'SHJS', 'SHNI', 'SHSML', 'SIBL', 'SIEM', 'SINDM', 'SITC', 'SKRS', 'SLGL', 'SLYT', 'SMCPL', 'SML', 'SNAI', 'SNBL', 'SNGP', 'SPEL', 'SPL', 'SPWL', 'SRVI', 'SSGC', 'SSML', 'STCL', 'STJT', 'STML', 'STPL', 'SUHJ', 'SURC', 'SUTM', 'SYM', 'SYS', 'SZTM', 'TATM', 'TCORP', 'TELE', 'TGL', 'THALL', 'THCCL', 'TICL', 'TOMCL', 'TOWL', 'TPL', 'TPLI', 'TPLP', 'TPLRF', 'TREET', 'TRG', 'TRIPF', 'TRSM', 'TSBL', 'TSMF', 'TSML', 'TSPL', 'UBDL', 'UBL', 'UCAPM', 'UDPL', 'UNIC', 'UNITY', 'UPFL', 'UVIC', 'WAHN', 'WAVES', 'WTL', 'YOUW', 'ZAHID', 'ZELP', 'ZIL', 'ZTL']
