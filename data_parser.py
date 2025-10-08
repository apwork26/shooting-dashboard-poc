"""
NRAI Shooting Analysis Dashboard - Data Parser
==============================================
Based on ISSF Results Parser v5.6 - Adapted for README structure

Changes from issf_parser_v5.6.py:
- Class name: ISSFResultsParser → DataParser
- Export filenames: Simplified (no timestamps)
- Default directory: 'data/raw'
- Export directory: 'data/processed'

All parsing logic remains IDENTICAL!
"""

import re
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

class DataParser:
    """Data Parser v5.6 - Based on ISSF Parser v5.6"""

    VERSION = "5.6"

    def __init__(self, pdf_directory: str = 'data/raw'):
        self.pdf_directory = Path(pdf_directory)
        self.last_comp_info = None

        # Auto-create directory structure
        self.pdf_directory.mkdir(parents=True, exist_ok=True)
        Path('data/processed').mkdir(parents=True, exist_ok=True)
        Path('data/exports').mkdir(parents=True, exist_ok=True)

    def extract_text_from_pdf(self, pdf_path: Path) -> List[str]:
        """Extract text page by page."""
        if not PDF_AVAILABLE:
            raise ImportError("Install: pip install pdfplumber")

        pages_text = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        pages_text.append(text)
        except Exception as e:
            print(f"  Error reading PDF: {e}")

        return pages_text

    def detect_tournament_from_filename(self, filename: str) -> str:
        """Extract tournament from filename."""
        filename_upper = filename.upper()

        if 'ASC' in filename_upper or 'ASIAN' in filename_upper:
            return 'Asian Championship'
        elif 'ECH' in filename_upper or 'EUROPEAN' in filename_upper:
            return 'European Championship'
        elif 'WC' in filename_upper:
            return 'World Cup'
        elif 'GP' in filename_upper:
            return 'Grand Prix'

        return 'Unknown'

    def detect_competition_info(self, text: str, pdf_filename: str, use_last: bool = True) -> Dict[str, str]:
        """Extract competition metadata - SIMPLE VERSION using filename"""

        # SIMPLE: Use PDF filename as tournament name
        tournament_name = pdf_filename.replace('.pdf', '').replace('.PDF', '')

        # Parse filename format: "10mAR_MW_ASC-Shymkent-2025"
        parts = tournament_name.split('_')

        # Extract readable tournament name from filename
        if len(parts) >= 3:
            # parts[2] = "ASC-Shymkent-2025" or "WC-RP-Munich-2025"
            tournament_part = parts[2]
            segments = tournament_part.split('-')

            # Get tournament type
            tournament_type = segments[0]
            location = segments[1] if len(segments) > 1 else 'Unknown'
            year = segments[-1] if len(segments) > 0 and segments[-1].isdigit() else '2025'

            # Create clean tournament name
            type_map = {
                'ASC': 'Asian Championship',
                'ECH': 'European Championship',
                'WC': 'World Cup',
                'GP': 'Grand Prix'
            }

            comp_type = type_map.get(tournament_type, tournament_type)
            full_name = f"{comp_type} - {location} {year}"
        else:
            comp_type = 'Unknown'
            full_name = tournament_name
            location = 'Unknown'

        info = {
            'competition': comp_type,
            'full_title': full_name,
            'location': location,
            'event': 'Unknown',
            'stage': 'Unknown',
            'date': 'Unknown',
            'gender': 'Unknown'
        }

        lines = text.split('\n')[:50]

        # Extract event
        for line in lines:
            line_upper = line.upper()
            if '10M AIR RIFLE' in line_upper or '10 M AIR RIFLE' in line_upper:
                info['event'] = '10m Air Rifle'
                break
            elif '10M AIR PISTOL' in line_upper:
                info['event'] = '10m Air Pistol'
                break

        # Extract gender
        for line in lines:
            line_upper = line.upper()
            if 'MEN' in line_upper and 'WOMEN' not in line_upper:
                info['gender'] = 'Men'
            elif 'WOMEN' in line_upper:
                info['gender'] = 'Women'

        # Extract stage
        for line in lines:
            line_upper = line.upper()
            if 'QUALIFICATION' in line_upper and 'FINAL' not in line_upper:
                info['stage'] = 'Qualification'
                break
            elif 'FINAL' in line_upper and 'QUALIFICATION' not in line_upper:
                info['stage'] = 'Final'
                break

        # Fill from last comp info if needed
        if use_last and self.last_comp_info:
            for key in ['event', 'gender']:
                if info[key] == 'Unknown':
                    info[key] = self.last_comp_info.get(key, 'Unknown')

        # Save for next page
        if info['event'] != 'Unknown':
            self.last_comp_info = info.copy()

        return info

    def parse_qualification(self, text: str, pdf_name: str) -> List[Dict]:
        """Parse qualification."""
        records = []
        comp_info = self.detect_competition_info(text, pdf_name)

        if comp_info['stage'] != 'Qualification':
            return records

        lines = text.split('\n')

        # Find table start
        table_start = -1
        for i, line in enumerate(lines):
            if 'Rank' in line and 'Bib' in line and 'Name' in line:
                table_start = i + 1
                break

        if table_start == -1:
            return records

        # Parse qualification lines
        last_rank = 0
        for i in range(table_start, len(lines)):
            line = lines[i].strip()

            # Stop conditions
            if not line or 'Summary' in line or 'Legend' in line or 'Number of athletes' in line:
                break

            record = self._parse_qual_line_v53(line, comp_info, pdf_name)

            if record:
                # Auto-detect gender switch (rank drops from >10 to 1)
                if record['rank'] == 1 and last_rank > 10:
                    if comp_info['gender'] == 'Men':
                        comp_info = comp_info.copy()
                        comp_info['gender'] = 'Women'
                        self.last_comp_info = comp_info.copy()
                    elif comp_info['gender'] == 'Women':
                        comp_info = comp_info.copy()
                        comp_info['gender'] = 'Men'
                        self.last_comp_info = comp_info.copy()

                # Apply current metadata
                record['tournament'] = comp_info['competition']
                record['event'] = comp_info['event']
                record['gender'] = comp_info['gender']
                record['date'] = comp_info['date']

                records.append(record)
                last_rank = record['rank']

        return records

    def _parse_qual_line_v53(self, line: str, comp_info: Dict, pdf_name: str) -> Optional[Dict]:
        """Parse qualification line - v5.3 (series-anchored NOC detection)."""
        parts = line.split()
        if len(parts) < 10:
            return None

        try:
            # 1. Extract rank
            if not parts[0].isdigit() or int(parts[0]) >= 200:
                return None
            rank = int(parts[0])

            # 2. Find first series score (100-110 range) - ANCHOR POINT
            series_start_idx = -1
            for i in range(2, len(parts)):
                try:
                    val = float(parts[i])
                    if 95.0 <= val <= 110.0:
                        series_start_idx = i
                        break
                except:
                    continue

            if series_start_idx == -1 or series_start_idx < 3:
                return None

            # 3. NOC is immediately before series
            noc_idx = series_start_idx - 1
            noc_token = parts[noc_idx]

            # 4. Handle merged NOC (e.g., "BalasahebIND")
            noc_match = re.search(r'([A-Z]{3})$', noc_token)
            if noc_match:
                noc = noc_match.group(1)
                if len(noc_token) > 3:
                    name_suffix = noc_token[:-3]
                    bib_idx = 1
                    name_parts = parts[2:noc_idx] + [name_suffix]
                else:
                    bib_idx = 1
                    name_parts = parts[2:noc_idx]
            else:
                return None

            # 5. Extract bib
            if not parts[bib_idx].isdigit():
                return None
            bib = int(parts[bib_idx])

            # 6. Construct name
            name = ' '.join(name_parts) if name_parts else "Unknown"

            # 7. Extract 6 series scores
            series = []
            for i in range(series_start_idx, series_start_idx + 6):
                if i < len(parts):
                    try:
                        val = float(parts[i])
                        if 95.0 <= val <= 110.0:
                            series.append(val)
                    except:
                        break

            if len(series) != 6:
                return None

            # 8. Extract total
            total = sum(series)
            if series_start_idx + 6 < len(parts):
                try:
                    val = float(parts[series_start_idx + 6])
                    if 600.0 <= val <= 660.0:
                        total = val
                except:
                    pass

            # 9. Extract remarks
            remarks = ''
            if series_start_idx + 7 < len(parts):
                remarks_text = parts[series_start_idx + 7].upper()
                if 'Q' in remarks_text:
                    remarks = 'Q'
                elif 'RPO' in remarks_text:
                    remarks = 'RPO'
                elif 'WR' in remarks_text:
                    remarks = 'WR'

            qualified = 'Q' in remarks

            return {
                'tournament': comp_info['competition'],
                'full_tournament_name': comp_info.get('full_title', comp_info['competition']),
                'source_file': pdf_name,
                'event': comp_info['event'],
                'gender': comp_info['gender'],
                'stage': 'Qualification',
                'date': comp_info['date'],
                'rank': rank,
                'bib': bib,
                'name': name,
                'noc': noc,
                'series_1': series[0],
                'series_2': series[1],
                'series_3': series[2],
                'series_4': series[3],
                'series_5': series[4],
                'series_6': series[5],
                'total': total,
                'qualified': qualified,
                'remarks': remarks
            }

        except Exception:
            return None

    def parse_finals(self, text: str, pdf_name: str) -> List[Dict]:
        """Parse finals."""
        records = []
        comp_info = self.detect_competition_info(text, pdf_name)

        if comp_info['stage'] != 'Final':
            return records

        lines = text.split('\n')
        table_start = -1

        for i, line in enumerate(lines):
            if 'Rk' in line and 'Name' in line and ('NOC' in line or 'Nat' in line):
                table_start = i + 2
                break

        if table_start == -1:
            return records

        i = table_start
        while i < len(lines) - 5:
            line = lines[i].strip()

            if not line or 'Legend' in line or 'Summary' in line:
                break

            first_token = line.split()[0] if line.split() else ""

            if first_token.isdigit() and 1 <= int(first_token) <= 8:
                athlete_lines = []
                for j in range(6):
                    if i + j < len(lines):
                        athlete_lines.append(lines[i + j])

                athlete_data = self._parse_finals_athlete(athlete_lines, comp_info, pdf_name)
                if athlete_data:
                    records.append(athlete_data)

                i += 6
            else:
                i += 1

        return records

    def _parse_finals_athlete(self, lines: List[str], comp_info: Dict, pdf_name: str) -> Optional[Dict]:
        """Parse finals athlete."""
        try:
            if not lines or len(lines) < 2:
                return None

            line0_parts = lines[0].split()
            if len(line0_parts) < 5:
                return None

            rank = int(line0_parts[0])
            bib = int(line0_parts[1])

            # Find NOC
            noc_idx = -1
            for i in range(2, min(len(line0_parts), 10)):
                token = line0_parts[i]
                if len(token) == 3 and token.isupper() and token.isalpha():
                    noc_idx = i
                    break

            if noc_idx == -1:
                return None

            noc = line0_parts[noc_idx]
            lastname = ' '.join(line0_parts[2:noc_idx])

            line1_parts = lines[1].split()
            firstname = line1_parts[0] if line1_parts else ""
            name = f"{lastname} {firstname}".strip()

            # Extract numbers from line0
            numbers_line0 = []
            for token in line0_parts[noc_idx + 1:]:
                try:
                    numbers_line0.append(float(token))
                except:
                    pass

            stage_totals = [n for n in numbers_line0 if 50 < n < 300]
            total = stage_totals[-1] if stage_totals else 0

            # Extract shots
            shot_matrix = []
            for line_idx in range(1, min(len(lines), 6)):
                tokens = lines[line_idx].split()
                start_idx = 1 if line_idx == 1 else 0
                row_shots = []

                for token in tokens[start_idx:]:
                    try:
                        val = float(token)
                        if 9.0 <= val <= 11.0:
                            row_shots.append(val)
                    except:
                        pass

                shot_matrix.append(row_shots)

            # Order shots column by column
            max_cols = max(len(row) for row in shot_matrix) if shot_matrix else 0
            ordered_shots = []
            for col in range(max_cols):
                for row in shot_matrix:
                    if col < len(row):
                        ordered_shots.append(row[col])

            # Extract remarks
            remarks = ''
            full_text = ' '.join(lines)
            if 'WR' in full_text and 'Records' not in full_text:
                remarks = 'WR'
            elif 'AsR' in full_text:
                remarks = 'AsR'
            elif 'ER' in full_text:
                remarks = 'ER'

            return {
                'tournament': comp_info['competition'],
                'full_tournament_name': comp_info.get('full_title', comp_info['competition']),
                'source_file': pdf_name,
                'event': comp_info['event'],
                'gender': comp_info['gender'],
                'stage': 'Final',
                'date': comp_info['date'],
                'rank': rank,
                'bib': bib,
                'name': name,
                'noc': noc,
                'total': total,
                'num_shots': len(ordered_shots),
                'stage_totals': ','.join(map(str, stage_totals)),
                'shots': ','.join(map(str, ordered_shots)),
                'remarks': remarks
            }

        except Exception:
            return None

    def parse_single_pdf(self, pdf_path: Path) -> Tuple[List[Dict], List[Dict]]:
        """Parse single PDF."""
        print(f"  Processing: {pdf_path.name}")

        # Reset metadata for each new PDF
        self.last_comp_info = None

        pages = self.extract_text_from_pdf(pdf_path)
        if not pages:
            return [], []

        qual_records = []
        finals_records = []

        for page_text in pages:
            qual = self.parse_qualification(page_text, pdf_path.name)
            qual_records.extend(qual)

            finals = self.parse_finals(page_text, pdf_path.name)
            finals_records.extend(finals)

        print(f"  → Qual: {len(qual_records)} | Finals: {len(finals_records)}")

        return qual_records, finals_records

    def parse_all_pdfs(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Parse all PDFs."""
        print(f"\n{'='*60}")
        print(f"Data Parser v{self.VERSION} (based on ISSF Parser)")
        print("="*60)
        print(f"Directory: {self.pdf_directory}")

        pdf_files = list(self.pdf_directory.glob("*.pdf"))

        if not pdf_files:
            print("\nNo PDF files found")
            return pd.DataFrame(), pd.DataFrame()

        print(f"Found {len(pdf_files)} PDF files\n")

        all_qual = []
        all_finals = []

        for pdf_file in pdf_files:
            qual, finals = self.parse_single_pdf(pdf_file)
            all_qual.extend(qual)
            all_finals.extend(finals)

        qual_df = pd.DataFrame(all_qual)
        finals_df = pd.DataFrame(all_finals)

        print(f"\n{'='*60}")
        print("EXTRACTION COMPLETE")
        print("="*60)
        print(f"Qualification: {len(qual_df)} records")
        print(f"Finals: {len(finals_df)} records")

        if len(qual_df) > 0:
            print(f"\nQual → {qual_df['name'].nunique()} athletes | {qual_df['noc'].nunique()} countries")
            if 'gender' in qual_df.columns:
                gender_counts = qual_df['gender'].value_counts()
                print(f"  Men: {gender_counts.get('Men', 0)} | Women: {gender_counts.get('Women', 0)}")
            if 'event' in qual_df.columns:
                unknown_events = len(qual_df[qual_df['event'] == 'Unknown'])
                if unknown_events == 0:
                    print(f"  ✅ NO Unknown entries!")
                else:
                    print(f"  ⚠️  {unknown_events} Unknown entries")

        if len(finals_df) > 0:
            print(f"Finals → {finals_df['name'].nunique()} athletes | {finals_df['noc'].nunique()} countries")

        return qual_df, finals_df

    def export_results(self, qual_df: pd.DataFrame, finals_df: pd.DataFrame,
                      output_dir: str = "data/processed"):
        """Export results to data/processed/ directory."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"\nExporting results...")

        if len(qual_df) > 0:
            qual_file = output_path / "qualifications.csv"
            qual_df.to_csv(qual_file, index=False, encoding='utf-8-sig')
            print(f"  ✓ {qual_file.name}")

        if len(finals_df) > 0:
            finals_file = output_path / "finals.csv"
            finals_df.to_csv(finals_file, index=False, encoding='utf-8-sig')
            print(f"  ✓ {finals_file.name}")

        print(f"\nLocation: {output_path.absolute()}")
        print(f"\n{'='*60}")
        print(f"✅ Data Parser v{self.VERSION} Complete!")
        print("="*60)


def main():
    """Main execution."""
    if not PDF_AVAILABLE:
        print("ERROR: pdfplumber not installed")
        print("Run: pip install pdfplumber")
        return

    PDF_DIR = "data/raw"
    OUTPUT_DIR = "data/processed"

    parser = DataParser(PDF_DIR)
    qual_df, finals_df = parser.parse_all_pdfs()

    if len(qual_df) > 0 or len(finals_df) > 0:
        parser.export_results(qual_df, finals_df, OUTPUT_DIR)
    else:
        print("\nNo data extracted")


if __name__ == "__main__":
    main()
