# 🎯 Shooting Analysis Dashboard

**Interactive web dashboard for analyzing shooting competition data from ISSF Results Books**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Data Format](#data-format)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

The Shooting Analysis Dashboard is a comprehensive analytics platform designed to extract, process, and visualize shooting competition data from ISSF (International Shooting Sport Federation) result PDFs. Built with Streamlit, it provides coaches, athletes, and analysts with powerful tools to understand performance patterns, identify trends, and make data-driven decisions.

### Key Capabilities

- **Automated PDF Parsing**: Extract structured data from ISSF Results Books
- **Advanced Analytics**: Calculate 25+ performance metrics including consistency scores, pressure performance, and shot quality
- **Interactive Visualizations**: Explore data through dynamic charts and heatmaps using Plotly
- **Cross-Tournament Analysis**: Track athlete performance across multiple competitions
- **Country Comparisons**: Analyze success rates and performance by country
- **Shot-by-Shot Analysis**: Detailed finals performance with running totals and elimination tracking

---

## ✨ Features

### 📊 Six Comprehensive Tabs

#### 1. **Overview**
- Dashboard metrics: Total athletes, qualification rate, average scores
- Top performers in qualification and finals
- Tournament comparison statistics
- Score distribution analysis

#### 2. **Qualification Analysis**
- Complete qualification results with all series scores
- Series consistency heatmap for top athletes
- Performance metrics: consistency score, distance from cutoff, percentile rank
- Qualification trends and patterns

#### 3. **Finals Analysis**
- Finals results with shot quality metrics
- Shot quality distribution (Excellent ≥10.5, GQS 10.0-10.4, Average, Poor)
- Pressure performance indicators
- Stage-wise performance breakdown

#### 4. **Athlete Deep Dive**
- Individual athlete performance summary
- Tournament history with detailed statistics
- Finals shot sequence visualization
- Recovery pattern analysis after poor shots

#### 5. **Cross-Tournament**
- Multi-tournament athlete tracking
- Performance progression trends
- Score improvement analysis
- Comparative athlete performance charts

#### 6. **Insights**
- Key statistics and recommendations
- Country-wise performance rankings
- Tournament difficulty analysis
- Automated performance insights

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (optional, for cloning)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/shooting-analysis-dashboard.git
cd shooting-analysis-dashboard

# Create virtual environment (recommended)
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Dashboard

```bash
# Run the Streamlit app
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

---

## 📦 Installation

### Method 1: Using pip (Recommended)

```bash
pip install -r requirements.txt
```

### Method 2: Manual Installation

```bash
pip install streamlit>=1.28.0
pip install pandas>=2.0.0
pip install numpy>=1.24.0
pip install plotly>=5.17.0
pip install pdfplumber>=0.10.0
pip install python-dateutil>=2.8.2
pip install openpyxl>=3.1.0
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| streamlit | ≥1.28.0 | Web dashboard framework |
| pandas | ≥2.0.0 | Data manipulation |
| numpy | ≥1.24.0 | Numerical computing |
| plotly | ≥5.17.0 | Interactive visualizations |
| pdfplumber | ≥0.10.0 | PDF text extraction |
| python-dateutil | ≥2.8.2 | Date parsing utilities |
| openpyxl | ≥3.1.0 | Excel export support |

---

## 💻 Usage

### 1. Preparing Your Data

Place your ISSF Results PDF files in the `data/raw/` directory:

```
shooting-analysis-dashboard/
├── data/
│   ├── raw/                          # Place PDF files here
│   │   ├── 10mAR_MW_ASC-Shymkent-2025.pdf
│   │   ├── 10mAR_MW_WC-RP-Munich-2025.pdf
│   │   └── ...
│   └── processed/                    # Auto-generated CSV files
│       ├── qualifications.csv
│       └── finals.csv
```

### 2. PDF Filename Format

For best results, name your PDF files using this format:

```
{Event}_{Gender}_{Tournament}-{Location}-{Year}.pdf
```

Examples:
- `10mAR_MW_ASC-Shymkent-2025.pdf` → Asian Championship - Shymkent 2025
- `10mAR_MW_WC-RP-Munich-2025.pdf` → World Cup - Munich 2025
- `10mAR_MW_GP-10m-Ruse-2025.pdf` → Grand Prix - Ruse 2025

**Tournament Codes:**
- `ASC` = Asian Championship
- `ECH` = European Championship
- `WC` = World Cup
- `GP` = Grand Prix

### 3. Running the Parser Manually

```bash
# Run the data parser to extract data from PDFs
python data_parser.py
```

This will:
1. Scan `data/raw/` for PDF files
2. Extract qualification and finals data
3. Save processed CSV files to `data/processed/`
4. Display extraction statistics

### 4. Using the Dashboard

```bash
# Launch the Streamlit dashboard
streamlit run app.py
```

**Dashboard Features:**
- **Automatic Data Loading**: Loads processed CSV files automatically
- **Auto-Parsing**: If no CSV files exist, automatically parses PDFs
- **Interactive Filters**: Filter by tournament, gender, country
- **Real-time Updates**: Changes reflect immediately
- **Export Options**: Download filtered data as CSV

### 5. Filtering Data

Use the sidebar to filter data:
- **Tournaments**: Select specific competitions
- **Gender**: Filter by Men/Women
- **Country**: Focus on specific countries (e.g., IND, USA, CHN)

---

## 📁 Project Structure

```
shooting-analysis-dashboard/
│
├── app.py                          # Main Streamlit dashboard application
├── data_parser.py                  # PDF parsing and data extraction
├── analytics_engine.py             # Performance metrics calculation
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── .gitignore                      # Git ignore rules
│
├── data/
│   ├── raw/                        # Input: ISSF PDF files
│   ├── processed/                  # Output: Extracted CSV files
│   │   ├── qualifications.csv      # Qualification data
│   │   └── finals.csv              # Finals data
│   └── exports/                    # User exports
│
└── docs/
    └── deployment_guide.md         # Deployment instructions
```

### Core Files

#### **app.py**
Main Streamlit application with 6 tabs:
- Data loading and caching
- Interactive visualizations
- Filtering and drill-down capabilities
- Export functionality

#### **data_parser.py**
PDF parsing engine (v5.6):
- Extracts text from ISSF Results PDFs using pdfplumber
- Parses qualification results (6 series + totals)
- Parses finals results (shot-by-shot data)
- Handles multiple events, genders, and tournaments
- Auto-detects competition metadata from filenames

**Key Features:**
- Series-anchored NOC detection algorithm
- Auto gender-switch detection
- Robust error handling
- Supports merged name-NOC fields

#### **analytics_engine.py**
Performance metrics engine (v2.0):
- Calculates 25+ qualification metrics
- Analyzes finals shot quality
- Tracks cross-tournament progression
- Generates country success rates
- Provides automated insights

**Metrics Calculated:**
- Consistency scores (series standard deviation)
- Distance from qualification cutoff
- Percentile rankings
- Shot quality percentages (GQS, Excellent, Poor)
- Pressure performance (stage comparison)
- Recovery patterns after poor shots
- Running totals and elimination progression

---

## 📊 Data Format

### Input: PDF Files

The parser supports ISSF Results Book PDFs with:
- Qualification tables: Rank, Bib, Name, NOC, 6 series, Total
- Finals tables: Rank, Bib, Name, NOC, shots, stage totals

### Output: CSV Files

#### **qualifications.csv**
```csv
tournament,full_tournament_name,source_file,event,gender,stage,date,rank,bib,name,noc,series_1,series_2,series_3,series_4,series_5,series_6,total,qualified,remarks
Asian Championship,Asian Championship - Shymkent 2025,10mAR_MW_ASC-Shymkent-2025.pdf,10m Air Rifle,Men,Qualification,21 AUG 2025,1,1234,John Doe,USA,105.2,104.8,105.5,104.9,105.3,105.1,630.8,True,Q
```

**Columns:**
- `tournament`: Simplified tournament name (e.g., "World Cup")
- `full_tournament_name`: Full name with location (e.g., "World Cup - Munich 2025")
- `series_1` to `series_6`: Individual series scores (out of ~110)
- `total`: Total qualification score (600-660)
- `qualified`: Boolean - Top 8 qualification
- `remarks`: Q (Qualified), WR (World Record), etc.

#### **finals.csv**
```csv
tournament,full_tournament_name,source_file,event,gender,stage,date,rank,bib,name,noc,total,num_shots,stage_totals,shots,remarks
World Cup,World Cup - Munich 2025,10mAR_MW_WC-RP-Munich-2025.pdf,10m Air Rifle,Men,Final,23 AUG 2025,1,1234,John Doe,USA,252.8,24,103.5,208.3,252.8,10.5,10.3,10.4,...,WR
```

**Columns:**
- `total`: Final score
- `num_shots`: Number of shots fired
- `stage_totals`: Cumulative stage scores
- `shots`: Individual shot values (comma-separated)
- `remarks`: WR (World Record), AsR (Asian Record), ER (European Record)

---

## 🌐 Deployment

### Streamlit Community Cloud (Free)

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/yourusername/shooting-dashboard.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to https://share.streamlit.io
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set main file: `app.py`
   - Click "Deploy!"

3. **Share Your App**
   Your app will be live at: `https://your-app-name.streamlit.app`

### Other Deployment Options

- **Heroku**: `heroku create && git push heroku main`
- **Google Cloud Run**: See `docs/deployment_guide.md`
- **AWS EC2**: Deploy using Docker container
- **Local Network**: Run with `streamlit run app.py --server.address 0.0.0.0`

---

## 📈 Performance Metrics

### Qualification Metrics
- **Consistency Score**: Standard deviation of 6 series scores (lower = more consistent)
- **Distance from Cutoff**: Points above/below 8th place cutoff score
- **Percentile Rank**: Position within the tournament field (%)
- **Series Pattern**: Trend analysis (Improving/Declining/Stable)
- **Weak Series Count**: Number of series significantly below average

### Finals Metrics
- **GQS Percentage**: Shots ≥10.0 (Good Quality Shot threshold)
- **Excellent Percentage**: Shots ≥10.5
- **Pressure Performance**: Stage 2 avg - Stage 1 avg
- **Shot Consistency**: Standard deviation of all shots
- **Recovery Success Rate**: Ability to recover after poor shots (<9.5)
- **Elimination Performance**: Shot quality in critical elimination rounds

### Cross-Tournament Metrics
- **Score Progression**: Improvement/decline across tournaments
- **Rank Progression**: Position changes over time
- **Multi-Tournament Average**: Weighted performance across events
- **Improvement Rate**: Points gained per tournament

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

### Reporting Issues
- Use GitHub Issues to report bugs
- Include sample PDF file (if possible)
- Describe expected vs actual behavior
- Include error messages and screenshots

### Feature Requests
- Open a GitHub Issue with [Feature Request] tag
- Describe the feature and use case
- Explain how it would benefit users

### Pull Requests
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guide
- Add docstrings to new functions
- Test with multiple PDF formats
- Update README if adding features

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 Shooting Analysis Dashboard Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

## 🙏 Acknowledgments

- **ISSF** (International Shooting Sport Federation) for standardized results format
- **Streamlit** team for the amazing dashboard framework
- **pdfplumber** contributors for PDF parsing capabilities
- **Plotly** team for interactive visualization library
- All contributors and users who provide feedback

---

## 📞 Support

- **Documentation**: See `docs/` folder for detailed guides
- **Issues**: Report bugs on [GitHub Issues](https://github.com/yourusername/shooting-dashboard/issues)
- **Discussions**: Ask questions in [GitHub Discussions](https://github.com/yourusername/shooting-dashboard/discussions)
- **Email**: your.email@example.com

---

## 🗺️ Roadmap

### Version 2.0 (Q2 2025)
- [ ] Multi-event support (50m, 25m Pistol)
- [ ] Historical data comparison (year-over-year)
- [ ] Advanced ML predictions
- [ ] Mobile-responsive design improvements

### Version 2.1 (Q3 2025)
- [ ] Real-time data streaming
- [ ] Live competition tracking
- [ ] Athlete ranking system
- [ ] Export to PowerPoint/PDF reports

### Future Considerations
- [ ] Integration with ISSF API (if available)
- [ ] Multi-language support
- [ ] Custom metric creation
- [ ] Team performance analysis
- [ ] Video analysis integration

---

## 📊 Statistics

![GitHub repo size](https://img.shields.io/github/repo-size/yourusername/shooting-dashboard)
![GitHub stars](https://img.shields.io/github/stars/yourusername/shooting-dashboard?style=social)
![GitHub forks](https://img.shields.io/github/forks/yourusername/shooting-dashboard?style=social)

---

## 🌟 Star History

If you find this project useful, please consider giving it a ⭐ on GitHub!

---

**Built with ❤️ for the shooting sports community**

---

*Last updated: October 2025*
