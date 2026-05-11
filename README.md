#  Global Energy Production & Consumption Dashboard

**Project 9 — Data Visualization Final Project**  
**Course:** Data Visualization | **Dataset:** Our World in Data — Energy Dataset

---

##  Project Overview

This interactive Plotly Dash dashboard explores how countries around the world produce and consume energy, tracking the historic shift from fossil fuels to renewables over decades. Using the **Our World in Data Energy Dataset**, the dashboard covers 23,377 country-year records spanning 1900–2025 across 314 unique entities (countries + regional aggregates).

The dashboard enables users to:
- Compare energy consumption and renewable adoption rates across countries
- Investigate the relationship between economic development and energy use patterns
- Analyze the distribution of energy metrics like per capita consumption and carbon intensity
- Track the global energy transition from fossil fuels to renewables over time

---

##  Project Structure

```
DATA-VISUALIZATION-PROJECT/
├── charts/
│   ├── bubble_plots.py                         
│   ├── Histogram_and_BoxPlot.py              
│   ├── scatter_plots.py                     
│   ├── bar_column_plots.py                     
│   ├── violin_plots.py                         
│   ├── line_chart.py                         
│   └── area_chart.py                           
├── data/
│   └── owid-energy-raw-data.csv               
├── preprocessing/
│   ├── owid_energy_visualization_cleaned.csv  
│   └── preprocessing.ipynb                    
├── README.md                               
└── requirements.txt                            
```

---

##  Dataset Description

| Property | Details |
|---|---|
| **Name** | Our World in Data — Energy Dataset |
| **Source** | https://github.com/owid/energy-data |
| **Raw File** | `owid-energy-raw-data.csv` |
| **Cleaned File** | `owid_energy_visualization_cleaned.csv` |
| **Records** | 23,377 country-year rows |
| **Entities** | 314 (countries + regional aggregates) |
| **Time Range** | 1900 – 2025 |
| **Original Columns** | 120+ |

### Cleaned Dataset Columns

| Column | Description |
|---|---|
| `country` | Country name (title-cased, whitespace-stripped) |
| `year` | Year of the record |
| `population` | Population of the country |
| `gdp` | Gross Domestic Product (USD) |
| `coal_consumption` | Coal energy consumption (TWh) |
| `oil_consumption` | Oil energy consumption (TWh) |
| `gas_consumption` | Natural gas energy consumption (TWh) |
| `solar_consumption` | Solar energy consumption (TWh) |
| `wind_consumption` | Wind energy consumption (TWh) |
| `renewables_consumption` | Total renewables consumption (TWh) |
| `fossil_fuel_consumption` | Total fossil fuel consumption (TWh) |
| `electricity_generation` | Total electricity generated (TWh) |
| `decade` | Decade bucket derived from year (e.g. 1990, 2000) |

---

##  Charts Implemented

All chart types required by the course (Weeks 1–9) are included:

| Week | Chart Type | Dashboard Usage |
|---|---|---|
| Week 1 | Column Chart & Bar Chart | Top countries by total energy consumption |
| Week 2 | Stacked / Clustered Column & Bar | Energy mix breakdown (coal, oil, gas, solar, wind) per country |
| Week 3 | Scatter Chart | GDP vs. electricity generation relationship |
| Week 4 | Bubble Chart | GDP vs. fossil fuel consumption, sized by population |
| Week 5 | Histogram | Distribution of renewables consumption across countries |
| Week 6 | Box Chart | Energy metric spread across decades |
| Week 7 | Violin Chart | Distribution shape of renewable vs. fossil fuel consumption by decade |
| Week 8 | Line Chart | Energy consumption trends over time per country |
| Week 9 | Area Chart | Stacked renewable vs. fossil fuel share over time |

---

##  Interactive Elements

The dashboard includes the following interactive controls (≥ 3 as required):

1. **Country Dropdown** — Select one or multiple countries to filter all charts
2. **Year Range Slider** — Adjust the time window for time-series charts
3. **Energy Source Checklist** — Toggle specific energy sources (coal, oil, gas, solar, wind, renewables)
4. **Decade Radio Buttons** — Filter distribution and comparison charts by decade

All controls are wired to charts via **Dash callbacks** for dynamic, real-time updates.

---

##  How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Launch the Dashboard

```bash
python app.py
```

### 3. Open in Browser

```
http://127.0.0.1:8050/
```

---

##  Requirements

```
dash>=2.14.0
plotly>=5.18.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
```

> Full list available in `requirements.txt`

---

##  Data Preprocessing Summary

The full preprocessing steps are documented in `notebooks/preprocessing.ipynb`. The steps performed are:

1. **Load** raw dataset from `owid-energy-raw-data.csv` (120+ columns)
2. **Inspect** shape, column names, data types, and descriptive statistics
3. **Analyze missing values** — computed missing count and percentage per column, sorted by severity
4. **Drop high-missing columns** — removed all columns with more than **70% missing values**
5. **Fill remaining missing values**:
   - Numeric columns → filled with **column median**
   - Categorical columns → filled with **column mode**
6. **Remove duplicate rows**
7. **Filter to countries only** — kept rows where `iso_code` is not null (removes continental/regional aggregates)
8. **Select important columns** — retained 12 energy and economic columns relevant to the dashboard
9. **Create `decade` column** — derived from `year` using integer division (`year // 10 * 10`)
10. **Clean country names** — stripped whitespace and applied title-case formatting
11. **Outlier detection** — identified outliers per numeric column using the IQR method (1.5× IQR fence); outliers were flagged and reviewed but not removed to preserve real-world extremes
12. **Export** final cleaned dataset as `owid_energy_visualization_cleaned.csv`

---




