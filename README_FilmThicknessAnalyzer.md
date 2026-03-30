# Film Thickness Analysis Dashboard

A Dash application for analyzing film thickness data on 300mm diameter wafers with interactive visualizations.

## Features

### Interactive Visualizations
- **Contour Plots**: 2D contour maps showing film thickness distribution across the wafer surface (±150mm axes)
- **Radial Profiles**: Spline plots from center to edge (0-150mm radius) showing thickness variation
- **Multi-Condition Comparison**: Dropdown selection to analyze single or multiple conditions simultaneously

### Analysis Zones
- **Center Zone**: 0-50mm radius
- **Mid Zone**: 50-100mm radius  
- **Edge Zone**: 100-150mm radius

### Summary Statistics
- Mean thickness and standard deviation for each condition
- Zone-wise standard deviation analysis
- Detailed breakdown by zone with min/max values

## Installation & Setup

1. Make sure your virtual environment is activated:
   ```bash
   .\venv\Scripts\Activate.ps1
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

### Step-by-Step Terminal Commands

1. **Activate the Python 3.11 virtual environment**:
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```
   
   You should see `(venv)` at the beginning of your prompt, indicating the environment is active.

2. **Verify Python version** (optional):
   ```bash
   python --version
   ```
   Should show: `Python 3.11.6`

3. **Run the application**:
   ```bash
   python film_thickness_analyzer.py
   ```
   
   You should see output like:
   ```
   Dash is running on http://127.0.0.1:8050/
   * Debug mode: on
   ```

4. **Open your browser** and navigate to: `http://127.0.0.1:8050`

5. **To stop the app**: Press `Ctrl+C` in the terminal

### Troubleshooting
- If packages aren't found, make sure the venv is activated (step 1)
- If you see import errors, reinstall packages: `pip install -r requirements.txt`

## Data Structure

The app expects:
- `BTM/condition-table.txt`: Tab-separated file with WaferID and process conditions
- `BTM/LN1718SS07-disp-cond.csv`: Measurement data with coordinates and thickness values

### Expected Columns
- **Condition Table**: WaferID, DispT, PumpT, DispSS, RlxT, RlxSS, Cast, Other
- **Measurement Data**: WaferID, Point No, Film Thickness, Fit Rate, X[mm], Y[mm]

## Usage

1. **Select Conditions**: Use the dropdown to select one or multiple process conditions
2. **Contour Tab**: View 2D thickness distribution maps with measurement points overlay
3. **Radial Profile Tab**: Compare thickness profiles from center to edge across conditions
4. **Summary Table Tab**: View statistical analysis including zone-wise standard deviations

## Wafer Specifications
- Diameter: 300mm (±150mm coordinate range)
- Measurement coordinate system: X,Y in millimeters
- Radial analysis: 0-150mm from center