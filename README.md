# PV Radiation vs. Generation  
*Quantifying the gap between solar potential (PVGIS) and realized generation (ANEEL) with loss-inclusive analysis.*

Key Features:
- 🌍 Geospatial Matching: Links ventures to nearest radiation coordinates via cKDTree

- 📉 Loss Modeling: Accounts for temperature, aging, and system inefficiencies

- ⚡ Parallelized Workflow: Uses multiprocessing for large datasets

- 📊 Dual Visualization: Plots hourly radiation vs. generation curves

### **Data Attribution** 
> **Data Sources**  
> - ANEEL photovoltaic ventures (via [cleaner tool](https://github.com/Mekepi/aneel-mmdg-photovoltaic-cleaner))  
> - PVGIS v5.3 radiation data  
> Full details in [DATA_SOURCES.md](DATA_SOURCES.md).  
