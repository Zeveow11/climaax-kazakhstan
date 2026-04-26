# Climate Risk Assessment for Kazakhstan

An adaptation of the [CLIMAAX CRA Handbook](https://github.com/CLIMAAX/crabook) 
for Kazakhstan, replacing European-specific datasets with globally available 
alternatives while maintaining the original scientific methodology.

## About

This repository provides Jupyter notebook workflows for climate risk assessment 
across multiple hazards relevant to Kazakhstan, including droughts, heatwaves, 
river floods, wildfire, and extreme precipitation.

The risk assessment follows the **Risk = Hazard × Exposure × Vulnerability** 
framework from the original CLIMAAX project, adapted for Kazakhstan's 
administrative structure (oblasts).

## Data Sources

| Component | Original CLIMAAX | This Repository |
|-----------|-----------------|-----------------|
| Administrative boundaries | NUTS (Europe) | Kazakhstan OSM boundaries |
| Climate data | E-OBS | ISIMIP3b (SSP1-2.6, SSP3-7.0) |
| Population exposure | Eurostat | WorldPop |
| Development index | Eurostat | Global Data Lab subnational HDI |
| Agricultural data | JRC | Kazakhstan Statistics Committee |

## Workflows

- ✅ Relative Drought Risk Assessment
- 🔜 Heatwave Assessment
- 🔜 River Flood Assessment
- 🔜 Wildfire Assessment
- 🔜 Extreme Precipitation Assessment

## Attribution

This repository is adapted from the CLIMAAX CRA Handbook:

> CLIMAAX (2024). *CLIMAAX CRA Handbook* [Software].  
> https://doi.org/10.5281/zenodo.18186357  
> Licensed under Apache-2.0 OR CC-BY-4.0

## License

Apache-2.0 OR CC-BY-4.0