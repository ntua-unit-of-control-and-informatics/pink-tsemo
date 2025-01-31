**README**

**Abstract**

This algorithm is designed to optimize the reaction conditions for a chemical production process. It utilizes a multi-objective optimization strategy, specifically the TSEMO (Thompson Sampling Efficient Multi-objective Optimization) algorithm, to maximize yield and minimize undesirable by-products and costs. The algorithm reads input data, defines decision variables and objectives, trains an experimental emulator, and performs optimization to identify the best reaction conditions and process parameters.

**Input File Description**

The input file Input\_file.csv (NitroBENZENE\_DataMatrix\_4O\_2024\_12\_27\_22\_54\_vars\_units

.csv for the example) contains experimental data for the optimization process. The file is structured as follows:

* **First Row**: Column headers indicating the variables.

* **Second Row**: Contains the word "DATA" in each cell, signifying the start of the data section.

* **Subsequent Rows**: Experimental data for each variable.

* **Second Last Row**: Divides the set of variables into decision variables and objectives. Cells contain either "decision" or the optimization direction ("maximize" or "minimize").

* **Last Row**: Descriptions of each variable.

**Example of Input Data (CSV Format)**

NAME,R\_T,ResTime,Benz2AcidRatio,NByield,DNByield,CO2e,TOTALCOST

DATA,DATA,DATA,DATA,DATA,DATA,DATA,DATA

0,65,9.13008,0.786,60.52,0.00024,11.2974,15090.6

...

175,decision,decision,decision,maximize,minimize,minimize,minimize

176,Reactor Temperature (C),Residense time (hr),Benzen to acids ratio (w/w),NB\_Yield % (mol NB/mol C6H6),DNB\_Mol\_Yield (mol DNB/hr),CO2e (kg/hr),Total Cost ($)

**Algorithm Description**

The algorithm follows these steps:

1. **Setup and Data Reading**:

   * Install required packages.

   * Import necessary libraries.

   * Read the input CSV file into a DataFrame and a DataSet object.

2. **Domain Definition**:

   * Create a Domain object to specify decision variables and objectives.

   * Extract variable names and descriptions from the input file.

   * Define decision variables and their bounds.

   * Define objectives and their optimization directions (maximize or minimize).

3. **Training the Emulator**:

   * Instantiate an ExperimentalEmulator with the dataset and domain.

   * Train the emulator using cross-validation and a test set.

4. **Optimization**:

   * Instantiate the TSEMO strategy.

   * Use the Runner to perform closed-loop optimization with the emulator.

   * Run the optimization for a specified number of iterations.

5. **Results and Visualization**:

   * Generate a parity plot to evaluate the emulator's performance.

   * Generate Pareto plots to visualize the trade-offs between objectives.

   * For four objectives, create a 3D scatter plot with a heat map.

**Results**

The results produced by the algorithm include:

* **Parity Plot**: Shows experimental data against model predictions with an $r^2$ score.

* **Pareto Plot**: Visualizes the trade-offs between objectives, highlighting the optimal solutions.

* **3D Scatter Plot with Heat Map**: For four objectives, visualizes the relationships between three objectives in 3D space with the fourth objective represented by color.

These visualizations help in understanding the performance of the emulator and the optimization process, providing insights into the best reaction conditions for the chemical production process.

