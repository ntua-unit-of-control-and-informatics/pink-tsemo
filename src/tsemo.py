import numpy as np
from jaqpot_api_client.models.dataset import Dataset
from jaqpot_api_client.models.prediction_response import PredictionResponse
from summit.benchmarks import ExperimentalEmulator
from summit.domain import *
from summit.utils.dataset import DataSet
import pandas as pd

from summit import Runner
from summit.strategies import TSEMO


def handle_tsemo(dataset: Dataset) -> PredictionResponse:

    """## Load Data

    We now load in the data from past experiments, which we will use to train the emulator. Here, we import the data that we already have in the Summit package, but any data available in CSV format would work.
    """

    # Read in data into a DataSEt.
    ds = DataSet.read_csv("NitroBENZENE_DataMatrix_4O_2024_09_27_05_07.csv",)

    """## Domain
    
    We first need to create a :class:`~summit.domain.Domain`. A domain specifies the aspects of the reaction we will be optimizing. In optimization speak, these are the decision variables (those that are manipulated), constraints and objectives for a benchmark.
    """

    domain = Domain()

    """Above, we instantiate a new domain without any variables. Here, we are going to manipulate the catalyst, catalyst loading, base stoichiometry and temperature. Our objectives are to maximise yield and minimise turn over number (TON). We can use the increment operator `+=` to add variables to the domain. There are no constraints.
    
    ### Decision variables
    
    Below, we use `ContinuousVariable` to specify the rest of the decision variables. Each has `bounds`, which represent the minimum and maximum values of each variable.
    """

    # Decision variables

    lower_bound = np.floor(ds['R_T'].min())
    upper_bound = np.ceil(ds['R_T'].max())

    des_1 = "Reactor's temperature (C)"
    domain += ContinuousVariable(name="R_T", description=des_1, bounds=[lower_bound, upper_bound])

    lower_bound = np.floor(ds['ResTime'].min())
    upper_bound = np.ceil(ds['ResTime'].max())

    des_2 = "Residence time (hr)"
    domain += ContinuousVariable(
        name="ResTime", description=des_2, bounds=[lower_bound, upper_bound]
    )

    lower_bound = np.floor(ds['B2A_ratio'].min())
    upper_bound = np.ceil(ds['B2A_ratio'].max())

    des_3 = "Benzene to Acids solution ratio (w/w)"
    domain += ContinuousVariable(
        name="B2A_ratio", description=des_3, bounds=[lower_bound, upper_bound]
    )

    """### Objectives
    
    Finally, we specify the objectives. We use `ContinuousVariable` again, but set `is_objective` to `True` and specify whether to maximize (or minimize) each objective.
    """

    # Objectives

    # 'NB_Yield',  'DNB_Yield',  'CO2e',  'TOTALCOST'

    lower_bound = np.floor(ds['NB_Yield'].min())
    upper_bound = np.ceil(ds['NB_Yield'].max())

    des_4 = (
        "NitroBENZENE yield % (mol C₆H₅NO₂/mol C6H6 %)"
    )
    domain += ContinuousVariable(
        name="NB_Yield",
        description=des_4,
        bounds=[lower_bound, upper_bound],
        is_objective=True,
        maximize=True,
    )

    lower_bound = np.floor(ds['DNB_Yield'].min())
    upper_bound = np.ceil(ds['DNB_Yield'].max())

    des_5 = (
        "DinitroBENZENE yield (mol C₆H₄N₂O₄/hr)"
    )
    domain += ContinuousVariable(
        name="DNB_Yield",
        description=des_5,
        bounds=[lower_bound, upper_bound],
        is_objective=True,
        maximize=False,
    )

    lower_bound = np.floor(ds['CO2e'].min())
    upper_bound = np.ceil(ds['CO2e'].max())

    des_6 = "Emissions CO2e (kg CO2e/hr)"
    domain += ContinuousVariable(
        name="CO2e",
        description=des_6,
        bounds=[lower_bound, upper_bound],
        is_objective=True,
        maximize=False,
    )

    lower_bound = np.floor(ds['TOTALCOST'].min())
    upper_bound = np.ceil(ds['TOTALCOST'].max())
    # annualized total cost reduced to cost per kg nitrobenzene

    des_7 = "TOTAL COST  ($)"
    domain += ContinuousVariable(
        name="TOTALCOST",
        description=des_7,
        bounds=[lower_bound, upper_bound],
        is_objective=True,
        maximize=False,
    )

    """When working inside a Jupyter Notebook, we can view the domain by putting it at the end of a cell and pressing enter.
    
    #### Domain view
    """

    domain

    """Note that we are using a :class:`~summit.dataset.Dataset`. In the CSV, it is essential that the columns match the domain and an extra row is added below each column name with the word DATA (see [here](https://github.com/sustainable-processes/summit/blob/master/summit/benchmarks/data/reizman_suzuki_case_1.csv) for an example).
    
    ## Train the Emulator
    
    Now we only need two lines to train the experimental emulator!  We first instantiate `ExperimentalEmulator` passing in the dataset, domain and a name for the model.  Next we train it with two-fold [cross-validation](https://machinelearningmastery.com/k-fold-cross-validation/) and a test set size of 25%.
    
    This step will take some time. Change verbose to 1 if you want streaming updates of the training.
    """

    emul = ExperimentalEmulator(model_name='my_reizman', domain=domain, dataset=ds)
    emul.train(max_epochs=1000, cv_fold=2, test_size=0.1, verbose=0)

    """The training returns a `scores` dictionary from [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate), which contains the results from each cross-validation fold. It might be difficult to understand these scores, so we show some more intuitive evaluation methods next.
    
    ## Evaluate Emulator
    
    A [parity plot](https://en.wikipedia.org/wiki/Parity_plot) shows experimental data against model predictions. We can do this for both the train and test sets. The $r^2$ score is shown, which varies between 0 and 1 with 1 being perfect fit.
    """

    fig, ax = emul.parity_plot(include_test=True)

    """## Run TSEMO"""

    exp=emul


    strategy = TSEMO(exp.domain, transform=None)

    # Use the runner to run closed loop optimisation
    r = Runner(
        strategy=strategy, experiment=exp,max_iterations=50
    )
    r.run()

    """## Pareto plot
    
    have to run this!
    """

    # Plot the results
    fig, ax = exp.pareto_plot(colorbar=True)
    _ = ax.set_xlabel('NB_Yield % (mol/mol)')
    _ = ax.set_ylabel('DNB_Yield (mol/hr)')

    exp.pareto_data.round(2)

    par_eto = exp.pareto_data.round(2)
    type(par_eto)
    
    """## Save file"""

     # import pandas as pd
    # from datetime import datetime

    # utc_now = datetime.utcnow()

    # import pytz

    # # Define the timezone for Greece (Eastern European Time)
    # greek_tz = pytz.timezone('Europe/Athens')

    # # Convert UTC time to Greek time
    # greek_now = utc_now.astimezone(greek_tz)

    # formatted_time = greek_now.strftime('%Y_%m_%d_%H_%M')
    # csv_filename = f"NitroBENZENE_4O_pareto{formatted_time}.csv"

    # print(f"File name with Greek time: {csv_filename}")
    # par_eto.to_csv(csv_filename, index=False)

    # # Generate a download link for the CSV file
    # from google.colab import files
    # files.download(csv_filename)
    
    """### 3D Pareto Plot
    
    ### Heat Map
    
    ### Heat Map with 3 objectives in x, y and z and a 4th objective with colors
    """

    # Import necessary libraries
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.colors import Normalize

    # Load your CSV file, skipping rows that are not needed
    #df = pd.DataFrame(dataset.input)
    df = par_eto


    # Load the CSV file, skipping the first two rows and selecting columns 4, 5, 6, and 7
    # data = np.loadtxt('NitroBENZENE_4O_pareto2024_11_27_16_23.csv', delimiter=',', skiprows=2, usecols=(3, 4, 5, 6))
    data = par_eto


    # Create the ndarray
    Obj4_pareto_ndarray = np.array(data)

    # Print the ndarray to verify
    print(Obj4_pareto_ndarray)

    T_Rs_Rt_dummy=Obj4_pareto_ndarray

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Extract columns for plotting
    x = T_Rs_Rt_dummy[:, 0]
    y = T_Rs_Rt_dummy[:, 1]
    z = T_Rs_Rt_dummy[:, 2]
    w = T_Rs_Rt_dummy[:, 3]

    # Create 3D scatter plot with heat map
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(x, y, z, c=w, cmap='jet')

    # Add color bar which maps values to colors
    colorbar = plt.colorbar(sc, pad=0.1)  # Adjust pad to move color bar away from the plot
    colorbar.set_label('TOTALCOST ($)')

    # Set labels
    ax.set_xlabel('NB_Yield % (mol NB/mol C6H6)')
    ax.set_ylabel('DNB_Mol_Yield (mol DNB/hr)')
    ax.set_zlabel('CO2e (kg/hr)')
    ax.set_title('3D Plot with Heat Map for Total Cost ($)')

    # Adjust layout to ensure labels are fully visible
    plt.tight_layout()

    plt.show()

    """# END"""
