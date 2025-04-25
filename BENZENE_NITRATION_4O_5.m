clear; clc; close all

%% LOAD ASPEN SIMULATION
% Aspen = actxserver('Apwn.Document.39.0');
Aspen = actxserver('Apwn.Document.40.0');
[stat,mess] = fileattrib;
% Get the current directory
currentDir = pwd;
% Construct the path to the Aspen Plus file
bkpFile = fullfile(currentDir, 'NITRATION_BENZENE_v16_20241223.bkp');
% Call Aspen Plus with the constructed file path
Aspen.invoke('InitFromArchive2', bkpFile);

Aspen.Visible = 1;
Aspen.SuppressDialogs = 1;

% stop
%% INPUT VARIABLES
% RTemp = [45:10:95];
% RVol = [1:2:30];
% AcidsMF = [800:00:2000];

RTemp = [45:25:95];
RVol = [1:10:30]*1000;
AcidsMF = [800:400:2000];

RTemp = [95];
RVol = 11000;
AcidsMF =  1200;

RTemp = [65:5:95];
RVol = [10000:1000:14000];
AcidsMF = [800:100:1200];


% initializations
T_Rs_Rt = [];
BENZENEin=[];
moleBENZENEin=[];

NitroBENZENE = [];
Mol_NitroBENZENE=[];

DiNitroBENZENE = [];
Mol_DiNitroBENZENE=[];

NByield = [];
DNByield =[];

NB_Mol_Percentage_Yield=[];
DNB_Mol_Percentage_Yield=[];

CO2e =[];
TOTALCOST = [];

No_of_pairs = length(RTemp)*length(RVol)*length(AcidsMF)
% stop

for i=1:length(RTemp)
    Aspen.Tree.FindNode("\Data\Blocks\RCSTR\Input\TEMP").Value  = RTemp(i); % Reactor temperature

    for j = 1:length(RVol)
        Aspen.Tree.FindNode("\Data\Blocks\RCSTR\Input\VOL").Value  = RVol(j); % Reactor volume

        for k=1:length(AcidsMF)
            Aspen.Tree.FindNode("\Data\Streams\MIXACID\Input\TOTFLOW\MIXED").Value = AcidsMF(k);

            Aspen.Reinit;
            Aspen.Engine.Run2(1);
            while Aspen.Engine.IsRunning == 1
                % pause(0.5);
                pause(5);
            end

            ResTime(j) = Aspen.Tree.FindNode("\Data\Blocks\RCSTR\Output\TOT_RES_TIME").Value; % Residence time
            Benz2AcidRatio(k) = Aspen.Tree.FindNode("\Data\Flowsheeting Options\Calculator\BNZACDRT\Output\WRITE_VAL\7").Value; % Benzene to Acid volumetric flow ratio   

            T_Rs_Rt = [T_Rs_Rt; RTemp(i) ResTime(j)  Benz2AcidRatio(k)];
            i_j_k = [i j k]

            BENZENEin = [BENZENEin; Aspen.Tree.FindNode("\Data\Streams\C6H6\Input\TOTFLOW\MIXED").Value];
            moleBENZENEin = [moleBENZENEin; Aspen.Tree.FindNode("\Data\Streams\C6H6\Output\MOLEFLOW\MIXED\C6H6").Value];
            

            NitroBENZENE = [NitroBENZENE; Aspen.Tree.FindNode("\Data\Streams\BOTOMRAD\Output\MASSFLOW\MIXED\C6H5NO2").Value];
            DiNitroBENZENE = [DiNitroBENZENE; Aspen.Tree.FindNode("\Data\Streams\BOTOMRAD\Output\MASSFLOW\MIXED\M-DIN-01").Value];

            Mol_NitroBENZENE = [Mol_NitroBENZENE; Aspen.Tree.FindNode("\Data\Streams\BOTOMRAD\Output\MOLEFLOW\MIXED\C6H5NO2").Value];
            Mol_DiNitroBENZENE = [Mol_DiNitroBENZENE; Aspen.Tree.FindNode("\Data\Streams\BOTOMRAD\Output\MOLEFLOW\MIXED\M-DIN-01").Value];

            NByield = [NByield; NitroBENZENE(end)/BENZENEin(end)];
            DNByield = [DNByield; DiNitroBENZENE(end)/BENZENEin(end)];

            NB_Mol_Percentage_Yield =[NB_Mol_Percentage_Yield; Mol_NitroBENZENE(end)/moleBENZENEin(end)*100];
            DNB_Mol_Percentage_Yield=[DNB_Mol_Percentage_Yield; Mol_DiNitroBENZENE(end)];

            % CO2e 
            CO2e  = [CO2e;  Aspen.Tree.FindNode("\Data\Results Summary\Utility-Sum\Output\HTOTCO2E\PLANT").Value];
            % Annualized cost 
            TOTALCOST = [TOTALCOST; Aspen.Tree.FindNode("\Data\Flowsheeting Options\Calculator\TOTALCST\Output\WRITE_VAL\15").Value];% annualized total cost 
        end
    end
end

Aspen.Close; Aspen.Quit;

ALL_DATA = [[0:No_of_pairs-1]' T_Rs_Rt(:,1) T_Rs_Rt(:,2) T_Rs_Rt(:,3) NB_Mol_Percentage_Yield DNB_Mol_Percentage_Yield CO2e TOTALCOST];

minValues = min(ALL_DATA);
maxValues = max(ALL_DATA);

%% DataCellArray .csv save
% Existing cell array
dataCellArray = {'NAME'  'R_T',    'ResTime',  'B2A_ratio',  'NB_Yield',  'DNB_Yield',  'CO2e',  'TOTALCOST'; ...
                            'TYPE'  'DATA',  'DATA',       'DATA',         'DATA',        'DATA',          'DATA',   'DATA'; ...
    };

% Convert the numeric matrix to a cell array
newCellArray = num2cell(ALL_DATA);

dataCellArray2 = {'decision'	'decision'	'decision'	'maximize'	'minimize'	'minimize'	'minimize'; ...
'Reactor Temperature (C)'	'Residense time (hr)'	'Benzen to acids ratio (w/w)'	'NB_Yield % (mol NB/mol C6H6)'	'DNB_Mol_Yield (mol DNB/hr)'	'CO2e (kg/hr)'	'Total Cost ($)'; ...
    };

% dataCellArray2 = [[size(newCellArray,1) + 1; size(newCellArray,1) + 2] [ 1 2;3 4]];

% Append the new cell array below the existing dataCellArray
combinedCellArray = [dataCellArray; newCellArray; num2cell([size(newCellArray,1) + 1; size(newCellArray,1) + 2]) dataCellArray2];

% Get current date and time
currentDateTime = datestr(now, 'yyyy_mm_dd_HH_MM');

% Create the filename with the current date and time appended
filename = ['NitroBENZENE_DataMatrix_4O_' currentDateTime '.csv'];

save 

% Write the combined cell array to a CSV file
writecell(combinedCellArray, filename);

% Display a message confirming the data has been saved
disp(['Data successfully saved to ', filename]);


% % % SCALING
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Preallocate the scaled matrix
% ALL_DATA_scaled = zeros(size(ALL_DATA));
%
% % Calculate the min and max for each column
% min_vals = min(ALL_DATA);
% max_vals = max(ALL_DATA);
%
% % Scale each column to the [0, 1] range
% for i = 1:size(ALL_DATA, 2)
%     ALL_DATA_scaled(:, i) = (ALL_DATA(:, i) - min_vals(i)) / (max_vals(i) - min_vals(i));
% end
%
% % Display the scaled matrix
% disp(ALL_DATA_scaled);
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Preallocate the scaled matrix
% CO2e_scaled = zeros(size(CO2e));
% BENZENE_scaled = zeros(size(BENZENE));
% TOTALCOST_scaled = zeros(size(TOTALCOST));
%
% % Calculate the min and max for each column
% min_vals_CO2e = min(CO2e);
% max_vals_CO2e = max(CO2e);
%
% min_vals_BENZENE = min(BENZENE);
% max_vals_BENZENE = max(BENZENE);
%
% min_vals_TOTALCOST = min(TOTALCOST);
% max_vals_TOTALCOST = max(TOTALCOST);
%
% % Scale each column to the [0, 1] range
%     CO2e_scaled  = (CO2e  - min_vals_CO2e ) / (max_vals_CO2e  - min_vals_CO2e );
%     BENZENE_scaled  = (BENZENE  - min_vals_BENZENE ) / (max_vals_BENZENE  - min_vals_BENZENE );
%     TOTALCOST_scaled  = (TOTALCOST  - min_vals_TOTALCOST ) / (max_vals_TOTALCOST  - min_vals_TOTALCOST );
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ALL_DATA = ALL_DATA_scaled;
% CO2e = CO2e_scaled;
% BENZENE = BENZENE_scaled;
% TOTALCOST = TOTALCOST_scaled;

%% PLOTS
% Extract columns for plotting
x = T_Rs_Rt(:, 1);
y = T_Rs_Rt(:, 2);
z = T_Rs_Rt(:, 3);

% Create 3D plot
figure;
scatter3(x, y, z, 'filled');
xlabel('RTemp (°C)');
ylabel('ResTime (hr)');
zlabel('Benz2AcidRatio');
title('3D Plot of Temperature, Residence Time, and Benzene/Acids solution volumetric flow Ratio');
grid on;

'new PLOTS'
pause

%% OLD plots
close all
% Plotting ALL_DATA in a subplot
subplot(2,2,1);
plot(ALL_DATA(:,1), ALL_DATA(:,2), 'k*');
xlabel('T (C)');
ylabel('Res Time (hr)');
title('ALL_DATA Plot');

% Plotting CO2e vs BENZENE in a subplot
subplot(2,2,2);
plot(NByield, DNByield, 'o');
xlabel('NByield');
ylabel('DNByield');
title('CO2e vs BENZENE Plot');

% 3D plot for P, T, and TOTALCOST in a subplot with heat map
subplot(2,2,3);
[X, Y] = meshgrid(ALL_DATA(:,1), ALL_DATA(:,2)); % Create a meshgrid for the surface plot
Z = griddata(ALL_DATA(:,1), ALL_DATA(:,2), TOTALCOST, X, Y); % Interpolate the TOTALCOST data
surf(X, Y, Z); % Create a surface plot
xlabel('T (C)');
ylabel('Res Time (hr)');
zlabel('TOTALCOST ($/item)');
title('Heat Map of P, T, and TOTALCOST');

% 3D plot for CO2e, BENZENE, and TOTALCOST in a subplot with heat map
subplot(2,2,4);
plot3(CO2e, NByield, TOTALCOST, 'o');
xlabel('CO2e (kg/hr)');
ylabel('NByield)');
zlabel('TOTALCOST ($/item)');
title('  ');
grid on


pause
close all
plot3(CO2e, NByield, TOTALCOST, 'o');
xlabel('CO2e (kg/hr)');
ylabel('NByield)');
zlabel('TOTALCOST ($/item)');
title('Heat Map of TOTALCOST vs CO2e and NByield');
grid on

pause
close all

plot(ALL_DATA(:,3), TOTALCOST, 'k*');
xlabel('ResTime (hr)');
ylabel('TOTALCOST $');
title('Cost vs ResTime');

%% DataCellArray .csv save
% Existing cell array
% dataCellArray = {'NAME' 'R_T', 'ResTime', 'Benz2AcidRatio', 'CO2e', 'NByield', 'DNByield', 'CO2e', 'TOTALCOST'; ...
%     'TYPE' 'DATA', 'DATA', 'DATA', 'DATA', 'DATA', 'DATA', 'DATA', 'DATA'; ...
%     };

dataCellArray = {'NAME' 'R_T', 'ResTime', 'Benz2AcidRatio', 'NByield', 'DNByield', 'CO2e', 'TOTALCOST'; ...
    'TYPE' 'DATA', 'DATA', 'DATA', 'DATA', 'DATA', 'DATA', 'DATA'; ...
    };


% Convert the numeric matrix to a cell array
newCellArray = num2cell(ALL_DATA);

% Append the new cell array below the existing dataCellArray
combinedCellArray = [dataCellArray; newCellArray];

% Get current date and time
currentDateTime = datestr(now, 'yyyy_mm_dd_HH_MM');

% Create the filename with the current date and time appended
filename = ['NitroBENZENE_DataMatrix_4O_' currentDateTime '.csv'];

save

% Write the combined cell array to a CSV file
writecell(combinedCellArray, filename);

% Display a message confirming the data has been saved
disp(['Data successfully saved to ', filename]);

