## CellMigrationGym:An Open Deep Reinforcement Learning Framework for Cell Migration Stud

# Introduction <br />
This study presents an open framework, CellMigrationGym, to standardize  the DRL approach to study cell migration and infer the underlying biology. Built upon common packages (OpenAI Gym, PyBullet, and DRL libraries), the CellMigrationGym provides powerful and flexible functions to investigate cell migration behavior. This study also presents a demonstration of CellMigrationGym to investigate a representative cell migration behavior, namely, Cpaaa intercalation. The demonstration reveals technical details of the framework setup, and most importantly, it also displays valuable functions of CellMigrationGym for cell migration study, such as 1) multiple observational data preparation and standardization, 2) different migration mechanisms exploration (such as gradient-driven and collective cell movements associated), and 3) evaluation of neighboring cell’s influence on the cell migration.

# Package Requirements <br />
gym                       0.18.0 <br />
pybullet                  3.0.8<br />
pytorch                   0.4.1 <br />
numpy                     1.19.4<br />
matplotlib                3.3.3<br />

# File structure<br />
./DRL:folder for DRL related model and data<br />
--./DRL/Embryo: environment for DRL.<br />
--./DRL/saved_data/: folder that used for saving the output data when exploring the successful scenarios for DRL.<br />
--./DRL/trained_models/:folder with all the pre-trained DRL models.<br />
----./DRL/trained_models/drl_model.pkl: checkpoint of the trained DQN<br />
----./DRL/trained_models/neighbor_model.pkl: checkpoint of the trained Neighbor Relationship Model<br />

./HDRL:folder for DRL related model and data.<br />
--./HDRL/Embryo: environment for HDRL.<br />
--./HDRL/saved_data/: folder that used for saving the output data when exploring the successful scenarios for HDRL.<br />
--./HDRL/trained_models/:folder with all the pre-trained HDRL models.<br />
----./HDRL/trained_models/hdrl_llmodel.pkl: checkpoint of the trained lower-level HDQN.<br />
----./HDRL/trained_models/hdrl_hlmodel.pkl: checkpoint of the trained Higher-level HDQN.<br />
----./HDRL/trained_models/neighbor_model.pkl: checkpoint of the trained Neighbor Relationship Model.<br />

./data/: folder with textual data of nuclei.<br />
--./data/data_description.txt: a brief description of the input textual embryonic data.<br />
--./data/Cpaaa_[0-2]: textual embryonic data of nuclei for Cpaaa migration training and evaluation.<br />

./observation:data with textual data of nuclei for observational purpose.<br />
--./observation/embryo_data/[01-06]:textual embryonic data of nuclei for observational purpose.<br />
--./observation/saved_data:processed observational data for analysis.<br />
--./observation/observation_analysis.ipynb:observational data analysis for figure 6.<br />

./simulation_data_visualization.ipynb: simulation data analysis for figure 7, 8 and 10.<br />

# Usage <br />
Explore the successful scenarios with DRL: Command: ```python3 ./DRL/simulation.py --em [0-2]```<br />
Explore the successful scenarios with HDRL: Command: ```python3 ./HDRL/simulation.py --em [0-2]```<br />

Two Files are generated in the 'saved_data' folder after the evaluation:<br />
**cpaaa_locations.pkl**: location of Cpaaa at each time step.<br />
**target_locations.pkl**: location of the target cell (ABarpaapp) at each time step.<br />

Two pickle files generated for each run above can be used for data analysis and visualization for figure 7,8 and 10.

# Citation <br />
Will update after paper submission.
