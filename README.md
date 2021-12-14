## CellMigrationGym: An Open Deep Reinforcement Learning Framework for Cell Migration Study

# Introduction <br />
This study presents an open framework, CellMigrationGym, to standardize  the DRL approach to study cell migration and infer the underlying biology. Built upon common packages (OpenAI Gym, PyBullet, and DRL libraries), the CellMigrationGym provides powerful and flexible functions to investigate cell migration behavior. This study also presents a demonstration of CellMigrationGym to investigate a representative cell migration behavior, namely, Cpaaa intercalation. The demonstration reveals technical details of the framework setup, and most importantly, it also displays valuable functions of CellMigrationGym for cell migration study, such as 1) multiple observational data preparation and standardization, 2) different migration mechanisms exploration (such as gradient-driven and collective cell movements associated), and 3) evaluation of neighboring cell’s influence on the cell migration.

# Package Requirements <br />
gym                       0.18.0<br />
pybullet                  3.0.8<br />
pytorch                   0.4.1 <br />
numpy                     1.19.4<br />
matplotlib                3.3.3<br />
scikit-learn              0.22<br />

# File structure<br />
./DRL: **Folder**  for DRL related model and data<br />
-- ./Embryo: Environment for DRL.<br />
-- ./saved_data/: **Folder**  that used for saving the output data when exploring the successful scenarios for DRL. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Currently stored npy file are used for figure 7.<br />
-- ./trained_models/: **Folder**  with all the pre-trained DRL models.<br />
---- drl_model.pkl: DRL model of the trained DQN<br />
---- neighbor_model.pkl: Trained Neighbor Relationship Model<br />

./HDRL: **Folder** for DRL related model and data.<br />
--./Embryo: Environment for HDRL.<br />
--./saved_data/: **Folder**  that used for saving the output data when exploring the successful scenarios for HDRL. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Currently stored npy file are used for figure 8.<br />
--./trained_models/:**Folder**  with all the pre-trained HDRL models.<br />
---- hdrl_llmodel.pkl: Trained lower-level HDQN model.<br />
---- hdrl_hlmodel.pkl: Trained  Higher-level HDQN model.<br />
---- neighbor_model.pkl: Trained Neighbor Relationship Model.<br />

./data/: **Folder**  with textual data of nuclei.<br />
-- ./Cpaaa_[0-2]: Textual embryonic data of nuclei for Cpaaa migration training and evaluation.<br />
-- data_description.txt: A brief description of the input textual embryonic data.<br />

./observation: Data with textual data of nuclei for observational purpose.<br />
-- ./embryo_data/[01-06]: Textual embryonic data of nuclei for observational purpose.<br />
-- ./saved_data: Processed observational data for analysis.<br />
-- observation_analysis.ipynb: Observational data analysis for figure 6.<br />

simulation_data_visualization.ipynb: Simulation data analysis for figure 7, 8 and 10.<br />

# Usage <br />
Explore the successful scenarios with DRL: Command: ```python3 ./DRL/simulation.py --em [0-2]```<br />
Explore the successful scenarios with HDRL: Command: ```python3 ./HDRL/simulation.py --em [0-2]```<br />

Two Files are generated in the 'saved_data' folder after the evaluation:<br />
**cpaaa_locations.pkl**: location of Cpaaa at each time step.<br />
**target_locations.pkl**: location of the target cell (ABarpaapp) at each time step.<br />

Two pickle files generated for each run above can be used for data analysis and visualization for figure 7,8 and 10.

# FAQ <br />
Q: No module name 'sklearn.forest.ensemble' after installing scikit-learn.<br />
A: sklearn.ensemble.forest was renamed to sklearn.ensemble._forest, Please install an older sklearn version: ```pip install scikit-learn==0.22```<br />

Q: ImportError: numpy.core.multiarray failed to import.<br />
A: Please try ```pip install numpy --upgrade```<br />

Q: Unable to install pytorch 0.4.1.<br />
A: Please try ```pip install torch==0.4.1```<br />




# Citation <br />
Will update after paper submission.
